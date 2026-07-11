//! Circuit compiler: name resolution, cancellation, diagonal fusion and
//! general gate fusion.
//!
//! Pipeline:
//! 1. **cancel** — drop adjacent self-inverse pairs (`h h`, `cx cx`, …).
//! 2. **lower** — resolve gate names to matrices ([`crate::gates`]),
//!    permute every matrix to *sorted-qubit* convention.
//! 3. **fuse diagonals** — diagonal gates commute with each other and with
//!    anything on disjoint qubits, so runs of them collapse into a single
//!    table applied in one sweep (a QFT's controlled-phase ladder becomes
//!    one diagonal per layer).
//! 4. **fuse general** — greedily absorb adjacent gates while the combined
//!    support stays ≤ `fusion_max` qubits (default 5, like qiskit-aer).
//!    One fused 5-qubit gate = one memory sweep instead of many.
//! 5. **finalize** — split trailing measurements (sampled analytically)
//!    from dynamic circuits (measure/reset/conditionals mid-circuit → the
//!    executor re-runs per shot).

use crate::gates;
use crate::ir::{GateInstr, Instr, Program, Reg, C64};
use std::collections::HashSet;

/// A compiled operation, in sorted-qubit matrix convention.
#[derive(Debug, Clone)]
pub enum COp {
    Unitary {
        qubits: Vec<u32>,
        mat: Vec<C64>,
    },
    Diagonal {
        qubits: Vec<u32>,
        diag: Vec<C64>,
    },
    Measure {
        qubit: u32,
        clbit: u32,
    },
    Reset {
        qubit: u32,
    },
    If {
        creg_offset: u32,
        creg_len: u32,
        value: u64,
        inner: Box<COp>,
    },
    /// Fusion fence; stripped after the fusion passes, never executed.
    Barrier,
}

impl COp {
    fn support(&self) -> Option<&[u32]> {
        match self {
            COp::Unitary { qubits, .. } | COp::Diagonal { qubits, .. } => Some(qubits),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct Stats {
    /// Gate instructions in the source program (after macro expansion).
    pub input_gates: usize,
    /// Gates removed by the cancellation pass.
    pub cancelled: usize,
    /// Executable ops after all fusion passes.
    pub output_ops: usize,
    /// Largest fused-gate width (qubits).
    pub max_fused: usize,
}

#[derive(Debug, Clone)]
pub struct Compiled {
    pub n_qubits: u32,
    pub n_clbits: u32,
    pub cregs: Vec<Reg>,
    /// Body ops. For non-dynamic circuits this excludes the trailing
    /// measurements (see `final_measures`).
    pub ops: Vec<COp>,
    /// True if the circuit needs per-shot execution (mid-circuit
    /// measurement, reset, or classical control).
    pub dynamic: bool,
    /// Trailing `(qubit, clbit)` measurements of a non-dynamic circuit.
    pub final_measures: Vec<(u32, u32)>,
    pub stats: Stats,
}

#[derive(Debug, Clone, Copy)]
pub struct CompileOptions {
    /// Maximum *dense* fused-gate width in qubits (max 6). 0 disables all
    /// fusion (including diagonal fusion); any value ≥ 1 keeps diagonal
    /// fusion on. Executor default: 1 for the CPU backend, 5 for Metal —
    /// dense fusion trades memory sweeps for a compute-bound matmul, which
    /// wins on the GPU and loses on the CPU (measured on M4 Pro).
    pub fusion_max: u8,
    /// Maximum diagonal-fusion width in qubits (table is 2^k entries,
    /// hard cap 12).
    pub diag_max: u8,
}

impl Default for CompileOptions {
    fn default() -> Self {
        CompileOptions {
            fusion_max: 1,
            diag_max: 10,
        }
    }
}

pub fn compile(p: &Program, opt: &CompileOptions) -> Result<Compiled, crate::Error> {
    let n_qubits = p.n_qubits();
    let n_clbits = p.n_clbits();
    if n_qubits == 0 {
        return Err(crate::Error::InvalidCircuit("no qubits declared".into()));
    }
    if n_qubits > 48 {
        return Err(crate::Error::InvalidCircuit(format!(
            "{n_qubits} qubits is beyond any state-vector budget"
        )));
    }
    if n_clbits > 64 {
        return Err(crate::Error::InvalidCircuit(
            "more than 64 classical bits are not supported".into(),
        ));
    }

    let input_gates = count_gates(&p.instrs);
    let cancelled_instrs = cancel(&p.instrs);
    let cancelled = input_gates - count_gates(&cancelled_instrs);

    let mut ops = Vec::with_capacity(cancelled_instrs.len());
    for ins in &cancelled_instrs {
        lower(ins, p, &mut ops)?;
    }

    let opt = CompileOptions {
        fusion_max: opt.fusion_max.min(crate::state::MAX_FUSED_QUBITS as u8),
        diag_max: opt.diag_max.min(12),
    };
    // Diagonal fusion is a memory-bound win on every backend, so it runs
    // whenever fusion is enabled at all; `fusion_max` only bounds the
    // *dense* fusion width below.
    if opt.fusion_max >= 1 {
        ops = fuse_diagonals(ops, opt.diag_max as usize);
        ops = fuse_general(ops, opt.fusion_max as usize);
    }
    ops.retain(|op| !matches!(op, COp::Barrier));

    // Finalize: trailing measurements vs dynamic body.
    let mut split = ops.len();
    while split > 0 && matches!(ops[split - 1], COp::Measure { .. }) {
        split -= 1;
    }
    let body_dynamic = ops[..split]
        .iter()
        .any(|op| matches!(op, COp::Measure { .. } | COp::Reset { .. } | COp::If { .. }));

    let (ops, dynamic, final_measures) = if body_dynamic {
        (ops, true, vec![])
    } else {
        let final_measures: Vec<(u32, u32)> = ops[split..]
            .iter()
            .map(|op| match op {
                COp::Measure { qubit, clbit } => (*qubit, *clbit),
                _ => unreachable!(),
            })
            .collect();
        ops.truncate(split);
        (ops, false, final_measures)
    };

    let max_fused = ops
        .iter()
        .filter_map(|o| o.support().map(|s| s.len()))
        .max()
        .unwrap_or(0);

    Ok(Compiled {
        n_qubits,
        n_clbits,
        cregs: p.cregs.clone(),
        stats: Stats {
            input_gates,
            cancelled,
            output_ops: ops.len(),
            max_fused,
        },
        ops,
        dynamic,
        final_measures,
    })
}

fn count_gates(instrs: &[Instr]) -> usize {
    instrs
        .iter()
        .map(|i| match i {
            Instr::Gate(_) => 1,
            Instr::If { op, .. } => count_gates(std::slice::from_ref(op)),
            _ => 0,
        })
        .sum()
}

/// Drop adjacent self-inverse pairs acting on identical qubits, looking
/// through gates on disjoint qubits. Barriers and non-gate ops fence.
fn cancel(instrs: &[Instr]) -> Vec<Instr> {
    let mut out: Vec<Instr> = Vec::with_capacity(instrs.len());
    'next: for ins in instrs {
        if let Instr::Gate(g) = ins {
            if g.params.is_empty() && gates::SELF_INVERSE.contains(&g.name.as_str()) {
                for i in (0..out.len()).rev() {
                    match &out[i] {
                        Instr::Gate(pg) => {
                            if pg.qubits.iter().any(|q| g.qubits.contains(q)) {
                                if pg.name == g.name && pg.qubits == g.qubits {
                                    out.remove(i);
                                    continue 'next;
                                }
                                break;
                            }
                        }
                        _ => break,
                    }
                }
            }
        }
        out.push(ins.clone());
    }
    out
}

/// Resolve one instruction into `COp`s (sorted-qubit convention).
fn lower(ins: &Instr, p: &Program, out: &mut Vec<COp>) -> Result<(), crate::Error> {
    let n_qubits = p.n_qubits();
    let n_clbits = p.n_clbits();
    match ins {
        Instr::Gate(g) => {
            if let Some(op) = lower_gate(g, n_qubits)? {
                out.push(op);
            }
        }
        Instr::Measure { qubit, clbit } => {
            check_q(*qubit, n_qubits)?;
            if *clbit >= n_clbits {
                return Err(crate::Error::InvalidCircuit(format!(
                    "clbit {clbit} out of range (have {n_clbits})"
                )));
            }
            out.push(COp::Measure {
                qubit: *qubit,
                clbit: *clbit,
            });
        }
        Instr::Reset { qubit } => {
            check_q(*qubit, n_qubits)?;
            out.push(COp::Reset { qubit: *qubit });
        }
        Instr::Barrier(_) => out.push(COp::Barrier),
        Instr::If { creg, value, op } => {
            let reg = p.cregs.get(*creg).ok_or_else(|| {
                crate::Error::InvalidCircuit(format!("creg #{creg} out of range"))
            })?;
            let mut inner = Vec::with_capacity(1);
            lower(op, p, &mut inner)?;
            for i in inner {
                out.push(COp::If {
                    creg_offset: p.creg_offset(*creg),
                    creg_len: reg.size,
                    value: *value,
                    inner: Box::new(i),
                });
            }
        }
    }
    Ok(())
}

fn check_q(q: u32, n: u32) -> Result<(), crate::Error> {
    if q >= n {
        return Err(crate::Error::InvalidCircuit(format!(
            "qubit {q} out of range (have {n})"
        )));
    }
    Ok(())
}

fn lower_gate(g: &GateInstr, n_qubits: u32) -> Result<Option<COp>, crate::Error> {
    let def = gates::lookup(&g.name)
        .ok_or_else(|| crate::Error::InvalidCircuit(format!("unknown gate '{}'", g.name)))?;
    if g.qubits.len() != def.arity as usize {
        return Err(crate::Error::InvalidCircuit(format!(
            "gate '{}' expects {} qubit(s), got {}",
            g.name,
            def.arity,
            g.qubits.len()
        )));
    }
    for &q in &g.qubits {
        check_q(q, n_qubits)?;
    }
    let mut seen = HashSet::new();
    if !g.qubits.iter().all(|q| seen.insert(*q)) {
        return Err(crate::Error::InvalidCircuit(format!(
            "gate '{}' applied to duplicate qubits {:?}",
            g.name, g.qubits
        )));
    }
    if g.name == "id" || g.name == "u0" {
        return Ok(None);
    }
    let m = gates::build(&g.name, &g.params).ok_or_else(|| {
        crate::Error::InvalidCircuit(format!(
            "gate '{}' expects {} parameter(s), got {}",
            g.name,
            def.n_params,
            g.params.len()
        ))
    })?;
    Ok(Some(match m {
        gates::GateMatrix::Unitary(mat) => {
            let (qs, mat) = permute_unitary_to_sorted(&g.qubits, &mat);
            COp::Unitary { qubits: qs, mat }
        }
        gates::GateMatrix::Diagonal(d) => {
            let (qs, d) = permute_diag_to_sorted(&g.qubits, &d);
            COp::Diagonal {
                qubits: qs,
                diag: d,
            }
        }
    }))
}

/// Map a local index in sorted-qubit convention to the equivalent index in
/// the gate's argument-order convention.
#[inline]
fn arg_index(j_sorted: usize, pos_in_sorted: &[usize]) -> usize {
    let mut j = 0usize;
    for (b, &pos) in pos_in_sorted.iter().enumerate() {
        j |= ((j_sorted >> pos) & 1) << b;
    }
    j
}

fn sort_map(qubits: &[u32]) -> (Vec<u32>, Vec<usize>) {
    let mut sorted: Vec<u32> = qubits.to_vec();
    sorted.sort_unstable();
    let pos: Vec<usize> = qubits
        .iter()
        .map(|q| sorted.iter().position(|s| s == q).unwrap())
        .collect();
    (sorted, pos)
}

pub fn permute_unitary_to_sorted(qubits: &[u32], mat: &[C64]) -> (Vec<u32>, Vec<C64>) {
    let (sorted, pos) = sort_map(qubits);
    if sorted == qubits {
        return (sorted, mat.to_vec());
    }
    let dim = 1usize << qubits.len();
    let mut out = vec![C64::default(); dim * dim];
    for i in 0..dim {
        let ia = arg_index(i, &pos);
        for j in 0..dim {
            let ja = arg_index(j, &pos);
            out[i * dim + j] = mat[ia * dim + ja];
        }
    }
    (sorted, out)
}

pub fn permute_diag_to_sorted(qubits: &[u32], d: &[C64]) -> (Vec<u32>, Vec<C64>) {
    let (sorted, pos) = sort_map(qubits);
    if sorted == qubits {
        return (sorted, d.to_vec());
    }
    let dim = 1usize << qubits.len();
    let mut out = vec![C64::default(); dim];
    for j in 0..dim {
        out[j] = d[arg_index(j, &pos)];
    }
    (sorted, out)
}

fn union_sorted(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut u: Vec<u32> = a.iter().chain(b).copied().collect();
    u.sort_unstable();
    u.dedup();
    u
}

/// Extract, for each index over `to`, the corresponding index over the
/// subset `from` (both sorted).
#[inline]
fn subset_index(i_to: usize, from: &[u32], to: &[u32]) -> usize {
    let mut j = 0usize;
    for (b, &q) in from.iter().enumerate() {
        let pos = to.iter().position(|t| *t == q).unwrap();
        j |= ((i_to >> pos) & 1) << b;
    }
    j
}

/// Embed a unitary on `from` into the larger sorted support `to`
/// (identity on the extra qubits).
pub fn embed_unitary(mat: &[C64], from: &[u32], to: &[u32]) -> Vec<C64> {
    if from == to {
        return mat.to_vec();
    }
    let dim_to = 1usize << to.len();
    let dim_from = 1usize << from.len();
    // Positions of `from` qubits inside `to`, as a bitmask over to-indices.
    let mut from_mask = 0usize;
    for &q in from {
        let pos = to.iter().position(|t| *t == q).unwrap();
        from_mask |= 1 << pos;
    }
    let rest_mask = (dim_to - 1) & !from_mask;
    let mut out = vec![C64::default(); dim_to * dim_to];
    for i in 0..dim_to {
        let ifrom = subset_index(i, from, to);
        let irest = i & rest_mask;
        for j in 0..dim_to {
            if (j & rest_mask) != irest {
                continue;
            }
            let jfrom = subset_index(j, from, to);
            out[i * dim_to + j] = mat[ifrom * dim_from + jfrom];
        }
    }
    out
}

pub fn embed_diag(d: &[C64], from: &[u32], to: &[u32]) -> Vec<C64> {
    if from == to {
        return d.to_vec();
    }
    let dim_to = 1usize << to.len();
    (0..dim_to).map(|i| d[subset_index(i, from, to)]).collect()
}

fn matmul(a: &[C64], b: &[C64], dim: usize) -> Vec<C64> {
    let mut out = vec![C64::default(); dim * dim];
    for i in 0..dim {
        for l in 0..dim {
            let ail = a[i * dim + l];
            if ail == C64::default() {
                continue;
            }
            for j in 0..dim {
                out[i * dim + j] += ail * b[l * dim + j];
            }
        }
    }
    out
}

/// Merge runs of diagonal ops. A later diagonal is pulled into the current
/// group only if its support avoids every non-diagonal op skipped so far
/// (diagonals commute with each other unconditionally, and with anything
/// acting on disjoint qubits).
fn fuse_diagonals(ops: Vec<COp>, diag_max: usize) -> Vec<COp> {
    let mut out = Vec::with_capacity(ops.len());
    let mut cur: Option<(Vec<u32>, Vec<C64>, usize)> = None; // support, table, count
    let mut blocked: HashSet<u32> = HashSet::new();

    let flush = |cur: &mut Option<(Vec<u32>, Vec<C64>, usize)>, out: &mut Vec<COp>| {
        if let Some((qs, d, _)) = cur.take() {
            out.push(COp::Diagonal {
                qubits: qs,
                diag: d,
            });
        }
    };

    for op in ops {
        match &op {
            COp::Diagonal { qubits, diag } => {
                let clean = qubits.iter().all(|q| !blocked.contains(q));
                match (&mut cur, clean) {
                    (Some((sup, table, count)), true)
                        if union_sorted(sup, qubits).len() <= diag_max =>
                    {
                        let u = union_sorted(sup, qubits);
                        let a = embed_diag(table, sup, &u);
                        let b = embed_diag(diag, qubits, &u);
                        *table = a.iter().zip(&b).map(|(x, y)| x * y).collect();
                        *sup = u;
                        *count += 1;
                    }
                    _ => {
                        flush(&mut cur, &mut out);
                        blocked.clear();
                        cur = Some((qubits.clone(), diag.clone(), 1));
                    }
                }
            }
            COp::Unitary { qubits, .. } => {
                if let Some((sup, _, _)) = &cur {
                    if qubits.iter().any(|q| sup.contains(q)) {
                        // Group can't move past this op: emit it first.
                        flush(&mut cur, &mut out);
                        blocked.clear();
                    } else {
                        blocked.extend(qubits.iter().copied());
                    }
                }
                out.push(op);
            }
            _ => {
                flush(&mut cur, &mut out);
                blocked.clear();
                out.push(op);
            }
        }
    }
    flush(&mut cur, &mut out);
    out
}

/// Greedy adjacent fusion: absorb the next gate while the union support
/// stays within `kmax` qubits. Groups that end up with a single original
/// op are emitted unchanged.
fn fuse_general(ops: Vec<COp>, kmax: usize) -> Vec<COp> {
    struct Group {
        support: Vec<u32>,
        mat: Vec<C64>,
        count: usize,
        first: COp,
    }
    let mut out = Vec::with_capacity(ops.len());
    let mut cur: Option<Group> = None;

    let to_matrix = |op: &COp| -> (Vec<u32>, Vec<C64>) {
        match op {
            COp::Unitary { qubits, mat } => (qubits.clone(), mat.clone()),
            COp::Diagonal { qubits, diag } => {
                let dim = 1usize << qubits.len();
                let mut m = vec![C64::default(); dim * dim];
                for (j, v) in diag.iter().enumerate() {
                    m[j * dim + j] = *v;
                }
                (qubits.clone(), m)
            }
            _ => unreachable!(),
        }
    };

    let flush = |cur: &mut Option<Group>, out: &mut Vec<COp>| {
        if let Some(g) = cur.take() {
            if g.count == 1 {
                out.push(g.first);
            } else {
                out.push(COp::Unitary {
                    qubits: g.support,
                    mat: g.mat,
                });
            }
        }
    };

    for op in ops {
        let fusable = matches!(&op, COp::Unitary { qubits, .. } | COp::Diagonal { qubits, .. }
            if qubits.len() <= kmax);
        if !fusable {
            flush(&mut cur, &mut out);
            out.push(op);
            continue;
        }
        match &mut cur {
            None => {
                let (support, mat) = to_matrix(&op);
                cur = Some(Group {
                    support,
                    mat,
                    count: 1,
                    first: op,
                });
            }
            Some(g) => {
                let (qs, m) = to_matrix(&op);
                let u = union_sorted(&g.support, &qs);
                if u.len() <= kmax {
                    let cur_e = embed_unitary(&g.mat, &g.support, &u);
                    let op_e = embed_unitary(&m, &qs, &u);
                    g.mat = matmul(&op_e, &cur_e, 1 << u.len());
                    g.support = u;
                    g.count += 1;
                } else {
                    flush(&mut cur, &mut out);
                    let (support, mat) = to_matrix(&op);
                    cur = Some(Group {
                        support,
                        mat,
                        count: 1,
                        first: op,
                    });
                }
            }
        }
    }
    flush(&mut cur, &mut out);
    out
}

#[cfg(test)]
#[allow(clippy::identity_op, clippy::erasing_op)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;

    fn compile_c(c: &Circuit, fusion_max: u8) -> Compiled {
        compile(
            &c.to_program(),
            &CompileOptions {
                fusion_max,
                ..Default::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn hh_cancels() {
        let mut c = Circuit::new(2);
        c.h(0).h(0).x(1);
        let out = compile_c(&c, 0);
        assert_eq!(out.stats.cancelled, 2);
        assert_eq!(out.ops.len(), 1);
    }

    #[test]
    fn cancellation_looks_through_disjoint_gates() {
        let mut c = Circuit::new(3);
        c.cx(0, 1).h(2).cx(0, 1);
        let out = compile_c(&c, 0);
        assert_eq!(out.stats.cancelled, 2);
        assert_eq!(out.ops.len(), 1);
    }

    #[test]
    fn bell_fuses_to_single_op() {
        let mut c = Circuit::new(2);
        c.h(0).cx(0, 1).measure_all();
        let out = compile_c(&c, 5);
        assert_eq!(out.ops.len(), 1);
        assert!(!out.dynamic);
        assert_eq!(out.final_measures, vec![(0, 0), (1, 1)]);
        match &out.ops[0] {
            COp::Unitary { qubits, mat } => {
                assert_eq!(qubits, &[0, 1]);
                // (CX)(H⊗I)|00> column: |00>+|11> / √2
                let s = std::f64::consts::FRAC_1_SQRT_2;
                assert!((mat[0] - C64::new(s, 0.0)).norm() < 1e-12);
                assert!((mat[12] - C64::new(s, 0.0)).norm() < 1e-12);
            }
            other => panic!("expected fused unitary, got {other:?}"),
        }
    }

    #[test]
    fn diagonal_run_fuses() {
        let mut c = Circuit::new(3);
        c.t(0).cz(0, 1).s(2).rz(1, 0.3);
        let out = compile_c(&c, 5);
        assert_eq!(out.ops.len(), 1, "{:?}", out.ops);
        assert!(matches!(&out.ops[0], COp::Diagonal { qubits, .. } if qubits == &[0, 1, 2]));
    }

    #[test]
    fn mid_circuit_measure_is_dynamic() {
        let mut c = Circuit::new(2);
        c.h(0).measure(0, 0).x(1).measure(1, 1);
        let out = compile_c(&c, 5);
        assert!(out.dynamic);
    }

    #[test]
    fn barrier_fences_fusion() {
        let mut c = Circuit::new(2);
        c.h(0);
        c.barrier();
        c.h(0);
        let out = compile_c(&c, 5);
        // barrier prevents both cancellation and fusion
        assert_eq!(out.ops.len(), 2);
    }

    #[test]
    fn embed_unitary_identity_on_rest() {
        // X on qubit 0 embedded into {0,1}: X ⊗ I
        let x = vec![
            C64::default(),
            C64::new(1.0, 0.0),
            C64::new(1.0, 0.0),
            C64::default(),
        ];
        let m = embed_unitary(&x, &[0], &[0, 1]);
        // |00> -> |01> (index 0 -> 1), |10> (2) -> |11> (3)
        assert_eq!(m[1 * 4 + 0], C64::new(1.0, 0.0));
        assert_eq!(m[0 * 4 + 1], C64::new(1.0, 0.0));
        assert_eq!(m[3 * 4 + 2], C64::new(1.0, 0.0));
        assert_eq!(m[2 * 4 + 3], C64::new(1.0, 0.0));
    }
}
