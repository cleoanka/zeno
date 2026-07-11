//! OpenQASM 2.0 front end: hand-written lexer + recursive-descent parser.
//!
//! Interface contract (the rest of the crate depends on exactly this):
//! - `parse_str(&str) -> Result<Program, QasmError>`
//! - `parse_file(&Path) -> Result<Program, Error>`
//! - `QasmError { line, col, msg }` with `Display`.
//!
//! Supported language (see `docs/QASM.md` for the full matrix):
//! - `OPENQASM 2.0;` header (mandatory, first statement).
//! - `include "qelib1.inc";` — recognized and satisfied *internally*; no file
//!   is read. The crate's native gate set ([`crate::gates::lookup`]) is always
//!   available, even without the include (a friendly superset of the spec).
//! - `qreg`/`creg` declarations, `gate` definitions (expanded inline at call
//!   sites, nesting depth ≤ 128), gate calls with spec broadcasting,
//!   `measure`, `reset`, `barrier`, `if (creg == n) qop;`.
//! - Parameter expressions: literals, `pi`, bound gate parameters,
//!   `+ - * / ^` (`^` binds tightest, right-assoc), unary `-`, parentheses,
//!   `sin cos tan exp ln sqrt` — evaluated to `f64` at parse time.
//!
//! Every error carries an accurate 1-based line and column, names the
//! offending token and says what was expected.

use crate::gates;
use crate::ir::{GateInstr, Instr, Program, Reg};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Maximum nesting depth when expanding user-defined gates.
const MAX_EXPANSION_DEPTH: usize = 128;

/// Maximum total number of qubits across all `qreg` declarations.
const MAX_TOTAL_QUBITS: u32 = 48;

/// Words that cannot be used as register, gate or parameter names.
const RESERVED: &[&str] = &[
    "OPENQASM", "include", "qreg", "creg", "gate", "opaque", "measure", "reset", "barrier", "if",
    "pi", "U", "CX",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QasmError {
    pub line: usize,
    pub col: usize,
    pub msg: String,
}

impl std::fmt::Display for QasmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}:{}: {}", self.line, self.col, self.msg)
    }
}

impl std::error::Error for QasmError {}

/// Parse OpenQASM 2.0 source into a [`Program`].
pub fn parse_str(src: &str) -> Result<Program, QasmError> {
    let tokens = lex(src)?;
    Parser::new(tokens).parse_program()
}

/// Parse an OpenQASM 2.0 file into a [`Program`].
pub fn parse_file(path: &Path) -> Result<Program, crate::Error> {
    let src = std::fs::read_to_string(path)?;
    Ok(parse_str(&src)?)
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    /// Numeric literal; `text` is the raw lexeme (kept for exact integer
    /// parsing), `value` its `f64` value, `is_int` whether it is a plain
    /// unsigned integer (no `.`, no exponent).
    Number {
        text: String,
        value: f64,
        is_int: bool,
    },
    Str(String),
    Semi,
    Comma,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    Arrow,
    EqEq,
    Eof,
}

impl Tok {
    /// Human-readable description used in error messages.
    fn describe(&self) -> String {
        match self {
            Tok::Ident(s) => format!("'{s}'"),
            Tok::Number { text, .. } => format!("number '{text}'"),
            Tok::Str(s) => format!("string \"{s}\""),
            Tok::Semi => "';'".into(),
            Tok::Comma => "','".into(),
            Tok::LParen => "'('".into(),
            Tok::RParen => "')'".into(),
            Tok::LBracket => "'['".into(),
            Tok::RBracket => "']'".into(),
            Tok::LBrace => "'{'".into(),
            Tok::RBrace => "'}'".into(),
            Tok::Plus => "'+'".into(),
            Tok::Minus => "'-'".into(),
            Tok::Star => "'*'".into(),
            Tok::Slash => "'/'".into(),
            Tok::Caret => "'^'".into(),
            Tok::Arrow => "'->'".into(),
            Tok::EqEq => "'=='".into(),
            Tok::Eof => "end of input".into(),
        }
    }
}

#[derive(Debug, Clone)]
struct Token {
    tok: Tok,
    line: usize,
    col: usize,
}

fn err(line: usize, col: usize, msg: impl Into<String>) -> QasmError {
    QasmError {
        line,
        col,
        msg: msg.into(),
    }
}

/// Character stream with 1-based line/column tracking.
struct Chars<'a> {
    it: std::iter::Peekable<std::str::Chars<'a>>,
    line: usize,
    col: usize,
}

impl<'a> Chars<'a> {
    fn new(src: &'a str) -> Self {
        Chars {
            it: src.chars().peekable(),
            line: 1,
            col: 1,
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.it.peek().copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.it.next()?;
        if c == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(c)
    }
}

fn lex(src: &str) -> Result<Vec<Token>, QasmError> {
    let mut cs = Chars::new(src);
    let mut out = Vec::new();
    loop {
        // Skip whitespace.
        while matches!(cs.peek(), Some(c) if c.is_whitespace()) {
            cs.bump();
        }
        let (line, col) = (cs.line, cs.col);
        let Some(c) = cs.peek() else {
            out.push(Token {
                tok: Tok::Eof,
                line,
                col,
            });
            return Ok(out);
        };
        let tok = match c {
            '/' => {
                cs.bump();
                if cs.peek() == Some('/') {
                    while matches!(cs.peek(), Some(c) if c != '\n') {
                        cs.bump();
                    }
                    continue;
                }
                Tok::Slash
            }
            ';' => {
                cs.bump();
                Tok::Semi
            }
            ',' => {
                cs.bump();
                Tok::Comma
            }
            '(' => {
                cs.bump();
                Tok::LParen
            }
            ')' => {
                cs.bump();
                Tok::RParen
            }
            '[' => {
                cs.bump();
                Tok::LBracket
            }
            ']' => {
                cs.bump();
                Tok::RBracket
            }
            '{' => {
                cs.bump();
                Tok::LBrace
            }
            '}' => {
                cs.bump();
                Tok::RBrace
            }
            '+' => {
                cs.bump();
                Tok::Plus
            }
            '*' => {
                cs.bump();
                Tok::Star
            }
            '^' => {
                cs.bump();
                Tok::Caret
            }
            '-' => {
                cs.bump();
                if cs.peek() == Some('>') {
                    cs.bump();
                    Tok::Arrow
                } else {
                    Tok::Minus
                }
            }
            '=' => {
                cs.bump();
                if cs.peek() == Some('=') {
                    cs.bump();
                    Tok::EqEq
                } else {
                    return Err(err(
                        line,
                        col,
                        "found a single '='; the only equality operator is '==' \
                         (as in 'if (c == 1)')",
                    ));
                }
            }
            '"' => {
                cs.bump();
                let mut s = String::new();
                loop {
                    match cs.bump() {
                        Some('"') => break,
                        Some('\n') | None => {
                            return Err(err(line, col, "unterminated string literal"));
                        }
                        Some(c) => s.push(c),
                    }
                }
                Tok::Str(s)
            }
            c if c.is_ascii_digit() || c == '.' => lex_number(&mut cs, line, col)?,
            c if c.is_ascii_alphabetic() || c == '_' => {
                let mut s = String::new();
                while matches!(cs.peek(), Some(c) if c.is_ascii_alphanumeric() || c == '_') {
                    s.push(cs.bump().unwrap());
                }
                Tok::Ident(s)
            }
            c => {
                return Err(err(line, col, format!("unexpected character '{c}'")));
            }
        };
        out.push(Token { tok, line, col });
    }
}

/// Lex a numeric literal: `[0-9]* ('.' [0-9]*)? ([eE] [+-]? [0-9]+)?`
/// with at least one mantissa digit.
fn lex_number(cs: &mut Chars, line: usize, col: usize) -> Result<Tok, QasmError> {
    let mut text = String::new();
    let mut has_digits = false;
    while matches!(cs.peek(), Some(c) if c.is_ascii_digit()) {
        text.push(cs.bump().unwrap());
        has_digits = true;
    }
    let mut is_int = true;
    if cs.peek() == Some('.') {
        is_int = false;
        text.push(cs.bump().unwrap());
        while matches!(cs.peek(), Some(c) if c.is_ascii_digit()) {
            text.push(cs.bump().unwrap());
            has_digits = true;
        }
    }
    if !has_digits {
        return Err(err(
            line,
            col,
            format!("malformed number '{text}': no digits"),
        ));
    }
    if matches!(cs.peek(), Some('e' | 'E')) {
        is_int = false;
        text.push(cs.bump().unwrap());
        if matches!(cs.peek(), Some('+' | '-')) {
            text.push(cs.bump().unwrap());
        }
        let mut exp_digits = false;
        while matches!(cs.peek(), Some(c) if c.is_ascii_digit()) {
            text.push(cs.bump().unwrap());
            exp_digits = true;
        }
        if !exp_digits {
            return Err(err(
                line,
                col,
                format!("malformed number '{text}': exponent has no digits"),
            ));
        }
    }
    let value = text
        .parse::<f64>()
        .map_err(|_| err(line, col, format!("malformed number '{text}'")))?;
    Ok(Tok::Number {
        text,
        value,
        is_int,
    })
}

// ---------------------------------------------------------------------------
// Parameter expressions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Func {
    Sin,
    Cos,
    Tan,
    Exp,
    Ln,
    Sqrt,
}

impl Func {
    fn from_name(name: &str) -> Option<Func> {
        Some(match name {
            "sin" => Func::Sin,
            "cos" => Func::Cos,
            "tan" => Func::Tan,
            "exp" => Func::Exp,
            "ln" => Func::Ln,
            "sqrt" => Func::Sqrt,
            _ => return None,
        })
    }
}

/// Parameter-expression AST. Identifiers are resolved to formal-parameter
/// indices at parse time, so evaluation cannot fail.
#[derive(Debug, Clone, PartialEq)]
enum Expr {
    Num(f64),
    Pi,
    Param(usize),
    Neg(Box<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Fun(Func, Box<Expr>),
}

impl Expr {
    fn eval(&self, params: &[f64]) -> f64 {
        match self {
            Expr::Num(v) => *v,
            Expr::Pi => std::f64::consts::PI,
            Expr::Param(i) => params[*i],
            Expr::Neg(e) => -e.eval(params),
            Expr::Bin(op, a, b) => {
                let (a, b) = (a.eval(params), b.eval(params));
                match op {
                    BinOp::Add => a + b,
                    BinOp::Sub => a - b,
                    BinOp::Mul => a * b,
                    BinOp::Div => a / b,
                    BinOp::Pow => a.powf(b),
                }
            }
            Expr::Fun(f, e) => {
                let v = e.eval(params);
                match f {
                    Func::Sin => v.sin(),
                    Func::Cos => v.cos(),
                    Func::Tan => v.tan(),
                    Func::Exp => v.exp(),
                    Func::Ln => v.ln(),
                    Func::Sqrt => v.sqrt(),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// User-defined gates (macros)
// ---------------------------------------------------------------------------

/// One statement inside a `gate` body. Qubit arguments are indices into the
/// definition's formal qubit-argument list.
#[derive(Debug, Clone)]
enum MacroStmt {
    Call {
        /// Resolved callee name: a native gate or a previously defined macro
        /// (`U`/`CX` are resolved to `u3`/`cx` at definition time).
        name: String,
        params: Vec<Expr>,
        args: Vec<usize>,
    },
    Barrier {
        args: Vec<usize>,
    },
}

#[derive(Debug, Clone)]
struct GateMacro {
    n_params: usize,
    n_qargs: usize,
    body: Vec<MacroStmt>,
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// A resolved qubit/clbit argument: a whole register or a single bit.
#[derive(Debug, Clone, Copy)]
enum ArgVal {
    /// (register index into `prog.qregs`/`prog.cregs`, size, global offset)
    Whole(usize, u32, u32),
    /// Global bit index.
    Bit(u32),
}

struct Parser {
    toks: Vec<Token>,
    pos: usize,
    prog: Program,
    /// Register name → index into `prog.qregs`.
    qregs: HashMap<String, usize>,
    /// Register name → index into `prog.cregs`.
    cregs: HashMap<String, usize>,
    macros: HashMap<String, GateMacro>,
}

impl Parser {
    fn new(toks: Vec<Token>) -> Self {
        Parser {
            toks,
            pos: 0,
            prog: Program::default(),
            qregs: HashMap::new(),
            cregs: HashMap::new(),
            macros: HashMap::new(),
        }
    }

    // -- token helpers ------------------------------------------------------

    fn peek(&self) -> &Token {
        &self.toks[self.pos.min(self.toks.len() - 1)]
    }

    fn advance(&mut self) -> Token {
        let t = self.toks[self.pos.min(self.toks.len() - 1)].clone();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn err_here(&self, msg: impl Into<String>) -> QasmError {
        let t = self.peek();
        err(t.line, t.col, msg)
    }

    fn expect(&mut self, want: &Tok, what: &str) -> Result<Token, QasmError> {
        if &self.peek().tok == want {
            Ok(self.advance())
        } else {
            Err(self.err_here(format!(
                "expected {} {what}, found {}",
                want.describe(),
                self.peek().tok.describe()
            )))
        }
    }

    fn eat(&mut self, want: &Tok) -> bool {
        if &self.peek().tok == want {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Expect an identifier usable as a fresh name (register, gate, formal
    /// parameter, …); rejects reserved words.
    fn expect_name(&mut self, what: &str) -> Result<(String, usize, usize), QasmError> {
        let t = self.peek().clone();
        match &t.tok {
            Tok::Ident(s) if RESERVED.contains(&s.as_str()) => Err(err(
                t.line,
                t.col,
                format!("'{s}' is a reserved word and cannot be used as {what}"),
            )),
            Tok::Ident(s) => {
                let s = s.clone();
                self.advance();
                Ok((s, t.line, t.col))
            }
            other => Err(err(
                t.line,
                t.col,
                format!(
                    "expected an identifier ({what}), found {}",
                    other.describe()
                ),
            )),
        }
    }

    /// Expect an unsigned integer literal, parsed with overflow checking.
    fn expect_uint(&mut self, max: u64, what: &str) -> Result<(u64, usize, usize), QasmError> {
        let t = self.peek().clone();
        match &t.tok {
            Tok::Number { text, is_int, .. } if *is_int => {
                let v = text.parse::<u64>().map_err(|_| {
                    err(
                        t.line,
                        t.col,
                        format!("integer '{text}' is too large for {what}"),
                    )
                })?;
                if v > max {
                    return Err(err(
                        t.line,
                        t.col,
                        format!("integer '{text}' is too large for {what} (max {max})"),
                    ));
                }
                self.advance();
                Ok((v, t.line, t.col))
            }
            other => Err(err(
                t.line,
                t.col,
                format!(
                    "expected an unsigned integer ({what}), found {}",
                    other.describe()
                ),
            )),
        }
    }

    // -- program ------------------------------------------------------------

    fn parse_program(mut self) -> Result<Program, QasmError> {
        self.parse_header()?;
        while self.peek().tok != Tok::Eof {
            self.parse_statement()?;
        }
        Ok(self.prog)
    }

    fn parse_header(&mut self) -> Result<(), QasmError> {
        let t = self.peek().clone();
        match &t.tok {
            Tok::Ident(s) if s == "OPENQASM" => {
                self.advance();
            }
            other => {
                return Err(err(
                    t.line,
                    t.col,
                    format!(
                        "expected 'OPENQASM 2.0;' as the first statement, found {}",
                        other.describe()
                    ),
                ));
            }
        }
        let t = self.peek().clone();
        match &t.tok {
            Tok::Number { text, value, .. } => {
                if *value != 2.0 {
                    return Err(err(
                        t.line,
                        t.col,
                        format!("unsupported OpenQASM version '{text}' (only 2.0 is supported)"),
                    ));
                }
                self.advance();
            }
            other => {
                return Err(err(
                    t.line,
                    t.col,
                    format!(
                        "expected version number '2.0' after 'OPENQASM', found {}",
                        other.describe()
                    ),
                ));
            }
        }
        self.expect(&Tok::Semi, "after the OPENQASM header")?;
        Ok(())
    }

    fn parse_statement(&mut self) -> Result<(), QasmError> {
        let t = self.peek().clone();
        let Tok::Ident(word) = &t.tok else {
            return Err(err(
                t.line,
                t.col,
                format!("expected a statement, found {}", t.tok.describe()),
            ));
        };
        match word.as_str() {
            "OPENQASM" => Err(err(
                t.line,
                t.col,
                "duplicate 'OPENQASM' header (it must appear exactly once, first)",
            )),
            "include" => self.parse_include(),
            "qreg" | "creg" => self.parse_reg_decl(),
            "gate" => self.parse_gate_def(),
            "opaque" => Err(err(
                t.line,
                t.col,
                "'opaque' gate declarations are unsupported (they have no simulation \
                 semantics); define the gate body with 'gate' instead",
            )),
            _ => {
                let instrs = self.parse_qop(false)?;
                self.prog.instrs.extend(instrs);
                Ok(())
            }
        }
    }

    fn parse_include(&mut self) -> Result<(), QasmError> {
        self.advance(); // 'include'
        let t = self.peek().clone();
        match &t.tok {
            Tok::Str(name) if name == "qelib1.inc" => {
                self.advance();
            }
            Tok::Str(name) => {
                return Err(err(
                    t.line,
                    t.col,
                    format!(
                        "cannot include \"{name}\": only \"qelib1.inc\" is supported \
                         (and it is satisfied internally — the native gate set is \
                         always available)"
                    ),
                ));
            }
            other => {
                return Err(err(
                    t.line,
                    t.col,
                    format!(
                        "expected a quoted filename after 'include', found {}",
                        other.describe()
                    ),
                ));
            }
        }
        self.expect(&Tok::Semi, "after the include")?;
        Ok(())
    }

    fn parse_reg_decl(&mut self) -> Result<(), QasmError> {
        let kw = self.advance(); // 'qreg' | 'creg'
        let is_q = matches!(&kw.tok, Tok::Ident(s) if s == "qreg");
        let (name, nline, ncol) = self.expect_name("a register name")?;
        if self.qregs.contains_key(&name) || self.cregs.contains_key(&name) {
            return Err(err(
                nline,
                ncol,
                format!("register '{name}' is already declared"),
            ));
        }
        self.expect(&Tok::LBracket, "before the register size")?;
        let (size, sline, scol) = self.expect_uint(u32::MAX as u64, "a register size")?;
        if size == 0 {
            return Err(err(sline, scol, "register size must be at least 1"));
        }
        self.expect(&Tok::RBracket, "after the register size")?;
        self.expect(&Tok::Semi, "after the register declaration")?;
        let size = size as u32;
        if is_q {
            let total = self.prog.n_qubits() as u64 + size as u64;
            if total > MAX_TOTAL_QUBITS as u64 {
                return Err(err(
                    sline,
                    scol,
                    format!(
                        "declaring 'qreg {name}[{size}]' would bring the total to {total} \
                         qubits; the maximum is {MAX_TOTAL_QUBITS}"
                    ),
                ));
            }
            self.qregs.insert(name.clone(), self.prog.qregs.len());
            self.prog.qregs.push(Reg { name, size });
        } else {
            self.cregs.insert(name.clone(), self.prog.cregs.len());
            self.prog.cregs.push(Reg { name, size });
        }
        Ok(())
    }

    // -- quantum operations ---------------------------------------------------

    /// Parse one quantum operation (gate call, measure, reset, barrier or if)
    /// including the trailing ';', returning the broadcast-expanded
    /// instructions. `in_if` restricts to the ops allowed under `if`.
    fn parse_qop(&mut self, in_if: bool) -> Result<Vec<Instr>, QasmError> {
        let t = self.peek().clone();
        let Tok::Ident(word) = &t.tok else {
            return Err(err(
                t.line,
                t.col,
                format!("expected a quantum operation, found {}", t.tok.describe()),
            ));
        };
        match word.as_str() {
            "measure" => self.parse_measure(),
            "reset" => self.parse_reset(),
            "barrier" if in_if => Err(err(
                t.line,
                t.col,
                "'barrier' cannot be the body of an 'if' statement \
                 (only gate calls, measure and reset can)",
            )),
            "barrier" => self.parse_barrier(),
            "if" if in_if => Err(err(t.line, t.col, "'if' statements cannot be nested")),
            "if" => self.parse_if(),
            "include" | "qreg" | "creg" | "gate" | "opaque" | "OPENQASM" => Err(err(
                t.line,
                t.col,
                format!("'{word}' is not a quantum operation (it cannot follow 'if')"),
            )),
            _ => self.parse_gate_call(),
        }
    }

    fn parse_gate_call(&mut self) -> Result<Vec<Instr>, QasmError> {
        let name_tok = self.advance();
        let (name, line, col) = match &name_tok.tok {
            Tok::Ident(s) => (s.clone(), name_tok.line, name_tok.col),
            _ => unreachable!("dispatched on Ident"),
        };
        // Resolve the callee: builtins U/CX, then user macros, then natives.
        let (resolved, n_params, arity, is_macro) = self.resolve_gate(&name, line, col)?;

        // Optional parameter list, evaluated immediately (top level has no
        // formal parameters in scope).
        let params = if self.peek().tok == Tok::LParen {
            let exprs = self.parse_param_list(&[])?;
            exprs.iter().map(|e| e.eval(&[])).collect::<Vec<f64>>()
        } else {
            Vec::new()
        };
        if params.len() != n_params {
            return Err(err(
                line,
                col,
                format!(
                    "gate '{name}' expects {n_params} parameter(s), got {}",
                    params.len()
                ),
            ));
        }

        // Qubit arguments.
        let mut args = Vec::new();
        loop {
            args.push(self.parse_q_arg()?);
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::Semi, "after the gate call")?;
        if args.len() != arity {
            return Err(err(
                line,
                col,
                format!(
                    "gate '{name}' expects {arity} qubit argument(s), got {}",
                    args.len()
                ),
            ));
        }

        // Broadcast and emit.
        let n = self.broadcast_len(&args, line, col)?;
        let mut out = Vec::new();
        for i in 0..n {
            let qubits: Vec<u32> = args
                .iter()
                .map(|a| match a {
                    ArgVal::Whole(_, _, off) => off + i,
                    ArgVal::Bit(q) => *q,
                })
                .collect();
            self.check_distinct(&qubits, &name, line, col)?;
            if is_macro {
                self.expand_macro(&resolved, &params, &qubits, 0, line, col, &mut out)?;
            } else {
                out.push(Instr::Gate(GateInstr {
                    name: resolved.clone(),
                    params: params.clone(),
                    qubits,
                }));
            }
        }
        Ok(out)
    }

    /// Resolve a gate name to `(resolved_name, n_params, arity, is_macro)`.
    fn resolve_gate(
        &self,
        name: &str,
        line: usize,
        col: usize,
    ) -> Result<(String, usize, usize, bool), QasmError> {
        if name == "U" {
            return Ok(("u3".into(), 3, 1, false));
        }
        if name == "CX" {
            return Ok(("cx".into(), 0, 2, false));
        }
        if let Some(m) = self.macros.get(name) {
            return Ok((name.into(), m.n_params, m.n_qargs, true));
        }
        if let Some(def) = gates::lookup(name) {
            return Ok((
                name.into(),
                def.n_params as usize,
                def.arity as usize,
                false,
            ));
        }
        let mut msg = format!("unknown gate '{name}'");
        let lower = name.to_lowercase();
        if lower != name && (gates::lookup(&lower).is_some() || self.macros.contains_key(&lower)) {
            msg.push_str(&format!(
                " (gate names are case-sensitive; did you mean '{lower}'?)"
            ));
        }
        Err(err(line, col, msg))
    }

    /// Parse `( expr, expr, ... )` with the given formal-parameter scope.
    fn parse_param_list(&mut self, scope: &[String]) -> Result<Vec<Expr>, QasmError> {
        self.expect(&Tok::LParen, "before the gate parameters")?;
        let mut exprs = Vec::new();
        if self.peek().tok != Tok::RParen {
            loop {
                exprs.push(self.parse_expr(scope)?);
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
        }
        self.expect(&Tok::RParen, "after the gate parameters")?;
        Ok(exprs)
    }

    /// Parse a quantum argument: `reg` or `reg[i]`, resolved against qregs.
    fn parse_q_arg(&mut self) -> Result<ArgVal, QasmError> {
        self.parse_reg_arg(true)
    }

    /// Parse a classical argument: `reg` or `reg[i]`, resolved against cregs.
    fn parse_c_arg(&mut self) -> Result<ArgVal, QasmError> {
        self.parse_reg_arg(false)
    }

    fn parse_reg_arg(&mut self, quantum: bool) -> Result<ArgVal, QasmError> {
        let t = self.peek().clone();
        let Tok::Ident(name) = &t.tok else {
            let kind = if quantum { "qubit" } else { "classical bit" };
            return Err(err(
                t.line,
                t.col,
                format!(
                    "expected a {kind} argument (a register name, optionally indexed), \
                     found {}",
                    t.tok.describe()
                ),
            ));
        };
        let name = name.clone();
        self.advance();
        let (idx, size, offset) = if quantum {
            match self.qregs.get(&name) {
                Some(&i) => (i, self.prog.qregs[i].size, self.prog.qreg_offset(i)),
                None => {
                    let msg = if self.cregs.contains_key(&name) {
                        format!(
                            "'{name}' is a classical register; a quantum register is expected here"
                        )
                    } else {
                        format!("unknown quantum register '{name}'")
                    };
                    return Err(err(t.line, t.col, msg));
                }
            }
        } else {
            match self.cregs.get(&name) {
                Some(&i) => (i, self.prog.cregs[i].size, self.prog.creg_offset(i)),
                None => {
                    let msg = if self.qregs.contains_key(&name) {
                        format!(
                            "'{name}' is a quantum register; a classical register is expected here"
                        )
                    } else {
                        format!("unknown classical register '{name}'")
                    };
                    return Err(err(t.line, t.col, msg));
                }
            }
        };
        if self.eat(&Tok::LBracket) {
            let (i, iline, icol) = self.expect_uint(u32::MAX as u64, "a register index")?;
            self.expect(&Tok::RBracket, "after the register index")?;
            if i >= size as u64 {
                return Err(err(
                    iline,
                    icol,
                    format!("index {i} out of range for register '{name}' of size {size}"),
                ));
            }
            Ok(ArgVal::Bit(offset + i as u32))
        } else {
            Ok(ArgVal::Whole(idx, size, offset))
        }
    }

    /// Broadcast length: all whole-register args must have equal size; if
    /// none, the length is 1.
    fn broadcast_len(&self, args: &[ArgVal], line: usize, col: usize) -> Result<u32, QasmError> {
        let mut n: Option<(u32, usize)> = None;
        for a in args {
            if let ArgVal::Whole(idx, size, _) = a {
                match n {
                    None => n = Some((*size, *idx)),
                    Some((s, first)) if s != *size => {
                        return Err(err(
                            line,
                            col,
                            format!(
                                "broadcast size mismatch: register '{}' has size {s} but \
                                 register '{}' has size {size}",
                                self.prog.qregs[first].name, self.prog.qregs[*idx].name
                            ),
                        ));
                    }
                    Some(_) => {}
                }
            }
        }
        Ok(n.map_or(1, |(s, _)| s))
    }

    /// Error if any qubit appears twice in one expanded gate application.
    fn check_distinct(
        &self,
        qubits: &[u32],
        gate: &str,
        line: usize,
        col: usize,
    ) -> Result<(), QasmError> {
        let mut seen = HashSet::new();
        for &q in qubits {
            if !seen.insert(q) {
                return Err(err(
                    line,
                    col,
                    format!(
                        "gate '{gate}' applied to duplicate qubit {} after broadcasting",
                        self.qubit_name(q)
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Pretty name (`q[3]`) for a global qubit index, for error messages.
    fn qubit_name(&self, q: u32) -> String {
        let mut off = 0u32;
        for r in &self.prog.qregs {
            if q < off + r.size {
                return format!("'{}[{}]'", r.name, q - off);
            }
            off += r.size;
        }
        format!("#{q}")
    }

    fn parse_measure(&mut self) -> Result<Vec<Instr>, QasmError> {
        let kw = self.advance(); // 'measure'
        let src = self.parse_q_arg()?;
        self.expect(&Tok::Arrow, "between the qubit and the classical target")?;
        let dst = self.parse_c_arg()?;
        self.expect(&Tok::Semi, "after the measure statement")?;
        // Normalize size-1 whole registers to their single bit.
        let norm = |a: ArgVal| match a {
            ArgVal::Whole(_, 1, off) => ArgVal::Bit(off),
            other => other,
        };
        match (norm(src), norm(dst)) {
            (ArgVal::Bit(q), ArgVal::Bit(c)) => Ok(vec![Instr::Measure { qubit: q, clbit: c }]),
            (ArgVal::Whole(qi, qs, qoff), ArgVal::Whole(ci, cs, coff)) => {
                if qs != cs {
                    return Err(err(
                        kw.line,
                        kw.col,
                        format!(
                            "measure size mismatch: quantum register '{}' has size {qs} but \
                             classical register '{}' has size {cs}",
                            self.prog.qregs[qi].name, self.prog.cregs[ci].name
                        ),
                    ));
                }
                Ok((0..qs)
                    .map(|i| Instr::Measure {
                        qubit: qoff + i,
                        clbit: coff + i,
                    })
                    .collect())
            }
            (ArgVal::Whole(qi, qs, _), ArgVal::Bit(_)) => Err(err(
                kw.line,
                kw.col,
                format!(
                    "cannot broadcast measure: '{}' is a register of size {qs} but the \
                     target is a single classical bit",
                    self.prog.qregs[qi].name
                ),
            )),
            (ArgVal::Bit(_), ArgVal::Whole(ci, cs, _)) => Err(err(
                kw.line,
                kw.col,
                format!(
                    "cannot broadcast measure: the source is a single qubit but '{}' is a \
                     classical register of size {cs}",
                    self.prog.cregs[ci].name
                ),
            )),
        }
    }

    fn parse_reset(&mut self) -> Result<Vec<Instr>, QasmError> {
        self.advance(); // 'reset'
        let arg = self.parse_q_arg()?;
        self.expect(&Tok::Semi, "after the reset statement")?;
        Ok(match arg {
            ArgVal::Bit(q) => vec![Instr::Reset { qubit: q }],
            ArgVal::Whole(_, size, off) => {
                (0..size).map(|i| Instr::Reset { qubit: off + i }).collect()
            }
        })
    }

    fn parse_barrier(&mut self) -> Result<Vec<Instr>, QasmError> {
        self.advance(); // 'barrier'
        let mut qubits = Vec::new();
        let mut seen = HashSet::new();
        loop {
            match self.parse_q_arg()? {
                ArgVal::Bit(q) => {
                    if seen.insert(q) {
                        qubits.push(q);
                    }
                }
                ArgVal::Whole(_, size, off) => {
                    for q in off..off + size {
                        if seen.insert(q) {
                            qubits.push(q);
                        }
                    }
                }
            }
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::Semi, "after the barrier")?;
        Ok(vec![Instr::Barrier(qubits)])
    }

    fn parse_if(&mut self) -> Result<Vec<Instr>, QasmError> {
        self.advance(); // 'if'
        self.expect(&Tok::LParen, "after 'if'")?;
        let t = self.peek().clone();
        let Tok::Ident(name) = &t.tok else {
            return Err(err(
                t.line,
                t.col,
                format!(
                    "expected a classical register name in the if condition, found {}",
                    t.tok.describe()
                ),
            ));
        };
        let name = name.clone();
        let Some(&creg) = self.cregs.get(&name) else {
            let msg = if self.qregs.contains_key(&name) {
                format!(
                    "'{name}' is a quantum register; an if condition tests a classical register"
                )
            } else {
                format!("unknown classical register '{name}' in if condition")
            };
            return Err(err(t.line, t.col, msg));
        };
        self.advance();
        if self.peek().tok == Tok::LBracket {
            return Err(self.err_here(format!(
                "an if condition must test a whole classical register \
                 ('if ({name} == n)'), not an indexed bit"
            )));
        }
        self.expect(&Tok::EqEq, "in the if condition")?;
        let (value, _, _) = self.expect_uint(u64::MAX, "the if comparison value")?;
        self.expect(&Tok::RParen, "after the if condition")?;
        let ops = self.parse_qop(true)?;
        Ok(ops
            .into_iter()
            .map(|op| Instr::If {
                creg,
                value,
                op: Box::new(op),
            })
            .collect())
    }

    // -- gate definitions ---------------------------------------------------

    fn parse_gate_def(&mut self) -> Result<(), QasmError> {
        self.advance(); // 'gate'
        let (name, nline, ncol) = self.expect_name("a gate name")?;
        if gates::lookup(&name).is_some() {
            return Err(err(
                nline,
                ncol,
                format!("gate '{name}' is already defined (it is a native gate)"),
            ));
        }
        if self.macros.contains_key(&name) {
            return Err(err(
                nline,
                ncol,
                format!("gate '{name}' is already defined"),
            ));
        }

        // Formal parameters.
        let mut params: Vec<String> = Vec::new();
        if self.eat(&Tok::LParen) {
            if self.peek().tok != Tok::RParen {
                loop {
                    let (p, pline, pcol) = self.expect_name("a gate parameter name")?;
                    if Func::from_name(&p).is_some() {
                        return Err(err(
                            pline,
                            pcol,
                            format!("'{p}' is a built-in function and cannot be a parameter name"),
                        ));
                    }
                    if params.contains(&p) {
                        return Err(err(pline, pcol, format!("duplicate parameter name '{p}'")));
                    }
                    params.push(p);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
            }
            self.expect(&Tok::RParen, "after the gate parameters")?;
        }

        // Formal qubit arguments.
        let mut qargs: Vec<String> = Vec::new();
        loop {
            let (q, qline, qcol) = self.expect_name("a gate qubit argument name")?;
            if qargs.contains(&q) || params.contains(&q) {
                return Err(err(qline, qcol, format!("duplicate argument name '{q}'")));
            }
            qargs.push(q);
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::LBrace, "before the gate body")?;

        // Body.
        let mut body = Vec::new();
        while self.peek().tok != Tok::RBrace {
            body.push(self.parse_gate_body_stmt(&name, &params, &qargs)?);
        }
        self.expect(&Tok::RBrace, "after the gate body")?;

        self.macros.insert(
            name,
            GateMacro {
                n_params: params.len(),
                n_qargs: qargs.len(),
                body,
            },
        );
        Ok(())
    }

    fn parse_gate_body_stmt(
        &mut self,
        gate_name: &str,
        params: &[String],
        qargs: &[String],
    ) -> Result<MacroStmt, QasmError> {
        let t = self.peek().clone();
        let Tok::Ident(word) = &t.tok else {
            return Err(err(
                t.line,
                t.col,
                format!(
                    "expected a gate call, 'barrier' or '}}' in the gate body, found {}",
                    t.tok.describe()
                ),
            ));
        };
        match word.as_str() {
            "barrier" => {
                self.advance();
                let mut args = Vec::new();
                loop {
                    args.push(self.parse_formal_qarg(gate_name, qargs)?);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(&Tok::Semi, "after the barrier")?;
                args.dedup();
                Ok(MacroStmt::Barrier { args })
            }
            "measure" | "reset" | "if" | "qreg" | "creg" | "gate" | "opaque" | "include" => {
                Err(err(
                    t.line,
                    t.col,
                    format!(
                        "'{word}' is not allowed inside a gate body \
                         (only gate calls and 'barrier' are)"
                    ),
                ))
            }
            callee => {
                let callee = callee.to_string();
                self.advance();
                if callee == gate_name {
                    return Err(err(
                        t.line,
                        t.col,
                        format!(
                            "gate '{callee}' is not defined yet at this point \
                             (recursive gate definitions are not allowed)"
                        ),
                    ));
                }
                let (resolved, n_params, arity, _) = self.resolve_gate(&callee, t.line, t.col)?;
                let exprs = if self.peek().tok == Tok::LParen {
                    self.parse_param_list(params)?
                } else {
                    Vec::new()
                };
                if exprs.len() != n_params {
                    return Err(err(
                        t.line,
                        t.col,
                        format!(
                            "gate '{callee}' expects {n_params} parameter(s), got {}",
                            exprs.len()
                        ),
                    ));
                }
                let mut args = Vec::new();
                loop {
                    args.push(self.parse_formal_qarg(gate_name, qargs)?);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(&Tok::Semi, "after the gate call")?;
                if args.len() != arity {
                    return Err(err(
                        t.line,
                        t.col,
                        format!(
                            "gate '{callee}' expects {arity} qubit argument(s), got {}",
                            args.len()
                        ),
                    ));
                }
                let mut seen = HashSet::new();
                if let Some(&dup) = args.iter().find(|&&a| !seen.insert(a)) {
                    return Err(err(
                        t.line,
                        t.col,
                        format!(
                            "gate '{callee}' applied to duplicate qubit argument '{}'",
                            qargs[dup]
                        ),
                    ));
                }
                Ok(MacroStmt::Call {
                    name: resolved,
                    params: exprs,
                    args,
                })
            }
        }
    }

    /// Parse a qubit argument inside a gate body: a formal argument name,
    /// never indexed.
    fn parse_formal_qarg(&mut self, gate_name: &str, qargs: &[String]) -> Result<usize, QasmError> {
        let t = self.peek().clone();
        let Tok::Ident(name) = &t.tok else {
            return Err(err(
                t.line,
                t.col,
                format!(
                    "expected a qubit argument of gate '{gate_name}', found {}",
                    t.tok.describe()
                ),
            ));
        };
        let name = name.clone();
        self.advance();
        if self.peek().tok == Tok::LBracket {
            return Err(self.err_here(format!(
                "cannot index '{name}' inside a gate body (gate arguments are single qubits)"
            )));
        }
        qargs.iter().position(|q| q == &name).ok_or_else(|| {
            err(
                t.line,
                t.col,
                format!("'{name}' is not declared as a qubit argument of gate '{gate_name}'"),
            )
        })
    }

    /// Inline-expand a user gate at a call site.
    #[allow(clippy::too_many_arguments)]
    fn expand_macro(
        &self,
        name: &str,
        params: &[f64],
        qubits: &[u32],
        depth: usize,
        line: usize,
        col: usize,
        out: &mut Vec<Instr>,
    ) -> Result<(), QasmError> {
        if depth >= MAX_EXPANSION_DEPTH {
            return Err(err(
                line,
                col,
                format!(
                    "expansion of gate '{name}' exceeds the nesting depth limit \
                     of {MAX_EXPANSION_DEPTH}"
                ),
            ));
        }
        let m = &self.macros[name];
        for stmt in &m.body {
            match stmt {
                MacroStmt::Call {
                    name: callee,
                    params: exprs,
                    args,
                } => {
                    let vals: Vec<f64> = exprs.iter().map(|e| e.eval(params)).collect();
                    let qs: Vec<u32> = args.iter().map(|&a| qubits[a]).collect();
                    if self.macros.contains_key(callee) {
                        self.expand_macro(callee, &vals, &qs, depth + 1, line, col, out)?;
                    } else {
                        out.push(Instr::Gate(GateInstr {
                            name: callee.clone(),
                            params: vals,
                            qubits: qs,
                        }));
                    }
                }
                MacroStmt::Barrier { args } => {
                    out.push(Instr::Barrier(args.iter().map(|&a| qubits[a]).collect()));
                }
            }
        }
        Ok(())
    }

    // -- expressions ----------------------------------------------------------

    /// expr := term { ('+'|'-') term }
    fn parse_expr(&mut self, scope: &[String]) -> Result<Expr, QasmError> {
        let mut lhs = self.parse_term(scope)?;
        loop {
            let op = match self.peek().tok {
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_term(scope)?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    /// term := unary { ('*'|'/') unary }
    fn parse_term(&mut self, scope: &[String]) -> Result<Expr, QasmError> {
        let mut lhs = self.parse_unary(scope)?;
        loop {
            let op = match self.peek().tok {
                Tok::Star => BinOp::Mul,
                Tok::Slash => BinOp::Div,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_unary(scope)?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    /// unary := '-' unary | power
    fn parse_unary(&mut self, scope: &[String]) -> Result<Expr, QasmError> {
        if self.eat(&Tok::Minus) {
            Ok(Expr::Neg(Box::new(self.parse_unary(scope)?)))
        } else {
            self.parse_power(scope)
        }
    }

    /// power := atom [ '^' unary ]   (right-associative, binds tightest)
    fn parse_power(&mut self, scope: &[String]) -> Result<Expr, QasmError> {
        let base = self.parse_atom(scope)?;
        if self.eat(&Tok::Caret) {
            let exp = self.parse_unary(scope)?;
            Ok(Expr::Bin(BinOp::Pow, Box::new(base), Box::new(exp)))
        } else {
            Ok(base)
        }
    }

    /// atom := number | 'pi' | param | func '(' expr ')' | '(' expr ')'
    fn parse_atom(&mut self, scope: &[String]) -> Result<Expr, QasmError> {
        let t = self.peek().clone();
        match &t.tok {
            Tok::Number { value, .. } => {
                let v = *value;
                self.advance();
                Ok(Expr::Num(v))
            }
            Tok::LParen => {
                self.advance();
                let e = self.parse_expr(scope)?;
                self.expect(&Tok::RParen, "to close the parenthesized expression")?;
                Ok(e)
            }
            Tok::Ident(name) if name == "pi" => {
                self.advance();
                Ok(Expr::Pi)
            }
            Tok::Ident(name) => {
                if let Some(f) = Func::from_name(name) {
                    self.advance();
                    self.expect(&Tok::LParen, "after the function name")?;
                    let e = self.parse_expr(scope)?;
                    self.expect(&Tok::RParen, "after the function argument")?;
                    return Ok(Expr::Fun(f, Box::new(e)));
                }
                if let Some(i) = scope.iter().position(|p| p == name) {
                    self.advance();
                    return Ok(Expr::Param(i));
                }
                let msg = if scope.is_empty() {
                    format!(
                        "unknown identifier '{name}' in expression \
                         (no gate parameters are in scope here)"
                    )
                } else {
                    format!(
                        "unknown identifier '{name}' in expression \
                         (gate parameters in scope: {})",
                        scope.join(", ")
                    )
                };
                Err(err(t.line, t.col, msg))
            }
            other => Err(err(
                t.line,
                t.col,
                format!(
                    "expected a number, 'pi', a parameter or '(' in expression, found {}",
                    other.describe()
                ),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BELL: &str =
        "OPENQASM 2.0;\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;\n";

    fn parse(src: &str) -> Program {
        parse_str(src).unwrap_or_else(|e| panic!("parse failed: {e}"))
    }

    fn gate(name: &str, params: &[f64], qubits: &[u32]) -> Instr {
        Instr::Gate(GateInstr {
            name: name.into(),
            params: params.to_vec(),
            qubits: qubits.to_vec(),
        })
    }

    fn assert_err(src: &str, line: usize, col: usize, substr: &str) {
        let e = parse_str(src).expect_err("expected a parse error");
        assert!(
            e.msg.contains(substr),
            "error message {:?} does not contain {:?}",
            e.msg,
            substr
        );
        assert_eq!((e.line, e.col), (line, col), "wrong position for: {e}");
    }

    // -- positive: structure, broadcasting, builtins --------------------------

    #[test]
    fn bell_exact_program() {
        let p = parse(BELL);
        assert_eq!(
            p,
            Program {
                qregs: vec![Reg {
                    name: "q".into(),
                    size: 2
                }],
                cregs: vec![Reg {
                    name: "c".into(),
                    size: 2
                }],
                instrs: vec![
                    gate("h", &[], &[0]),
                    gate("cx", &[], &[0, 1]),
                    Instr::Measure { qubit: 0, clbit: 0 },
                    Instr::Measure { qubit: 1, clbit: 1 },
                ],
            }
        );
    }

    #[test]
    fn broadcast_h_whole_register() {
        let p = parse("OPENQASM 2.0;\nqreg q[3];\nh q;\n");
        assert_eq!(
            p.instrs,
            vec![
                gate("h", &[], &[0]),
                gate("h", &[], &[1]),
                gate("h", &[], &[2])
            ]
        );
    }

    #[test]
    fn broadcast_cx_reg_reg() {
        let p = parse("OPENQASM 2.0;\nqreg a[2];\nqreg b[2];\ncx a,b;\n");
        assert_eq!(
            p.instrs,
            vec![gate("cx", &[], &[0, 2]), gate("cx", &[], &[1, 3])]
        );
    }

    #[test]
    fn broadcast_cx_reg_bit() {
        let p = parse("OPENQASM 2.0;\nqreg a[2];\nqreg b[2];\ncx a,b[1];\n");
        assert_eq!(
            p.instrs,
            vec![gate("cx", &[], &[0, 3]), gate("cx", &[], &[1, 3])]
        );
    }

    #[test]
    fn measure_broadcast_reg_to_reg() {
        // 'a' shifts the global offsets of 'q' to make sure they are used.
        let p = parse("OPENQASM 2.0;\nqreg a[1];\nqreg q[2];\ncreg c[2];\nmeasure q -> c;\n");
        assert_eq!(
            p.instrs,
            vec![
                Instr::Measure { qubit: 1, clbit: 0 },
                Instr::Measure { qubit: 2, clbit: 1 },
            ]
        );
    }

    #[test]
    fn builtin_u_and_cx() {
        let p = parse("OPENQASM 2.0;\nqreg q[2];\nU(0.1,0.2,0.3) q[0];\nCX q[0],q[1];\n");
        assert_eq!(
            p.instrs,
            vec![gate("u3", &[0.1, 0.2, 0.3], &[0]), gate("cx", &[], &[0, 1])]
        );
    }

    #[test]
    fn include_qelib1_is_internal() {
        let p = parse("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\nh q[0];\n");
        assert_eq!(p.instrs, vec![gate("h", &[], &[0])]);
    }

    #[test]
    fn comments_and_whitespace() {
        let src =
            "// leading comment\nOPENQASM 2.0; // trailing\n\n  qreg q[1];// c\n h\n q [ 0 ]\n ;\n";
        let p = parse(src);
        assert_eq!(p.instrs, vec![gate("h", &[], &[0])]);
    }

    #[test]
    fn empty_parens_on_parameterless_gate() {
        let p = parse("OPENQASM 2.0;\nqreg q[1];\nh() q[0];\n");
        assert_eq!(p.instrs, vec![gate("h", &[], &[0])]);
    }

    #[test]
    fn qreg_at_the_48_qubit_cap_parses() {
        let p = parse("OPENQASM 2.0;\nqreg q[48];\n");
        assert_eq!(p.n_qubits(), 48);
    }

    // -- positive: user-defined gates -----------------------------------------

    #[test]
    fn user_gate_with_params() {
        let src = "OPENQASM 2.0;\nqreg q[1];\n\
                   gate rot(t,s) a { rz(t/2) a; ry(s*2) a; rz(-t) a; }\n\
                   rot(pi,0.25) q[0];\n";
        let p = parse(src);
        let pi = std::f64::consts::PI;
        assert_eq!(
            p.instrs,
            vec![
                gate("rz", &[pi / 2.0], &[0]),
                gate("ry", &[0.5], &[0]),
                gate("rz", &[-pi], &[0]),
            ]
        );
    }

    #[test]
    fn user_gate_majority() {
        let src = "OPENQASM 2.0;\nqreg q[3];\n\
                   gate maj a,b,c { cx c,b; cx c,a; ccx a,b,c; }\n\
                   maj q[0],q[1],q[2];\n";
        let p = parse(src);
        assert_eq!(
            p.instrs,
            vec![
                gate("cx", &[], &[2, 1]),
                gate("cx", &[], &[2, 0]),
                gate("ccx", &[], &[0, 1, 2])
            ]
        );
    }

    #[test]
    fn user_gate_nesting() {
        let src = "OPENQASM 2.0;\nqreg q[3];\n\
                   gate maj a,b,c { cx c,b; cx c,a; ccx a,b,c; }\n\
                   gate twice a,b,c { maj a,b,c; maj c,b,a; }\n\
                   twice q[0],q[1],q[2];\n";
        let p = parse(src);
        assert_eq!(
            p.instrs,
            vec![
                gate("cx", &[], &[2, 1]),
                gate("cx", &[], &[2, 0]),
                gate("ccx", &[], &[0, 1, 2]),
                gate("cx", &[], &[0, 1]),
                gate("cx", &[], &[0, 2]),
                gate("ccx", &[], &[2, 1, 0]),
            ]
        );
    }

    #[test]
    fn user_gate_broadcasts_like_native() {
        let src = "OPENQASM 2.0;\nqreg q[2];\ngate flip a { x a; }\nflip q;\n";
        let p = parse(src);
        assert_eq!(p.instrs, vec![gate("x", &[], &[0]), gate("x", &[], &[1])]);
    }

    #[test]
    fn gate_body_barrier_expands() {
        let src = "OPENQASM 2.0;\nqreg q[2];\n\
                   gate f a,b { h a; barrier a,b; h b; }\n\
                   f q[1],q[0];\n";
        let p = parse(src);
        assert_eq!(
            p.instrs,
            vec![
                gate("h", &[], &[1]),
                Instr::Barrier(vec![1, 0]),
                gate("h", &[], &[0])
            ]
        );
    }

    #[test]
    fn deep_nesting_within_limit() {
        let mut src = String::from("OPENQASM 2.0;\nqreg q[1];\ngate g0 a { x a; }\n");
        for i in 1..=100 {
            src.push_str(&format!("gate g{i} a {{ g{} a; }}\n", i - 1));
        }
        src.push_str("g100 q[0];\n");
        let p = parse(&src);
        assert_eq!(p.instrs, vec![gate("x", &[], &[0])]);
    }

    // -- positive: expressions -------------------------------------------------

    fn param_of(expr: &str) -> f64 {
        let src = format!("OPENQASM 2.0;\nqreg q[1];\nu1({expr}) q[0];\n");
        let p = parse(&src);
        match &p.instrs[0] {
            Instr::Gate(g) => g.params[0],
            other => panic!("expected a gate, got {other:?}"),
        }
    }

    #[test]
    fn expr_pi_over_2() {
        assert_eq!(param_of("pi/2"), std::f64::consts::FRAC_PI_2);
    }

    #[test]
    fn expr_neg_pi() {
        assert_eq!(param_of("-pi"), -std::f64::consts::PI);
    }

    #[test]
    fn expr_pow_negative_exponent() {
        assert_eq!(param_of("2^-3"), 0.125);
    }

    #[test]
    fn expr_sin_pi_over_4() {
        let v = param_of("sin(pi/4)");
        assert!((v - std::f64::consts::FRAC_PI_4.sin()).abs() < 1e-15);
    }

    #[test]
    fn expr_precedence_and_functions() {
        assert_eq!(param_of("1+2*3"), 7.0);
        assert_eq!(param_of("2*3^2"), 18.0); // ^ binds tighter than *
        assert_eq!(param_of("-2^2"), -4.0); // ^ binds tighter than unary -
        assert_eq!(param_of("2^3^2"), 512.0); // ^ is right-associative
        assert_eq!(param_of("(1+2)*3"), 9.0);
        let v = param_of("ln(exp(1))+sqrt(4)-cos(0)+tan(0)");
        assert!((v - 2.0).abs() < 1e-12);
    }

    #[test]
    fn expr_scientific_notation() {
        assert_eq!(param_of("1.5e2"), 150.0);
        assert_eq!(param_of("2E-2"), 0.02);
        assert_eq!(param_of(".5"), 0.5);
    }

    // -- positive: measure / reset / barrier / if ------------------------------

    #[test]
    fn if_lowers_to_creg_index_not_offset() {
        let src = "OPENQASM 2.0;\nqreg q[1];\ncreg c0[2];\ncreg c1[3];\nif (c1 == 5) x q[0];\n";
        let p = parse(src);
        assert_eq!(
            p.instrs,
            vec![Instr::If {
                creg: 1, // index into Program.cregs — NOT the bit offset (2)
                value: 5,
                op: Box::new(gate("x", &[], &[0])),
            }]
        );
    }

    #[test]
    fn if_broadcasts_measure() {
        let src = "OPENQASM 2.0;\nqreg q[2];\ncreg c[2];\nif (c == 3) measure q -> c;\n";
        let p = parse(src);
        assert_eq!(
            p.instrs,
            vec![
                Instr::If {
                    creg: 0,
                    value: 3,
                    op: Box::new(Instr::Measure { qubit: 0, clbit: 0 }),
                },
                Instr::If {
                    creg: 0,
                    value: 3,
                    op: Box::new(Instr::Measure { qubit: 1, clbit: 1 }),
                },
            ]
        );
    }

    #[test]
    fn reset_broadcasts() {
        let p = parse("OPENQASM 2.0;\nqreg q[3];\nreset q;\nreset q[1];\n");
        assert_eq!(
            p.instrs,
            vec![
                Instr::Reset { qubit: 0 },
                Instr::Reset { qubit: 1 },
                Instr::Reset { qubit: 2 },
                Instr::Reset { qubit: 1 },
            ]
        );
    }

    #[test]
    fn barrier_union_dedupes() {
        let p = parse("OPENQASM 2.0;\nqreg a[2];\nqreg b[2];\nbarrier a[1], b, a;\n");
        assert_eq!(p.instrs, vec![Instr::Barrier(vec![1, 2, 3, 0])]);
    }

    // -- end-to-end -------------------------------------------------------------

    #[test]
    fn e2e_bell_counts() {
        let p = parse(BELL);
        let opts = crate::RunOptions {
            shots: 256,
            seed: Some(1),
            ..Default::default()
        };
        let r = crate::run_program(&p, &opts).expect("run");
        let keys: Vec<&str> = r.counts.0.keys().map(|s| s.as_str()).collect();
        assert_eq!(keys, ["00", "11"]);
    }

    // -- negative: every error names the token and carries line/col -------------

    #[test]
    fn err_unknown_gate() {
        assert_err(
            "OPENQASM 2.0;\nqreg q[1];\nfoo q[0];\n",
            3,
            1,
            "unknown gate 'foo'",
        );
    }

    #[test]
    fn err_unknown_gate_case_hint() {
        assert_err(
            "OPENQASM 2.0;\nqreg q[1];\nH q[0];\n",
            3,
            1,
            "did you mean 'h'",
        );
    }

    #[test]
    fn err_broadcast_size_mismatch() {
        assert_err(
            "OPENQASM 2.0;\nqreg a[2];\nqreg b[3];\ncx a,b;\n",
            4,
            1,
            "size mismatch",
        );
    }

    #[test]
    fn err_duplicate_qubit_after_broadcast() {
        assert_err(
            "OPENQASM 2.0;\nqreg q[2];\ncx q[1],q;\n",
            3,
            1,
            "duplicate qubit 'q[1]'",
        );
    }

    #[test]
    fn err_bad_version() {
        assert_err(
            "OPENQASM 3.0;\nqreg q[1];\n",
            1,
            10,
            "unsupported OpenQASM version '3.0'",
        );
    }

    #[test]
    fn err_unknown_include() {
        assert_err(
            "OPENQASM 2.0;\ninclude \"foo.inc\";\n",
            2,
            9,
            "only \"qelib1.inc\" is supported",
        );
    }

    #[test]
    fn err_param_count_mismatch() {
        assert_err(
            "OPENQASM 2.0;\nqreg q[1];\nrx q[0];\n",
            3,
            1,
            "expects 1 parameter(s), got 0",
        );
    }

    #[test]
    fn err_index_out_of_range() {
        assert_err(
            "OPENQASM 2.0;\nqreg q[2];\nh q[5];\n",
            3,
            5,
            "index 5 out of range for register 'q' of size 2",
        );
    }

    #[test]
    fn err_missing_semicolon() {
        assert_err(
            "OPENQASM 2.0;\nqreg q[1];\nh q[0]\nx q[0];\n",
            4,
            1,
            "expected ';'",
        );
    }

    #[test]
    fn err_too_many_qubits() {
        assert_err(
            "OPENQASM 2.0;\nqreg a[40];\nqreg b[9];\n",
            3,
            8,
            "maximum is 48",
        );
    }

    #[test]
    fn err_duplicate_register() {
        assert_err(
            "OPENQASM 2.0;\nqreg q[1];\ncreg q[1];\n",
            3,
            6,
            "already declared",
        );
    }

    #[test]
    fn err_opaque_unsupported() {
        assert_err("OPENQASM 2.0;\nopaque foo a;\n", 2, 1, "unsupported");
    }

    #[test]
    fn err_measure_size_mismatch() {
        assert_err(
            "OPENQASM 2.0;\nqreg q[3];\ncreg c[2];\nmeasure q -> c;\n",
            4,
            1,
            "measure size mismatch",
        );
    }

    #[test]
    fn err_if_indexed_creg() {
        assert_err(
            "OPENQASM 2.0;\nqreg q[1];\ncreg c[2];\nif (c[0] == 1) x q[0];\n",
            4,
            6,
            "whole classical register",
        );
    }

    #[test]
    fn err_redefine_native_gate() {
        assert_err("OPENQASM 2.0;\ngate h a { x a; }\n", 2, 6, "native gate");
    }

    #[test]
    fn err_measure_into_quantum_register() {
        assert_err(
            "OPENQASM 2.0;\nqreg q[1];\nqreg r[1];\nmeasure q -> r;\n",
            4,
            14,
            "'r' is a quantum register",
        );
    }

    #[test]
    fn err_unknown_identifier_in_expression() {
        assert_err(
            "OPENQASM 2.0;\nqreg q[1];\nrx(theta) q[0];\n",
            3,
            4,
            "unknown identifier 'theta'",
        );
    }

    #[test]
    fn err_measure_inside_gate_body() {
        assert_err(
            "OPENQASM 2.0;\ngate f a { measure a -> a; }\n",
            2,
            12,
            "not allowed inside a gate body",
        );
    }

    #[test]
    fn err_index_inside_gate_body() {
        assert_err(
            "OPENQASM 2.0;\ngate f a { x a[0]; }\n",
            2,
            15,
            "inside a gate body",
        );
    }

    #[test]
    fn err_zero_size_register() {
        assert_err("OPENQASM 2.0;\nqreg q[0];\n", 2, 8, "at least 1");
    }

    #[test]
    fn err_missing_header() {
        assert_err("qreg q[1];\n", 1, 1, "expected 'OPENQASM 2.0;'");
    }

    #[test]
    fn err_recursive_gate_definition() {
        assert_err(
            "OPENQASM 2.0;\ngate f a { f a; }\n",
            2,
            12,
            "recursive gate definitions are not allowed",
        );
    }

    #[test]
    fn err_expansion_depth_limit() {
        let mut src = String::from("OPENQASM 2.0;\nqreg q[1];\ngate g0 a { x a; }\n");
        for i in 1..=200 {
            src.push_str(&format!("gate g{i} a {{ g{} a; }}\n", i - 1));
        }
        src.push_str("g200 q[0];\n");
        let e = parse_str(&src).expect_err("expected depth error");
        assert!(
            e.msg.contains("nesting depth limit of 128"),
            "msg: {}",
            e.msg
        );
        assert_eq!((e.line, e.col), (204, 1));
    }

    #[test]
    fn err_single_equals() {
        assert_err(
            "OPENQASM 2.0;\nqreg q[1];\ncreg c[1];\nif (c = 1) x q[0];\n",
            4,
            7,
            "'=='",
        );
    }

    #[test]
    fn err_barrier_under_if() {
        assert_err(
            "OPENQASM 2.0;\nqreg q[1];\ncreg c[1];\nif (c == 1) barrier q;\n",
            4,
            13,
            "'barrier' cannot be the body",
        );
    }
}
