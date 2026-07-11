// grover.qasm — Grover search on 3 qubits, marked state |101>, 2 iterations
//
// Demonstrates: a phase oracle, the diffuser (inversion about the mean), and
// a user-defined gate (ccz built from h + ccx). With N = 8 and one marked
// state, the amplitude angle is t = asin(1/sqrt(8)), and after 2 iterations
// the success probability is sin^2(5t) ~ 0.9453.
// Expected output: key "101" (c[2] c[1] c[0] left to right) ~94.5% of shots;
// the other seven keys share the remaining ~5.5% (~0.8% each).
//
// Run: kuantum run examples/grover.qasm --shots 4000 --seed 11

OPENQASM 2.0;
include "qelib1.inc";

// Doubly-controlled Z: flips the phase of |111> on its three arguments.
// Symmetric, so any argument order works. Built as h·ccx·h on the target.
gate ccz a, b, c {
    h c;
    ccx a, b, c;
    h c;
}

qreg q[3];
creg c[3];

// Uniform superposition over all 8 basis states.
h q;

// ---- Grover iteration 1 -----------------------------------------------------
// Oracle for |101> (q[2]=1, q[1]=0, q[0]=1): map the marked state to |111>
// with X on the zero-position q[1], phase-flip |111> with ccz, undo the X.
x q[1];
ccz q[0], q[1], q[2];
x q[1];
// Diffuser: 2|s><s| - I, i.e. a phase flip on |000> conjugated by H^3.
h q;
x q;
ccz q[0], q[1], q[2];
x q;
h q;

// ---- Grover iteration 2 (same oracle + diffuser) ----------------------------
x q[1];
ccz q[0], q[1], q[2];
x q[1];
h q;
x q;
ccz q[0], q[1], q[2];
x q;
h q;

measure q -> c;
