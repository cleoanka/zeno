// bernstein_vazirani.qasm — Bernstein–Vazirani, hidden string s = 110101
//
// Demonstrates: recovering a hidden 6-bit string from ONE oracle call.
// The oracle computes f(x) = s.x (mod 2) into an ancilla prepared in |->,
// which kicks the phase (-1)^(s.x) back onto the data register; the final
// H layer turns that phase pattern into the string itself.
// Expected output: the single key "110101", 100% of shots (deterministic;
// c[5] prints leftmost, so the key reads s bit 5 .. bit 0).
//
// Run: target/debug/kuantum run examples/bernstein_vazirani.qasm --shots 4000 --seed 11

OPENQASM 2.0;
include "qelib1.inc";

qreg q[6];   // data register, q[i] learns bit i of s
qreg anc[1]; // oracle ancilla (never measured)
creg c[6];

// Ancilla to |-> so the oracle's XOR becomes a phase kickback.
x anc[0];
h anc[0];

// Data register into uniform superposition.
h q;

// Oracle U_f |x>|y> = |x>|y XOR s.x> for s = 110101:
// one CX per set bit of s (bits 0, 2, 4, 5). Zero bits (1, 3) get no gate.
cx q[0], anc[0];
cx q[2], anc[0];
cx q[4], anc[0];
cx q[5], anc[0];

// Undo the H layer: the register collapses onto |s> exactly.
h q;

measure q -> c;
