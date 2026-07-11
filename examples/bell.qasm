// bell.qasm — Bell pair (maximally entangled two-qubit state)
//
// Demonstrates: the smallest interesting circuit. H puts q[0] into an equal
// superposition, CX copies that superposition into entanglement, so the two
// bits are perfectly correlated shot-to-shot.
// Expected output: only the keys "00" and "11", ~50% each (never "01"/"10").
//
// Run: zeno run examples/bell.qasm --shots 4000 --seed 11

OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

h q[0];        // (|0> + |1>)/sqrt(2) on q[0]
cx q[0], q[1]; // entangle: (|00> + |11>)/sqrt(2)

measure q -> c;
