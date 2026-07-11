// ghz.qasm — 24-qubit GHZ state (Greenberger–Horne–Zeilinger)
//
// Demonstrates: large-scale entanglement and the fusion compiler on a 2^24
// (16M-amplitude) state vector. One H fans out through a CX chain so all 24
// qubits collapse together.
// Expected output: exactly two keys, "000000000000000000000000" and
// "111111111111111111111111", ~50% each — nothing in between.
//
// Run: zeno run examples/ghz.qasm --shots 4000 --seed 11

OPENQASM 2.0;
include "qelib1.inc";

qreg q[24];
creg c[24];

h q[0]; // superposition on the seed qubit

// CX chain propagates the superposition into a 24-qubit cat state:
// (|0...0> + |1...1>)/sqrt(2)
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
cx q[3], q[4];
cx q[4], q[5];
cx q[5], q[6];
cx q[6], q[7];
cx q[7], q[8];
cx q[8], q[9];
cx q[9], q[10];
cx q[10], q[11];
cx q[11], q[12];
cx q[12], q[13];
cx q[13], q[14];
cx q[14], q[15];
cx q[15], q[16];
cx q[16], q[17];
cx q[17], q[18];
cx q[18], q[19];
cx q[19], q[20];
cx q[20], q[21];
cx q[21], q[22];
cx q[22], q[23];

measure q -> c;
