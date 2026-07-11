// qft.qasm — 8-qubit QFT, then its exact inverse: a self-checking round trip
//
// Demonstrates: the quantum Fourier transform (H + controlled-phase ladder +
// bit-reversal swaps) and how to invert a circuit — same gates in REVERSE
// order with NEGATED cp angles (h and swap are self-inverse). We prepare the
// basis state |10110001> (X on q[7], q[5], q[4], q[0]), Fourier-transform it,
// transform back, and measure: the input must come back exactly.
// Expected output: the single key "10110001", 100% of shots (deterministic;
// the key prints c[7] leftmost .. c[0] rightmost, matching q[7]..q[0]).
//
// Run: target/debug/kuantum run examples/qft.qasm --shots 4000 --seed 11

OPENQASM 2.0;
include "qelib1.inc";

qreg q[8];
creg c[8];

// --- Prepare |10110001>: set the qubits that print as '1' -------------------
x q[0];
x q[4];
x q[5];
x q[7];

// --- Forward QFT ------------------------------------------------------------
// For j = 7 .. 0: H on q[j], then cp(pi/2^(j-k)) from each lower qubit q[k].
// The final swaps perform the QFT's bit reversal.
h q[7];
cp(pi/2) q[6], q[7];
cp(pi/4) q[5], q[7];
cp(pi/8) q[4], q[7];
cp(pi/16) q[3], q[7];
cp(pi/32) q[2], q[7];
cp(pi/64) q[1], q[7];
cp(pi/128) q[0], q[7];
h q[6];
cp(pi/2) q[5], q[6];
cp(pi/4) q[4], q[6];
cp(pi/8) q[3], q[6];
cp(pi/16) q[2], q[6];
cp(pi/32) q[1], q[6];
cp(pi/64) q[0], q[6];
h q[5];
cp(pi/2) q[4], q[5];
cp(pi/4) q[3], q[5];
cp(pi/8) q[2], q[5];
cp(pi/16) q[1], q[5];
cp(pi/32) q[0], q[5];
h q[4];
cp(pi/2) q[3], q[4];
cp(pi/4) q[2], q[4];
cp(pi/8) q[1], q[4];
cp(pi/16) q[0], q[4];
h q[3];
cp(pi/2) q[2], q[3];
cp(pi/4) q[1], q[3];
cp(pi/8) q[0], q[3];
h q[2];
cp(pi/2) q[1], q[2];
cp(pi/4) q[0], q[2];
h q[1];
cp(pi/2) q[0], q[1];
h q[0];
swap q[0], q[7];
swap q[1], q[6];
swap q[2], q[5];
swap q[3], q[4];

barrier q; // forward QFT done — state is now the Fourier transform of the input

// --- Inverse QFT: the forward list reversed, every cp angle negated ---------
swap q[3], q[4];
swap q[2], q[5];
swap q[1], q[6];
swap q[0], q[7];
h q[0];
cp(-pi/2) q[0], q[1];
h q[1];
cp(-pi/4) q[0], q[2];
cp(-pi/2) q[1], q[2];
h q[2];
cp(-pi/8) q[0], q[3];
cp(-pi/4) q[1], q[3];
cp(-pi/2) q[2], q[3];
h q[3];
cp(-pi/16) q[0], q[4];
cp(-pi/8) q[1], q[4];
cp(-pi/4) q[2], q[4];
cp(-pi/2) q[3], q[4];
h q[4];
cp(-pi/32) q[0], q[5];
cp(-pi/16) q[1], q[5];
cp(-pi/8) q[2], q[5];
cp(-pi/4) q[3], q[5];
cp(-pi/2) q[4], q[5];
h q[5];
cp(-pi/64) q[0], q[6];
cp(-pi/32) q[1], q[6];
cp(-pi/16) q[2], q[6];
cp(-pi/8) q[3], q[6];
cp(-pi/4) q[4], q[6];
cp(-pi/2) q[5], q[6];
h q[6];
cp(-pi/128) q[0], q[7];
cp(-pi/64) q[1], q[7];
cp(-pi/32) q[2], q[7];
cp(-pi/16) q[3], q[7];
cp(-pi/8) q[4], q[7];
cp(-pi/4) q[5], q[7];
cp(-pi/2) q[6], q[7];
h q[7];

measure q -> c; // must read back the input: 10110001
