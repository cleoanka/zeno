// teleport.qasm — quantum teleportation of |1> with classical feed-forward
//
// Demonstrates: a dynamic circuit — mid-circuit measurement into three 1-bit
// cregs and if()-conditioned X/Z corrections. Alice's state |1> on q[0] is
// destroyed by the Bell measurement and reappears on Bob's q[2], so the
// output bit is always 1 whatever the two random correction bits say.
// Expected output: four keys "1 0 0", "1 0 1", "1 1 0", "1 1 1", ~25% each.
// (Key = "out m1 m0": last-declared creg leftmost; out is always 1.)
//
// Run: kuantum run examples/teleport.qasm --shots 4000 --seed 11

OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];   // q[0]: Alice's payload, q[1]: Alice's Bell half, q[2]: Bob
creg m0[1];  // Alice's measurement of q[0] (drives the Z correction)
creg m1[1];  // Alice's measurement of q[1] (drives the X correction)
creg out[1]; // Bob's readout of the teleported state

// Payload: prepare |1> on q[0] (any state works; |1> makes the check crisp).
x q[0];

// Shared Bell pair between Alice (q[1]) and Bob (q[2]).
h q[1];
cx q[1], q[2];

// Alice's Bell-basis measurement of payload + her Bell half.
cx q[0], q[1];
h q[0];
measure q[0] -> m0[0];
measure q[1] -> m1[0];

// Bob's classically controlled corrections (feed-forward).
if (m1 == 1) x q[2];
if (m0 == 1) z q[2];

// Bob now holds the payload: this bit is deterministically 1.
measure q[2] -> out[0];
