#!/usr/bin/env python
"""qiskit-aer oracle for zeno's trajectory-sampled noise (tests/noise.rs).

Reads a JSON spec on stdin:

    {
      "shots": int,
      "seed": int,
      "model": { the seven zeno NoiseModel fields },
      "circuits": [{"name": str, "n": int,
                    "gates": [[gate_name, [qubits...]], ...]}, ...]
    }

Builds the SAME noise semantics with qiskit-aer primitives (NoiseModel,
pauli_error, amplitude_damping_error, ReadoutError), runs each circuit
with trailing measurements of every qubit, and prints
{"<name>": {"<bitstring>": count}} on stdout. Counts keys follow the
qiskit convention (bit n-1 ... bit 0), which matches zeno's single-creg
key format exactly.
"""

import json
import sys
from itertools import product

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    amplitude_damping_error,
    pauli_error,
)


def uniform_depolarizing(p, k):
    """zeno convention: w.p. `p` apply a uniform NON-IDENTITY Pauli string
    (4^k - 1 options, each p / (4^k - 1)).

    This equals aer's depolarizing_error(lam, k) with
    lam = p * 4^k / (4^k - 1); it is built explicitly from pauli_error to
    sidestep the parameter-convention trap.
    """
    strings = ["".join(s) for s in product("IXYZ", repeat=k)]
    probs = [("I" * k, 1.0 - p)] + [
        (s, p / (4**k - 1)) for s in strings if set(s) != {"I"}
    ]
    return pauli_error(probs)


def qubit_trio(m):
    """zeno's per-qubit channel order after the depolarizing step: bit
    flip, then phase flip, then amplitude damping. QuantumError.compose
    applies self first, then the argument — matching zeno's order.
    """
    err = pauli_error([("X", m["bit_flip"]), ("I", 1.0 - m["bit_flip"])])
    err = err.compose(pauli_error([("Z", m["phase_flip"]), ("I", 1.0 - m["phase_flip"])]))
    err = err.compose(amplitude_damping_error(m["amplitude_damping"]))
    return err


def main():
    spec = json.load(sys.stdin)
    m = spec["model"]

    trio = qubit_trio(m)
    err1 = uniform_depolarizing(m["depolarizing_1q"], 1).compose(trio)
    # Per-qubit channels on distinct qubits commute, so the tensor order
    # (identical trios) is immaterial.
    err2 = uniform_depolarizing(m["depolarizing_2q"], 2).compose(trio.tensor(trio))

    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(err1, ["x", "y", "z", "h", "s", "t", "rz"])
    nm.add_all_qubit_quantum_error(err2, ["cx", "cz", "swap"])
    nm.add_all_qubit_readout_error(
        ReadoutError(
            [
                [1.0 - m["readout_flip_0to1"], m["readout_flip_0to1"]],
                [m["readout_flip_1to0"], 1.0 - m["readout_flip_1to0"]],
            ]
        )
    )

    sim = AerSimulator(noise_model=nm, seed_simulator=spec["seed"])
    out = {}
    for c in spec["circuits"]:
        qc = QuantumCircuit(c["n"], c["n"])
        for name, qubits in c["gates"]:
            getattr(qc, name)(*qubits)
        qc.measure(range(c["n"]), range(c["n"]))
        counts = sim.run(qc, shots=spec["shots"]).result().get_counts()
        out[c["name"]] = {k: int(v) for k, v in counts.items()}
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
