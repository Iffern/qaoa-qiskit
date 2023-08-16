import numpy as np
from qiskit import QiskitError, QuantumCircuit, QuantumRegister
from qiskit.utils.arithmetic import normalize_vector
from qiskit.utils.validation import validate_min

from StateVectorCircuit import StateVectorCircuit


class CustomInitialState:
    def __init__(self, num_qubits: int, state_vector: np.ndarray = None):
        validate_min('num_qubits', num_qubits, 1)
        self._num_qubits = num_qubits
        self._circuit = None
        if len(state_vector) != np.power(2, self._num_qubits):
            raise QiskitError('The state vector length {} is incompatible with '
                              'the number of qubits {}'.format(len(state_vector), self._num_qubits))
        self._state_vector = normalize_vector(state_vector)
        self._state = None

    def construct_circuit(self):
        circuit = QuantumCircuit()
        register = QuantumRegister(self._num_qubits, name='q')
        circuit.add_register(register)
        svc = StateVectorCircuit(self._state_vector)
        svc.construct_circuit(circuit=circuit, register=register)
        self._circuit = circuit
        return self._circuit.copy()
