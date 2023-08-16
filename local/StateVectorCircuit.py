from qiskit import QiskitError, QuantumCircuit
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.transpiler.passes import Unroller
from qiskit.utils.arithmetic import is_power_of_2, log2, normalize_vector


def convert_to_basis_gates(circuit):
    unroller = Unroller(basis=['u', 'cx'])
    return dag_to_circuit(unroller.run(circuit_to_dag(circuit)))


class StateVectorCircuit:
    def __init__(self, state_vector):
        if not is_power_of_2(len(state_vector)):
            raise QiskitError('The length of the input state vector needs to be a power of 2.')
        self._num_qubits = log2(len(state_vector))
        self._state_vector = normalize_vector(state_vector)

    def construct_circuit(self, circuit, register):
        temp = QuantumCircuit(register)
        if len(register) < self._num_qubits:
            raise QiskitError('Insufficient register are provided for the intended state-vector.')

        temp.initialize(self._state_vector, [register[i] for i in range(self._num_qubits)])
        temp = convert_to_basis_gates(temp)
        temp.data = [g for g in temp.data if not g[0].name == 'reset']
        circuit.compose(temp, inplace=True)
        return circuit
