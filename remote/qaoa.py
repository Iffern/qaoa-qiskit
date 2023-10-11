from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.opflow import PauliSumOp
from qiskit.utils import QuantumInstance
from qiskit.exceptions import QiskitError
from qiskit.utils.mitigation import CompleteMeasFitter
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import WarmStartQAOAOptimizer, RecursiveMinimumEigenOptimizer, \
    MinimumEigenOptimizer, GoemansWilliamsonOptimizer


class Publisher:
    def __init__(self, messenger):
        self._messenger = messenger

    def callback(self, *args, **kwargs):
        text = list(args)
        for k, v in kwargs.items():
            text.append({k: v})
        self._messenger.publish(text)


def main(backend, user_messenger, **kwargs):
    mandatory = {"operator"}
    missing = mandatory - set(kwargs.keys())
    if len(missing) > 0:
        raise ValueError(f"The following mandatory arguments are missing: {missing}.")

    serialized_inputs = {}

    operator = kwargs["operator"]

    if not isinstance(operator, PauliSumOp):
        try:
            operator = PauliSumOp.from_list([(str(operator), 1)])
        except QiskitError as err:
            raise QiskitError(
                f"Cannot convert {operator} of type {type(operator)} to a PauliSumOp."
            ) from err

    serialized_inputs["operator"] = operator.primitive.to_list()

    offset = kwargs.get("offset", None)
    serialized_inputs["offset"] = offset

    initial_point = kwargs.get("initial_point", None)
    serialized_inputs["initial_point"] = initial_point if initial_point is None else list(initial_point)

    optimizer = kwargs.get("optimizer", None)
    serialized_inputs["optimizer"] = {
        "__class__.__name__": optimizer.__class__.__name__,
        "__class__": str(optimizer.__class__),
        "settings": getattr(optimizer, "settings", {}),
    }

    pre_solver_str = kwargs.get("pre_solver", None)

    if pre_solver_str == "GoemansWilliamsonOptimizer":
        pre_solver = GoemansWilliamsonOptimizer(num_cuts=3)
    else:
        pre_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())

    reps = kwargs.get("reps", 1)
    serialized_inputs["reps"] = reps

    shots = kwargs.get("shots", 1024)
    serialized_inputs["shots"] = shots

    alpha = kwargs.get("alpha", 1.0)  # CVaR expectation
    serialized_inputs["alpha"] = alpha

    optimization_level = kwargs.get("optimization_level", 1)
    serialized_inputs["optimization_level"] = optimization_level

    warm_start = kwargs.get("warm_start", False)
    serialized_inputs["warm_start"] = warm_start

    recursive = kwargs.get("recursive", False)
    serialized_inputs["recursive"] = recursive

    min_num_vars = kwargs.get("min_num_vars", 1)
    serialized_inputs["min_num_vars"] = min_num_vars

    measurement_error_mitigation = kwargs.get("measurement_error_mitigation", False)
    serialized_inputs["measurement_error_mitigation"] = measurement_error_mitigation

    if measurement_error_mitigation:
        # allow for TensoredMeasFitter as soon as runtime runs on latest Terra
        measurement_error_mitigation_cls = CompleteMeasFitter
        measurement_error_mitigation_shots = shots
    else:
        measurement_error_mitigation_cls = None
        measurement_error_mitigation_shots = None

    # set up quantum instance
    quantum_instance = QuantumInstance(
        backend,
        shots=shots,
        optimization_level=optimization_level,
        measurement_error_mitigation_cls=measurement_error_mitigation_cls,
        measurement_error_mitigation_shots=measurement_error_mitigation_shots
    )

    # publisher for user-server communication
    publisher = Publisher(user_messenger)

    # dictionary to store the history of the optimization
    history = {"nfevs": [], "params": [], "energy": [], "std": []}

    def store_history_and_forward(nfevs, params, energy, std):
        # store information
        history["nfevs"].append(nfevs)
        history["params"].append(params)
        history["energy"].append(energy)
        history["std"].append(std)

        # and forward information to users callback
        publisher.callback(nfevs, params, energy, std)

    qaoa_mes = QAOA(optimizer=optimizer,
                    reps=reps,
                    initial_point=initial_point,
                    callback=store_history_and_forward,
                    quantum_instance=quantum_instance)
    pre_solver = pre_solver
    if warm_start and recursive:
        qaoa = RecursiveMinimumEigenOptimizer(WarmStartQAOAOptimizer(qaoa=qaoa_mes,
                                                                     pre_solver=pre_solver,
                                                                     relax_for_pre_solver=False),
                                              min_num_vars=min_num_vars)
    elif warm_start:
        qaoa = WarmStartQAOAOptimizer(qaoa=qaoa_mes, pre_solver=pre_solver, relax_for_pre_solver=False)
    elif recursive:
        qaoa = RecursiveMinimumEigenOptimizer(MinimumEigenOptimizer(qaoa_mes), min_num_vars=min_num_vars)
    else:
        qaoa = MinimumEigenOptimizer(qaoa_mes)

    qp = QuadraticProgram()
    qp.from_ising(operator, offset)
    result = qaoa.solve(qp)

    parsed_samples = []

    for sample in result.samples:
        parsed_samples.append(
            {"fval": float(sample.fval), "x": sample.x.tolist(), "probability": float(sample.probability)})

    serialized_result = {
        "x": result.x,
        "fval": result.fval,
        "raw_result": result.raw_results,
        "samples": parsed_samples,
        "optimizer_history": history,
        "inputs": serialized_inputs,
    }

    return serialized_result