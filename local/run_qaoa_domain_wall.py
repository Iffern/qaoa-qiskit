import argparse
import datetime
import json
import math
import os
from collections import defaultdict

import numpy as np
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Sampler
from qiskit_ibm_provider import IBMProvider
from qiskit_optimization.converters import QuadraticProgramToQubo

from QAOA import QAOA
from MinimumEigenOptimizer import MinimumEigenOptimizer
from WarmStartQAOAOptimizer import WarmStartQAOAOptimizer
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B
from RecursiveMinimumEigenOptimizer import RecursiveMinimumEigenOptimizer
from CustomInitialState import CustomInitialState

from NumpyMinimumEigensolver import NumPyMinimumEigensolver
from polynomial_program import PolynomialProgram
from workflows_domain_wall_encoding import get_cost_model, get_deadline_model, get_machine_usage_model, is_correct, \
    get_quadratic_problem, get_stats_for_result, get_indices
from qiskit.opflow import SummedOp, Z, I, X


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if np.iscomplexobj(obj):
            return obj.real
        return json.JSONEncoder.default(self, obj)


class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super().__call__(parser, namespace, values, option_string)


def get_filename(args):
    filename = ""
    if args.warm_start and args.recursive:
        filename += "WS-R-QAOA_"
    elif args.warm_start:
        filename += "WS-QAOA_"
    elif args.recursive:
        filename += "R-QAOA_"
    else:
        filename += "QAOA_"
    suffix = 'custom_mixer_' if args.custom_mixer else ''
    suffix += datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename += f'AerSimulator_{args.optimizer}_{str(args.reps)}_Workflows_domain_wall_{suffix}.json'
    return filename


parser = argparse.ArgumentParser(
    prog='QAOA runner',
    description='Runs selected QAOA variant on problem instance')

parser.add_argument('--warm_start', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--recursive', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("-reps", default=1, type=int)
parser.add_argument("-min_num_vars", default=3, type=int)
parser.add_argument('-initial_point', action=Store_as_array, type=float, nargs='+')
parser.add_argument("-optimization_level", default=3, type=int)
parser.add_argument("-resilience_level", default=1, type=int)
parser.add_argument("-optimizer", default='COBYLA', type=str)
parser.add_argument("-step", default=0.01, type=float)
parser.add_argument("-num_initial_solutions", default=1, type=int)
parser.add_argument("-num_experiments", default=1, type=int)
parser.add_argument("--custom_mixer", default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

if args.optimizer == 'COBYLA':
    optimizer = COBYLA()
elif args.optimizer == 'L_BFGS_B':
    optimizer = L_BFGS_B(eps=args.step)

pre_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver(res_filter=is_correct))

history = {"nfevs": [], "params": [], "energy": [], "metadata": []}

A = 1
B = 20
C = 10

pp = PolynomialProgram(10)

pp.add_objective(get_cost_model(pp.x), A)
pp.add_objective(get_machine_usage_model(pp.x), B)
pp.add_objective(get_deadline_model(pp.x), C)

qubit_op, offset = pp.to_ising()

quadratic_program = QuadraticProgramToQubo().convert(get_quadratic_problem())

if args.initial_point is None or len(args.initial_point) != 2:
    initial_point = args.initial_point
else:
    initial_point = np.tile(args.initial_point, args.reps)

mixer = None
if args.custom_mixer:
    mixer = SummedOp([
        - I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ X,
        - I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ Z ^ X,
        I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ X ^ Z,
        - I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ X ^ I,
        - I ^ I ^ I ^ I ^ I ^ I ^ I ^ X ^ I ^ I,
        - I ^ I ^ I ^ I ^ I ^ I ^ Z ^ X ^ I ^ I,
        I ^ I ^ I ^ I ^ I ^ I ^ X ^ Z ^ I ^ I,
        - I ^ I ^ I ^ I ^ I ^ I ^ X ^ I ^ I ^ I,
        - I ^ I ^ I ^ I ^ I ^ X ^ I ^ I ^ I ^ I,
        - I ^ I ^ I ^ I ^ Z ^ X ^ I ^ I ^ I ^ I,
        I ^ I ^ I ^ I ^ X ^ Z ^ I ^ I ^ I ^ I,
        - I ^ I ^ I ^ I ^ X ^ I ^ I ^ I ^ I ^ I,
        I ^ I ^ I ^ X ^ I ^ I ^ I ^ I ^ I ^ I,
        I ^ I ^ X ^ I ^ I ^ I ^ I ^ I ^ I ^ I,
        I ^ X ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I,
        X ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I,
    ])

initial_state = None
if args.custom_mixer:
    indices = get_indices()
    initial_state = np.zeros(2 ** qubit_op.num_qubits, int)
    for idx in indices:
        initial_state[idx] = 1
    initial_state = CustomInitialState(qubit_op.num_qubits, state_vector=initial_state).construct_circuit()

def store_history_and_forward(nfevs, params, energy, meta):
    # store information
    history["nfevs"].append(nfevs)
    history["params"].append(params)
    history["energy"].append(energy)
    history["metadata"].append(meta)

fvals = defaultdict(int)
results = defaultdict(int)
binary_strings = []
probabilities = []
optimals = []
corrects = []
feasibles = []
incorrects = []
correct_configs = []
incorrect_configs = []
feasible_configs = []
nfevs = []
energies = []

def solve_and_store_result(qaoa):
    result = None
    try:
        result = qaoa.solve_hamiltonian(qubit_op, offset, quadratic_program)
    finally:
        x = None if result is None else result.x
        fval = None if result is None else result.fval
        x = ''.join(str(math.floor(x_i)) for x_i in x)

        fvals[fval] += 1
        results[x] += 1

        samples = []

        for sample in result.samples:
            samples.append(
                {"fval": float(sample.fval), "x": sample.x.tolist(), "probability": float(sample.probability)})

        dic_res = {}
        for s in samples:
            xes = s['x']
            xes.reverse()
            x = ''.join(str(math.floor(x)) for x in xes)
            dic_res[x] = s['probability']
        most_likely, optimal, correct, feasible, incorrect, correct_config, incorrect_config, feasible_config = \
            get_stats_for_result(dic_res)
        binary_string, prob = most_likely
        binary_strings.append(binary_string)
        probabilities.append(prob)
        optimals.append(optimal)
        corrects.append(correct)
        feasibles.append(feasible)
        incorrects.append(incorrect)
        correct_configs.append(correct_config)
        incorrect_configs.append(incorrect_config)
        feasible_configs.append(feasible_config)
        nfevs.append(len(history["nfevs"]))
        energies.append(min(history['energy']))


backend = IBMProvider(token=os.getenv("IBM_QUANTUM_TOKEN")).get_backend('')
noise_model = NoiseModel.from_backend(backend)
backend_options = {"noise_model": noise_model}
qaoa_mes = QAOA(Sampler(backend_options=backend_options), optimizer=optimizer, reps=args.reps,
                initial_point=initial_point, callback=store_history_and_forward, initial_state=initial_state,
                mixer=mixer)
if args.warm_start and args.recursive:
    qaoa = RecursiveMinimumEigenOptimizer(WarmStartQAOAOptimizer(qaoa=qaoa_mes,
                                                                 pre_solver=pre_solver,
                                                                 relax_for_pre_solver=False,
                                                                 num_initial_solutions=args.num_initial_solutions),
                                          min_num_vars=args.min_num_vars)
elif args.warm_start:
    qaoa = WarmStartQAOAOptimizer(qaoa=qaoa_mes, pre_solver=pre_solver,
                                  relax_for_pre_solver=False, num_initial_solutions=args.num_initial_solutions)
elif args.recursive:
    qaoa = RecursiveMinimumEigenOptimizer(MinimumEigenOptimizer(qaoa_mes), min_num_vars=args.min_num_vars)
else:
    qaoa = MinimumEigenOptimizer(qaoa_mes)

for i in range(0, args.num_experiments):
    history = {"nfevs": [], "params": [], "energy": [], "metadata": []}
    print('Experiment number ' + str(i))
    solve_and_store_result(qaoa)

with open(f'results/{get_filename(args)}', "w") as f:
    serialized_inputs = {
        "initial_point": initial_point if initial_point is None else list(initial_point),
        "optimizer": {
            "__class__.__name__": optimizer.__class__.__name__,
            "__class__": str(optimizer.__class__),
            "settings": getattr(optimizer, "settings", {}),
        },
        "reps": args.reps,
        "warm_start": args.warm_start,
        "recursive": args.recursive,
        "backend": "AerSimulator",
        "optimization_level": args.optimization_level,
        "resilience_level": args.resilience_level,
        "num_initial_solutions": args.num_initial_solutions
    }
    output = {
        "fvals": fvals,
        "results": results,
        "optimals": optimals,
        "probabilities": probabilities,
        "binary_strings": binary_strings,
        "corrects": corrects,
        "feasibles": feasibles,
        "incorrects": incorrects,
        "correct_configs": correct_configs,
        "incorrect_configs": incorrect_configs,
        "feasible_configs": feasible_configs,
        "nfevs": nfevs,
        "energies": energies
    }
    json.dump(output, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
