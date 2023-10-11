import argparse
import datetime
import json
import math
import os
import pickle

import numpy as np
from qiskit_ibm_provider import IBMProvider
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Sampler
from qiskit_optimization.algorithms import GoemansWilliamsonOptimizer, RecursiveMinimumEigenOptimizer
from qiskit_optimization.applications import Maxcut

from QAOA import QAOA
from MinimumEigenOptimizer import MinimumEigenOptimizer
from WarmStartQAOAOptimizer import WarmStartQAOAOptimizer
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B


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
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename += f'AerSimulator_{args.optimizer}_1_MaxCut_grid_{args.graph.replace(".pickle", "")}_{suffix}.json'
    return filename


parser = argparse.ArgumentParser(
    prog='QAOA runner',
    description='Runs selected QAOA variant on problem instance')

parser.add_argument('--warm_start', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--recursive', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("-min_num_vars", default=3, type=int)
parser.add_argument('-initial_gamma_position', type=float)
parser.add_argument("-optimization_level", default=3, type=int)
parser.add_argument("-resilience_level", default=1, type=int)
parser.add_argument("-optimizer", default='COBYLA', type=str)
parser.add_argument("-step", default=0.01, type=float)
parser.add_argument("-graph", type=str)
parser.add_argument("-num_initial_solutions", default=3, type=int)
parser.add_argument("-num_cuts", default=3, type=int)
parser.add_argument("-beta_step", type=float)
args = parser.parse_args()

step = args.beta_step

if args.optimizer == 'COBYLA':
    optimizer = COBYLA()
elif args.optimizer == 'L_BFGS_B':
    optimizer = L_BFGS_B(eps=args.step)

pre_solver = GoemansWilliamsonOptimizer(num_cuts=args.num_cuts)

history = {"nfevs": [], "params": [], "energy": [], "metadata": []}

G = pickle.load(open(args.graph, 'rb'))
max_cut = Maxcut(G)
qp = max_cut.to_quadratic_program()


def store_history_and_forward(nfevs, params, energy, meta):
    # store information
    history["nfevs"].append(nfevs)
    history["params"].append(params)
    history["energy"].append(energy)
    history["metadata"].append(meta)


final_result = {}


def solve_and_store_result(qaoa, beta):
    result = None
    try:
        result = qaoa.solve(qp)
    finally:
        x = None if result is None else result.x
        fval = None if result is None else result.fval
        x = ''.join(str(math.floor(x_i)) for x_i in x)

        final_result[beta] = {
            'x': x,
            'fval': fval,
            'min_energy': min(history['energy']),
            'nfev': len(history["nfevs"])
        }


backend = IBMProvider(token=os.getenv("IBM_QUANTUM_TOKEN")).get_backend('')
noise_model = NoiseModel.from_backend(backend)
backend_options = {"noise_model": noise_model}

gamma_bounds = [(-2 * np.pi, 2 * np.pi)]
beta_bounds = [(-2 * np.pi, 2 * np.pi)]

gamma = -2 * np.pi + args.initial_gamma_position * step

for beta in np.arange(-2 * np.pi, 2 * np.pi, step):
    history = {"nfevs": [], "params": [], "energy": [], "metadata": []}
    qaoa_mes = QAOA(Sampler(backend_options=backend_options), optimizer=optimizer, reps=1,
                    initial_point=[beta, gamma], callback=store_history_and_forward)
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
    print('Beta ' + str(beta))
    solve_and_store_result(qaoa, beta)

with open(f'results/{get_filename(args)}', "w") as f:
    serialized_inputs = {
        "optimizer": {
            "__class__.__name__": optimizer.__class__.__name__,
            "__class__": str(optimizer.__class__),
            "settings": getattr(optimizer, "settings", {}),
        },
        "reps": 1,
        "warm_start": args.warm_start,
        "recursive": args.recursive,
        "backend": "AerSimulator",
        "optimization_level": args.optimization_level,
        "resilience_level": args.resilience_level,
        "num_initial_solutions": args.num_initial_solutions,
        "graph": args.graph,
    }
    output = {
        'gamma': gamma,
        'beta_results': final_result,
    }

    json.dump(output, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
