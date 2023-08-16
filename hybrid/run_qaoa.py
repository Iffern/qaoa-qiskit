import argparse
import datetime
import json
import os
import pickle

import numpy as np
from QAOA import QAOA
from MinimumEigenOptimizer import MinimumEigenOptimizer
from WarmStartQAOAOptimizer import WarmStartQAOAOptimizer
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit_ibm_runtime import Session, QiskitRuntimeService, Options, Sampler
from qiskit_optimization.algorithms import GurobiOptimizer, GoemansWilliamsonOptimizer
from qiskit_optimization.applications import Maxcut
from RecursiveMinimumEigenOptimizer import RecursiveMinimumEigenOptimizer


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
    filename += f'{args.backend}_{str(args.reps)}_{args.pre_solver}_{args.problem}' \
                f'_{args.graph.replace(".pickle", "")}_{suffix}.json'
    return filename


parser = argparse.ArgumentParser(
    prog='QAOA runner',
    description='Runs selected QAOA variant on problem instance')

parser.add_argument('--warm_start', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--recursive', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('-backend', default='ibmq_qasm_simulator', type=str)
parser.add_argument("-reps", default=1, type=int)
parser.add_argument("-min_num_vars", default=3, type=int)
parser.add_argument('-initial_point', action=Store_as_array, type=float, nargs='+')
parser.add_argument("-pre_solver", default="GurobiOptimizer", type=str)
parser.add_argument("-problem", default="MaxCut", type=str)
parser.add_argument("-optimization_level", default=3, type=int)
parser.add_argument("-resilience_level", default=1, type=int)
parser.add_argument("-graph", type=str)
parser.add_argument("-optimizer", default='COBYLA', type=str)
parser.add_argument("-step", default=0.01, type=float)

args = parser.parse_args()

service = QiskitRuntimeService(token=os.getenv("IBM_QUANTUM_TOKEN"),
                               instance=f'{os.getenv("IBM_QUANTUM_HUB")}/{os.getenv("IBM_QUANTUM_GROUP")}/{os.getenv("IBM_QUANTUM_PROJECT")}',
                               channel='ibm_quantum')

if args.optimizer == 'COBYLA':
    optimizer = COBYLA()
elif args.optimizer == 'L_BFGS_B':
    optimizer = L_BFGS_B(eps=args.step)
if args.pre_solver == "GurobiOptimizer":
    pre_solver = GurobiOptimizer()
elif args.pre_solver == "GoemansWilliamsonOptimizer":
    pre_solver = GoemansWilliamsonOptimizer(num_cuts=3)

history = {"nfevs": [], "params": [], "energy": [], "metadata": []}

if args.problem == "MaxCut":
    G = pickle.load(open(args.graph, 'rb'))
    max_cut = Maxcut(G)
    qp = max_cut.to_quadratic_program()


def store_history_and_forward(nfevs, params, energy, meta):
    # store information
    history["nfevs"].append(nfevs)
    history["params"].append(params)
    history["energy"].append(energy)
    history["metadata"].append(meta)


options = Options()
options.optimization_level = args.optimization_level
options.resilience_level = args.resilience_level


def solve_and_store_result(qaoa):
    result = None
    try:
        result = qaoa.solve(qp)
    finally:
        with open(f'results/{get_filename(args)}', "w") as f:
            x = None if result is None else result.x
            fval = None if result is None else result.fval

            parsed_samples = []

            for sample in result.samples:
                parsed_samples.append(
                    {"fval": float(sample.fval), "x": sample.x.tolist(), "probability": float(sample.probability)})

            serialized_inputs = {
                "initial_point": args.initial_point if args.initial_point is None else list(args.initial_point),
                "optimizer": {
                    "__class__.__name__": optimizer.__class__.__name__,
                    "__class__": str(optimizer.__class__),
                    "settings": getattr(optimizer, "settings", {}),
                },
                "pre_solver": args.pre_solver,
                "reps": args.reps,
                "warm_start": args.warm_start,
                "recursive": args.recursive,
                "backend": args.backend,
                "problem": args.problem,
                "optimization_level": args.optimization_level,
                "resilience_level": args.resilience_level
            }
            if args.graph is not None:
                serialized_inputs["graph"] = args.graph
            output = {
                "x": x,
                "fval": fval,
                "optimizer_history": history,
                "inputs": serialized_inputs,
                "raw_result": result.raw_results,
                "samples": parsed_samples,
            }
            json.dump(output, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)


if not args.recursive:
    with Session(service=service, backend=args.backend) as session:
        qaoa_mes = QAOA(Sampler(session=session, options=options), optimizer=optimizer, reps=args.reps,
                        initial_point=args.initial_point, callback=store_history_and_forward)
        if args.warm_start:
            qaoa = WarmStartQAOAOptimizer(qaoa=qaoa_mes, pre_solver=pre_solver,
                                          relax_for_pre_solver=False)
        else:
            qaoa = MinimumEigenOptimizer(qaoa_mes)
        solve_and_store_result(qaoa)
else:
    qaoa_mes = QAOA(None, optimizer=optimizer, reps=args.reps,
                    initial_point=args.initial_point, callback=store_history_and_forward)
    if args.warm_start:
        qaoa = RecursiveMinimumEigenOptimizer(WarmStartQAOAOptimizer(qaoa=qaoa_mes,
                                                                     pre_solver=pre_solver,
                                                                     relax_for_pre_solver=False),
                                              min_num_vars=args.min_num_vars,
                                              service=service, backend=args.backend, options=options)
    else:
        qaoa = RecursiveMinimumEigenOptimizer(MinimumEigenOptimizer(qaoa_mes), min_num_vars=args.min_num_vars,
                                              service=service, backend=args.backend, options=options)
    solve_and_store_result(qaoa)
