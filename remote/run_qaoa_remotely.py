import pickle

import numpy as np
from qiskit.algorithms.optimizers import COBYLA
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_optimization.applications import Maxcut
from qiskit import IBMQ

account = ''
runtime_program = ''

IBMQ.save_account(account, hub='', group='', project='', overwrite=True)
provider = IBMQ.load_account()

QiskitRuntimeService.save_account(account, instance='', channel='', overwrite=True)
service = QiskitRuntimeService()

G = pickle.load(open('', 'rb'))

max_cut = Maxcut(G)
qp = max_cut.to_quadratic_program()

qubit_op, offset = qp.to_ising()

optimizer = COBYLA()

options = {"backend_name": "ibmq_qasm_simulator"}
runtime_inputs = {
    "operator": qubit_op,
    "offset": offset,
    "optimizer": optimizer,
    "initial_point": np.array([0.0, 0.0, 0.0, 0.0]),
    "reps": 2,
    "recursive": False,
    "warm_start": False,
    "optimization_level": 3,
    "min_num_vars": 3,
}
job = provider.runtime.run(program_id=runtime_program, options=options, inputs=runtime_inputs)

with open(f'jobs_details/jobs_to_store.txt', 'w') as f:
    f.write(f'{job.job_id()}\n')
