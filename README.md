# QAOA (and its versions) adaptation for IBMQ

This repository presents an adaptation for [Quantum Approximate Optimization Algorithm (QAOA)](https://arxiv.org/pdf/1411.4028.pdf) and three of its versions,
[Warm-Started](https://arxiv.org/pdf/2009.10095.pdf), [Recursive](https://arxiv.org/pdf/1910.08980.pdf) and [Warm-Started Recursive](https://arxiv.org/pdf/2009.10095.pdf), for IBMQ.

Implementation has been prepared in three versions:
- local, for which the quantum part of the code is run on simulator locally
- hybrid, for which thw quantum part is run remotely, on IBMQ, but the classical one is run locally
- remote, for which both parts of the code are run on IBMQ

## Limitations for various versions

### Local

The local version can be run on personal computer and supercomputer (e.g. Ares). All versions of the algorithm can be
run this way (including different version of classical solvers, like Goemans-Williamson), however the computation time will depend on the hardware. Running large problems or the recursive 
version of the algorithm on personal computer may be impossible due to large computation time.

### Hybrid

The hybrid version can be run on personal computer and supercomputer (e.g. Ares). All versions of the algorithm can be
run this way, however the computation time will depend on availability of quantum processors. The larger the queues to the
quantum processors, the more time the program will need to maintain an active session with remote platform.
Any restrictions on job runtime on supercomputers should be considered (e.g. at the time of writing this, the maximum
time the job could run on regular Ares partitions was 72h, which was not enough time to wait in queue to the largest available 
quantum processor at that time).

### Remote

The remote version is run on remote platform, which (at the time of writing this) means that the computational environment
for the classical part of the algorithm is not modifiable by user. In particular, it means that user is limited to 
certain modules and packages and cannot upload their own modifications of qiskit libraries easily. In the context of QAOA
it's worth to mention that Goemans-Williamson optimizer from qiskit library cannot be used out of the box.

## Repository content

### ares

The package contains some example bash scripts to run the code on supercomputers with SLURM (e.g. Ares). You can use them
as starting point for your own scripts and modify them accordingly.

### hybrid

The package contains a code for running a hybrid version of the algorithm. The code from qiskit library was modified to 
make it less prone to single job failures (some retries were added). The code was prepared to run Maximum Cut problem - for
other problems code modification will be required.

### local

The package contains a code for running a local version of the algorithm. 

The code was prepared to run:
- Maximum Cut problem:
  - `run_qaoa_initial_point.py` - to test algorithm against different initial points
  - `run_qaoa_max_cut_min_energy.py` - to see energies' levels in consecutive iterations of the algorithm
  - `run_qaoa_max_cut.py` - the general configuration
- Workflow problem:
  - `run_qaoa_one_hot.py` - one-hot version of the algorithm
  - `run_qaoa_binary.py` - binary version of the algorithm
  - `run_qaoa_domain_wall.py` - domain wall version of the algorithm

### remote

The package contains a code for running a remote version of the algorithm. To run a remote version you need to uplad a
program first:

````
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(channel="", token='', overwrite=True)
service = QiskitRuntimeService()
data = os.path.join(
    os.getcwd(), "remote/qaoa.py"
)
meta = os.path.join(
    os.getcwd(), "remote/qaoa_metadata.json"
)

program_id = service.upload_program(name = "qaoa", data = data, metadata = meta)
````

Then you can run it as any other program, e.g. (broader example in `MaxCut.ipynb`):
````
from qiskit import IBMQ

IBMQ.save_account('', overwrite = True)
provider = IBMQ.load_account()

options = {"backend_name": "ibmq_qasm_simulator"}
runtime_inputs = {
    "operator": qubit_op,
    "offset": offset,
    "optimizer": optimizer,
}
job = provider.runtime.run(program_id, options=options, inputs=runtime_inputs)
````
