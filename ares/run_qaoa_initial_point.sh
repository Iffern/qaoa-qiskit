#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --time=14:00:00
#SBATCH -A
#SBATCH -p
#SBATCH --array

module load python/3.10.4-gcccore-11.3.0

export IBM_QUANTUM_TOKEN=

cd $SLURM_SUBMIT_DIR

source ../qiskit/bin/activate

python run_qaoa_initial_point.py -beta_step 0.1 --warm_start --recursive -initial_gamma_position $SLURM_ARRAY_TASK_ID -graph graph_12_5.pickle -optimizer COBYLA -num_initial_solutions 3 -num_cuts 3