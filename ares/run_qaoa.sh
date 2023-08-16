#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --time 6:00:00
#SBATCH -A
#SBATCH -p
#SBATCH --array 1

module load python/3.10.4-gcccore-11.3.0

export IBM_QUANTUM_TOKEN=

cd $SLURM_SUBMIT_DIR

source ../qiskit/bin/activate

python run_qaoa_domain_wall.py --custom_mixer --recursive -initial_point 0.0 0.0 -reps $SLURM_ARRAY_TASK_ID -num_experiments 100 -num_initial_solutions 3