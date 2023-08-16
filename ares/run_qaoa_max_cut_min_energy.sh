#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --time=1:00:00
#SBATCH -A
#SBATCH -p
#SBATCH --array 8

module load python/3.10.4-gcccore-11.3.0

export IBM_QUANTUM_TOKEN=

cd $SLURM_SUBMIT_DIR

source ../qiskit/bin/activate

python run_qaoa_max_cut_min_energy.py -num_experiments 1 -graph graph_12_5.pickle -initial_point 0.0 0.0 -reps $SLURM_ARRAY_TASK_ID -optimizer L_BFGS_B -step 0.05 -num_initial_solutions 3 -num_cuts 3