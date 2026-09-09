#!/bin/bash
#SBATCH --job-name=AML25            # Job name    (default: sbatch)
#SBATCH --output=aml.out            # Output file (default: slurm-%j.out)
#SBATCH --error=aml.err             # Error file  (default: slurm-%j.out)
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=64          # Number of tasks per task
#SBATCH --constraint=EPYC_7763      # Select node with CPU
#SBATCH --mem-per-cpu=1G            # Memory per CPU
#SBATCH --time=36:00:00             # Wall clock time limit
#SBATCH --mail-type=END,FAIL        # Send an email when job ends

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load some modules
module load stack/2025-06 gcc/12.2.0 python/3.13.0 eth_proxy
module list

# Load environment
source installation.sh

# Run the program
# python3 main.py -val=0 --save_params=1 --load_params=0
python -m pip install ipykernel

# Register the venv as a Jupyter kernel
python -m ipykernel install --user \
    --name=.venv \
    --display-name "Python (.venv)"

# Verify registration
jupyter kernelspec list

# Execute notebook
jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.kernel_name=.venv \
    datacrunch.ipynb