#!/bin/bash
#SBATCH --job-name=AML25            # Job name    (default: sbatch)
#SBATCH --output=aml.out            # Output file (default: slurm-%j.out)
#SBATCH --error=aml.err             # Error file  (default: slurm-%j.out)
#SBATCH --nodes=6                   # Number of nodes
#SBATCH --ntasks=48                  # Number of tasks
#SBATCH --ntasks-per-node=8         # Number of tasks per node
#SBATCH --constraint=EPYC_7763      # Select node with CPU
#SBATCH --mem-per-cpu=4096          # Memory per CPU
#SBATCH --time=24:00:00             # Wall clock time limit
#SBATCH --mail-type=END,FAIL        # Send an email when job ends

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