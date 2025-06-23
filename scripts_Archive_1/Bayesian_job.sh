#!/bin/bash
#
#SBATCH -p igbmc                     # Partition
#SBATCH -N 1                         # Number of nodes
#SBATCH -n 1                         # Number of cores
#SBATCH --mem=64GB                   # Memory allocation (adjust as needed)
#SBATCH -t 0-2:00                    # Maximum runtime (D-HH:MM)

# Load Conda (if necessary, depending on your cluster)
#source ~/.bashrc  # Only needed if Conda is not available directly

# Activate your Conda environment
conda init
conda activate 312_main_env

# Run the Python script
python E_fitting_scanpy_v1.1_Bayesian.py >> output_Bayesian.txt
