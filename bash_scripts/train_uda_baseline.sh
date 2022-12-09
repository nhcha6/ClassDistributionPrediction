#!/bin/bash -l
#PBS -N C2N_target_baseline
#PBS -l ncpus=10
#PBS -l ngpus=1
#PBS -l mem=32GB
#PBS -l walltime=23:00:00
#PBS -l gputype=A100
#PBS -j oe
 
# activate conda environment
conda activate ssod

# Explicitly set the starting directory
cd ~/ssod/examples/train/xonsh

# Run the python script
xonsh train_gpu1.sh ./configs/baseline/baseline_uda.py C2N