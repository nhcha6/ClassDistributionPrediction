#!/bin/bash -l
#PBS -N C2B_labeled_prior
#PBS -l ncpus=10
#PBS -l ngpus=2
#PBS -l mem=32GB
#PBS -l walltime=72:00:00
#PBS -l gputype=A100
#PBS -j oe
 
# activate conda environment
conda activate ssod

# Explicitly set the starting directory
cd ~/ssod/examples/train/xonsh

# Run the python script
xonsh train_gpu2.sh ./configs/labelmatch/labelmatch_uda.py C2B