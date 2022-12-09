#!/bin/bash -l
#PBS -N C2B_novel
#PBS -l ncpus=10
#PBS -l ngpus=2
#PBS -l mem=32GB
#PBS -l walltime=72:00:00
#PBS -l gputype=A100
#PBS -j oe

DIR = '~/ssod/'
 
# activate conda environment
conda activate ssod

# Explicitly set the starting directory
cd ~/ssod/cluster_priors

# run domain prediction
xonsh predict_for_uda.sh C2B predicted dir

# Explicitly set the starting directory
cd ~/ssod/examples/train/xonsh

# Run the python script
xonsh train_gpu2.sh ./configs/labelmatch/labelmatch_uda_cluster.py C2B
