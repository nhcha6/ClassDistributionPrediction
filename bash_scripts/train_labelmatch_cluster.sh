# Explicitly set the starting directory
cd ../cluster_priors

# run domain prediction
xonsh predict_for_uda.sh C2B predicted '/home/nicolas/hpc-home/class_distribution_prediction/'

# Explicitly set the starting directory
cd ../examples/train/xonsh

# Run the python script
xonsh train_gpu2.sh ./configs/labelmatch/labelmatch_uda_cluster.py C2B
