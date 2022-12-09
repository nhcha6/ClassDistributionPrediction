#!/usr/bin/env bash
# note: please use xonsh, instead of bash
scenario = $ARG1
prior = $ARG2
dir = $ARG3

python distance_prior_regression.py --labeled_dataset @(scenario) --unlabeled_dataset @(scenario) --plot 0 --dir @(dir) --prior @(prior)