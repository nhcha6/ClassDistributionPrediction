#!/usr/bin/env bash
# note: please use xonsh, instead of bash

import os
cd ../..

cur_path = os.path.abspath(os.path.dirname(__file__))
$PYTHONPATH=cur_path

# create C2N dataset
# dataset = 'C2N'
# name_list = ['labeled_data', 'unlabeled_data', 'test_data']
# subset_list = ['None', 'night', 'night']
# for i in range(len(name_list)):
#     name = name_list[i]
#     subset = subset_list[i]
#     print(f'===============process {dataset}==============')
#     data_root = f'./dataset/{dataset}'
#     out_dir = f'./dataset/{dataset}/{name}.json'
#     data_dir = os.path.join(data_root, name)
#     python tools/datasets_uda/convert_xml_to_json.py --devkit_path @(data_dir) --out-name @(out_dir) --dataset city --subset @(subset)


# create C2B dataset
dataset = 'C2B'
name_list = ['labeled_data', 'unlabeled_data', 'test_data']
subset_list = ['None', 'daytime', 'daytime']
for i in range(len(name_list)):
    name = name_list[i]
    subset = subset_list[i]
    print(f'===============process {dataset}==============')
    data_root = f'./dataset/{dataset}'
    out_dir = f'./dataset/{dataset}/{name}.json'
    data_dir = os.path.join(data_root, name)
    python tools/datasets_uda/convert_xml_to_json.py --devkit_path @(data_dir) --out-name @(out_dir) --dataset city --subset @(subset)

# create C2F dataset
# for dataset in ['C2F']:
#   print(f'===============process {dataset}==============')
#   data_root = f'./dataset/{dataset}'
#   for name in ['labeled_data', 'test_data', 'unlabeled_data']:
#     out_dir = f'./dataset/{dataset}/{name}.json'
#     data_dir = os.path.join(data_root, name)
#     print('processing dataset')
#     python tools/datasets_uda/convert_xml_to_json.py --devkit_path @(data_dir) --out-name @(out_dir) --dataset city