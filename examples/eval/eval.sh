#!/usr/bin/env bash
# note: please use xonsh, instead of bash

import os
cd ../..
$LANG='zh_CN.UTF-8'
$LANGUAGE='zh_CN:zh:en_US:en'
$LC_ALL='C.UTF-8'

cur_path = os.path.abspath(os.path.dirname(__file__))
$PYTHONPATH=cur_path

GPU = 1
# config='./work_dirs/baseline_uda_C2B/baseline_uda_C2B.py'   # for coco-standard and coco-additional
# checkpoint=f'./work_dirs/baseline_uda_C2B/iter_48000.pth'

config = './saved_results/OAK1025/labelmatch_uda_prior/labelmatch_uda_OAK1025.py'
checkpoint = './saved_results/OAK1025/labelmatch_uda_prior/iter_14000_ema.pth'

# config='./configs/baseline/ema_config/baseline_voc.py'   # for coco-standard and coco-additional
# checkpoint=f'./pretrained_model/baseline/voc.pth'

eval_type='bbox'

if GPU>1:
  python -m torch.distributed.launch --nproc_per_node=@(GPU) --master_port=19005 \
  examples/eval/eval.py --config @(config) --checkpoint @(checkpoint) --launcher pytorch \
  --eval @(eval_type)
else:
  python examples/eval/eval.py --config @(config) --checkpoint @(checkpoint) --eval @(eval_type)
