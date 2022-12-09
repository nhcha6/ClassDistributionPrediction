import torch, torchvision
import mmdet
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import json
import os
import cv2

# os.chdir("/mnt/hpccs01/home/n11223243/ssod")
os.chdir("/home/nicolas/hpc-home/ssod/")

# Choose to use a config and initialize the detector
config='work_dirs/baseline_ssod_1_0_2/baseline_ssod_1_0_2.py'
# Setup a checkpoint file to load
# checkpoint='work_dirs/baseline_ssod_1_0_2/latest.pth'
checkpoint='configs/tests/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# load annotations
anno = 'dataset/coco/annotations/semi_supervised/instances_train2017.1@0.json'
# images
images = 'data/coco/images/train2017/'
# save path
save_path = ''

# initialize the detector
model = init_detector(config, checkpoint)

# iterating through each image in the annotation file
f = open(anno, 'rb')
anno_dict = json.load(f)
for img_data in anno_dict['images']:
    img_name = img_data['file_name']
    img_path = images + img_name

    # img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # cv2.imshow('color image',img_color) 
    # cv2.waitKey(0) 

    result = inference_detector(model, img_path)
    show_result_pyplot(model, img_path, result, score_thr=0.9, wait_time=0)
