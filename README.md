# Class Distribution Prediction for Reliable Domain Adaptive Object Detection

The following repository is built upon the [MMDetection-based Toolbox for Semi-Supervised Object Detection](https://github.com/hikvision-research/SSOD). For more details, see this repostitory or the associated [paper](https://arxiv.org/abs/2206.06608).

## Virtual Environment

Create a virtual environment using conda and the requirements.txt file. We use Linux with Python 3.7.
```bash
conda create --name myenv --file requirements.txt
```
## Dataset Preparation

1. Download [cityscapes](https://cityscapes-dataset.com), [cityscapes-foggy](https://cityscapes-dataset.com) and [BDD100K](https://bdd-data.berkeley.edu) from the website and organize them as follows:

   ```shell
   # cityscapes          |    # cityscapes_foggy      |   # BDD
   /data/city            |    /data/cityscapes_foggy  |   /data/BDD
     - VOC2007_citytrain |      - VOC2007_foggytrain  |     - VOC2007_bddtrain
       - ImageSets       |        - ImageSets         |       - ImageSets
       - JPEGImages      |        - JPEGImages        |       - JPEGImages
       - Annotations     |        - Annotations       |       - Annotations 
     - VOC2007_cityval   |      - VOC2007_foggyval    |     - VOC2007_bddval 
       - ImageSets       |        - ImageSets         |       - ImageSets
       - JPEGImages      |        - JPEGImages        |       - JPEGImages
       - Annotations     |        - Annotations       |       - Annotations 
   ```
The datasets should follow the VOC format, with image annotations as xml files in the 'Annotations' folder and images with the same names in 'JPEGImages'.

2. Once the datasets are in the correct format, we organise them into the three adaptation scenarios by creating dataset links. 

   ```shell
   cd tools/datasets_uda
   xonsh create_dataset_link.sh
   ```

3. We then convert the xml files to Coco format by running:

   ```bash
   cd tools/datasets_uda
   xonsh preprocess_dataset.sh
   ```
   Additionally the script 'convert_xml_to_json.py' can be edited to use only a subset of a dataset when creating the json annotation file. We utilize this to split BDD100k into the daytime and night subsets.
   
4. You can run dataset/browse_dataset.py to visualize the annotations in the json file. Firstly, edit example_config.py so that the desired dataset is referenced. Search 'TODO' to find the lines that need updating.

5. Generate the CLIP image embeddings for performing class distribution prediction. This is done before self-training is run to speed up class distribution prediction. Edit 'cluster_priors/save_clip_embeddings.py' to call the desired scenario and dataset, and run the python script. The embeddings will be saved in the folder 'cluster_priors/clip_embeddings/'.

## Training

#### 1. Use labeled data to train a baseline

Before training，please download the pretrained backbone ([resnet50](https://download.pytorch.org/models/resnet50-19c8e357.pth)) to `pretrained_model/backbone`.

```shell
# |---------------------|--------|------|---------|---------|
# | xonsh train_gpu2.sh | config | seed | percent | dataset |
# |---------------------|--------|------|---------|---------|
cd examples/train/xonsh
## ---dataset: coco-standard---
xonsh train_gpu2.sh ./configs/baseline/baseline_ssod.py 1 1 coco-standard
## ---dataset: voc---
# xonsh train_gpu2.sh ./configs/baseline/baseline_ssod.py 1 1 voc
## ---dataset: coco-additional---
# xonsh train_gpu8.sh ./configs/baseline/baseline_ssod.py 1 1 coco-additional
```

- In our implementation, we use 2-gpus to train except coco-additional.

- After training, we organize the pretrained baseline to `pretrained_model/baseline` as follows：

  ```shell
  pretrained_model/
  	└── baseline/
          ├── instances_train2017.1@1.pth
          ├── instances_train2017.1@5.pth
          ├── ...
          ├── voc.pth
          └── coco.pth
  ```

  - You can also change the `load_from` information in `config` file in step 2.

#### 2. Use labeled data + unlabeled data to train detector

```shell
## note: dataset is set to none in this step.
cd examples/train/xonsh
xonsh train_gpu8.sh ./configs/labelmatch/labelmatch_standard.py 1 1 none
```

- In our implementation, we use 8-gpus to train.
- You can also run `bash train_ssod.sh` in `examples/train/bash`

### Evaluation

```shell
# please change "config" and "checkpoint" in 'eval.sh' scripts to support different dataset and trained model
cd examples/eval
xonsh eval.sh
```


