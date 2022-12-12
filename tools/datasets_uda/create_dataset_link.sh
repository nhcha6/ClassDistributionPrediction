import os

# please change this to your own environment
prefix = '/mnt/hpccs01/home/n11223243/class_distribution_prediction/data/'

def create_folder(file_root):
  if not os.path.exists(file_root):
    os.makedirs(file_root)

cd ../..
create_folder('dataset')
cd dataset

# 1. C2B: Cityscapes as source, BDD100k as target, BDD100k as test
print('create C2B dataset symlink: ')
create_folder('C2B')
cd C2B
create_folder('labeled_data')
cd labeled_data
ln -s @(prefix)/city/VOC2007_citytrain/* .
cd ..
create_folder('unlabeled_data')
cd unlabeled_data
ln -s @(prefix)/BDD/VOC2007_bddtrain/* .
cd ..
create_folder('test_data')
cd test_data
ln -s @(prefix)/BDD/VOC2007_bddval/* .
cd ../..

# 2. C2F: Cityscapes as source, BDD100k as target, BDD100k as test
print('create C2F dataset symlink: ')
create_folder('C2F')
cd C2F
create_folder('labeled_data')
cd labeled_data
ln -s @(prefix)/city/VOC2007_citytrain/* .
cd ..
create_folder('unlabeled_data')
cd unlabeled_data
ln -s @(prefix)/cityscapes_foggy/VOC2007_citytrain/* .
cd ..
create_folder('test_data')
cd test_data
ln -s @(prefix)/cityscapes_foggy/VOC2007_cityval/* .
cd ../..

# 3. C2N: CityScapes as source, BDD100k Nightime as target, BDD100k Nightime as test
print('create C2N dataset symlink: ')
create_folder('C2N')
cd C2N
create_folder('labeled_data')
cd labeled_data
ln -s @(prefix)/city/VOC2007_citytrain/* .
cd ..
create_folder('unlabeled_data')
cd unlabeled_data
ln -s @(prefix)/BDD/VOC2007_bddtrain/* .
cd ..
create_folder('test_data')
cd test_data
ln -s @(prefix)/BDD/VOC2007_bddval/* .
cd ../..