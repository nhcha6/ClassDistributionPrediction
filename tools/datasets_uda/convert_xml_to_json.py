"""
Convert VOC-format dataset to COCO format
"""
import argparse
import os
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np

city = ['truck', 'car', 'rider', 'person', 'train', 'motorcycle', 'bicycle', 'bus']
car = ['car']
oak = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'monitor']
oak2 = ['bicycle', 'bus', 'car', 'chair', 'dining table', 'person', 'potted plant']
clad = ['Pedestrian', 'Cyclist', 'Car', 'Truck', 'Tram', 'Tricycle']


dataset_dict = {
    'city': city,
    'car': car,
    'oak': oak,
    'clad': clad,
    'voc': oak,
    'oak2': oak2
}

label_ids = None


def parse_xml(args):
    xml_path, img_path = args
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    bboxes = []
    labels = []
    bboxes_ignore = []
    labels_ignore = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        # change tvmonitor to monitor in voc dataset
        if name == 'tvmonitor':
            name = 'monitor'
        if name == 'pottedplant':
            name = 'potted plant'
        if name == 'diningtable':
            name = 'dining table'
        if name not in label_ids.keys():
            continue    
        
        label = label_ids[name]
    
        try:
            difficult = int(obj.find('difficult').text)
        except:
            difficult = False
        bnd_box = obj.find('bndbox')
        bbox = [
            round(float(bnd_box.find('xmin').text)),
            round(float(bnd_box.find('ymin').text)),
            round(float(bnd_box.find('xmax').text)),
            round(float(bnd_box.find('ymax').text))
        ]
        if difficult:
            bboxes_ignore.append(bbox)
            labels_ignore.append(label)
        else:
            bboxes.append(bbox)
            labels.append(label)
    if not bboxes:
        bboxes = np.zeros((0, 4))
        labels = np.zeros((0,))
    else:
        bboxes = np.array(bboxes, ndmin=2) - 1
        labels = np.array(labels)
    if not bboxes_ignore:
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0,))
    else:
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
        labels_ignore = np.array(labels_ignore)
    annotation = {
        'filename': img_path,
        'width': w,
        'height': h,
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
            'bboxes_ignore': bboxes_ignore.astype(np.float32),
            'labels_ignore': labels_ignore.astype(np.int64)
        }
    }
    return annotation


def cvt_annotations(devkit_path, out_file, classes, subset, dataset):
    annotations = []
    xml_root = os.path.join(devkit_path, 'Annotations')
    img_names = [a[:-4] for a in os.listdir(xml_root) if a.endswith('.xml')]

    # only daytime annotations for BDD
    if subset == 'daytime':
        count = 0
        img_names_new = []
        for name in img_names:
            path = osp.join(devkit_path, f'Annotations/{name}.xml')
            count+=1
            tree = ET.parse(path)
            root = tree.getroot()
            tod = root.find('timeofday').text
            if tod == 'daytime':
                print(count, tod)
                img_names_new.append(name)
        img_names = img_names_new
    
    # only night annotations for BDD
    if subset == 'night':
        count = 0
        img_names_new = []
        for name in img_names:
            path = osp.join(devkit_path, f'Annotations/{name}.xml')
            count+=1
            tree = ET.parse(path)
            root = tree.getroot()
            tod = root.find('timeofday').text
            if tod != 'daytime':
                print(count, tod)
                img_names_new.append(name)
        img_names = img_names_new

    # split generate different splits of the CLAD dataset
    if subset:
        if dataset == 'clad':
            count = 0
            img_names_new = []
            for name in img_names:
                path = osp.join(devkit_path, f'Annotations/{name}.xml')
                count+=1
                tree = ET.parse(path)
                root = tree.getroot()
                domain = root.find('domain').text
                # if domain == '2' or domain == '3' or domain == '5':
                # if domain == '2':
                if domain in subset:
                    print(count, domain)
                    img_names_new.append(name)
            img_names = img_names_new
    
    # split generate different splits of the CLAD dataset
    if subset:
        if dataset[0:3] == 'oak':
            lower = int(subset.split('-')[0])
            upper = int(subset.split('-')[1])
            img_names = sorted(img_names)[lower:upper]
            print(img_names)
            
            count = 0
        
    # extract only trainval data from VOC
    if dataset == 'voc':
        with open(f'{devkit_path}/ImageSets/Main/train_trainval.txt') as f:
            lines = f.readlines()
            img_names = [name.split(' ')[0] for name in lines]
            print(img_names)

    # only include images with 0.02 level fog
    if subset == '02':
        count = 0
        img_names_new = []
        for name in img_names:
            print(count)
            if name.split('.')[1] == subset:
                print(name)
                img_names_new.append(name)
            count+=1
        img_names = img_names_new

    img_paths = [
        f'JPEGImages/{img_name}.jpg' for img_name in img_names
    ]
    xml_paths = [osp.join(devkit_path, f'Annotations/{img_name}.xml') for img_name in img_names]
    global label_ids
    label_ids = {name: i for i, name in enumerate(classes)}
    
    part_annotations = mmcv.track_progress(parse_xml, list(zip(xml_paths, img_paths)))
    annotations.extend(part_annotations)
    if out_file.endswith('json'):
        annotations = cvt_to_coco_json(annotations, classes)
    mmcv.dump(annotations, out_file)
    return annotations

def cvt_to_coco_json(annotations, classes):
    image_id = 0
    annotation_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    def addAnnItem(annotation_id, image_id, category_id, bbox, difficult_flag):
        annotation_item = dict()
        annotation_item['segmentation'] = []

        seg = []
        # bbox[] is x1,y1,x2,y2
        # left_top
        seg.append(int(bbox[0]))
        seg.append(int(bbox[1]))
        # left_bottom
        seg.append(int(bbox[0]))
        seg.append(int(bbox[3]))
        # right_bottom
        seg.append(int(bbox[2]))
        seg.append(int(bbox[3]))
        # right_top
        seg.append(int(bbox[2]))
        seg.append(int(bbox[1]))

        annotation_item['segmentation'].append(seg)

        xywh = np.array(
            [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        annotation_item['area'] = int(xywh[2] * xywh[3])
        if difficult_flag == 1:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 1
        else:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 0
        annotation_item['image_id'] = int(image_id)
        annotation_item['bbox'] = xywh.astype(int).tolist()
        annotation_item['category_id'] = int(category_id)
        annotation_item['id'] = int(annotation_id)
        coco['annotations'].append(annotation_item)
        return annotation_id + 1

    for category_id, name in enumerate(classes):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id)
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    for ann_dict in annotations:
        file_name = ann_dict['filename']
        ann = ann_dict['ann']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(ann_dict['height'])
        image_item['width'] = int(ann_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)

        bboxes = ann['bboxes'][:, :4]
        labels = ann['labels']
        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            label = labels[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=0)

        bboxes_ignore = ann['bboxes_ignore'][:, :4]
        labels_ignore = ann['labels_ignore']
        for bbox_id in range(len(bboxes_ignore)):
            bbox = bboxes_ignore[bbox_id]
            label = labels_ignore[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=1)
        image_id += 1
    return coco


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert XML annotations to mmdetection format')
    parser.add_argument('--devkit_path', default='', help='devkit_path')
    parser.add_argument('--out-name', default='', help='output file name')
    parser.add_argument('--dataset', default='city')
    parser.add_argument('--subset', default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    devkit_path = args.devkit_path
    out_name = args.out_name
    dataset = args.dataset
    subset = args.subset

    out_dir = osp.dirname(out_name)
    mmcv.mkdir_or_exist(out_dir)

    classes = dataset_dict[dataset]
    cvt_annotations(devkit_path, out_name, classes, subset,dataset)

    print('Done!')


if __name__ == '__main__':
    main()
