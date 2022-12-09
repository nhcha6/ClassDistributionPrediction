import os
import pickle
import json
import xml.etree.ElementTree as ET


class Clusters():
    def __init__(self, dataset, root='.'):
        # labeled and unlabeled datasets for C2B
        self.dataset = dataset

        data_root = f'{root}/dataset/{dataset}'
        self.labeled_fp = os.path.join(data_root, 'labeled_data/JPEGImages/')
        self.unlabeled_fp = os.path.join(data_root, 'unlabeled_data/JPEGImages/')
        self.labeled_anno_fp = os.path.join(data_root, 'labeled_data.json')
        self.unlabeled_anno_fp = os.path.join(data_root, 'unlabeled_data.json')
        self.unabeled_anno_orig_fp = os.path.join(data_root, 'unlabeled_data/Annotations/')

        # if method=='road_type':
        #     self.method = method
        #     self.road_type()
        # elif method=='manual_prior':
        #     self.method = method
        #     self.manual_prior()
        # else:
        #     self.method = 'dummy'
        #     self.dummy_method()
    
    def manual_prior(self, predicted_class_distribution, boxes_per_image):
        print(predicted_class_distribution)
        print(boxes_per_image)

        # import unlabeled json file
        f = open(self.unlabeled_anno_fp)
        unlabeled_anno = json.load(f)

        # build dictionary of unlabeled images and their road_type
        image_names = [x['file_name'].split('/')[-1] for x in unlabeled_anno['images']]

        # assign to dictionary
        self.cluster_imgs = dict(
            cluster0 = image_names, 
        )

        # MANUALLY DEFINE THE CLASS DISTRIBUTION
        
        # prediction from dual regression method
        # predicted_class_distribution = [0.0594598602428718, 0.6265260473232964, 0.013047970694580203, 0.20557211065699907, 0.01070782118060481, 0.016982466658595773, 0.015051533008159382, 0.05265219023489254]
        # boxes_per_image = 15

        # labeled data
        # predicted_class_distribution = [0.009319788827688753, 0.5175436924660275, 0.034439383254874306, 0.3429453582115156, 0.0032590672587623376, 0.014084507042253537, 0.07107053688844842, 0.007337666050429761]

        # we set boxes per image to a conservative value of 10
        self.cluster_priors = dict(
            cluster0 = dict(boxes_per_image = boxes_per_image, cls_ratio = predicted_class_distribution)
        )

        if not os.path.isdir(f'./{self.dataset}'):
            os.mkdir(f'./{self.dataset}')

        with open(f'./{self.dataset}/manual_prior_cluster_prior.pkl', 'wb') as handle:
            pickle.dump(self.cluster_priors, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(f'./{self.dataset}/manual_prior_cluster_imgs.pkl', 'wb') as handle:
            pickle.dump(self.cluster_imgs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def dummy_method(self):
        # import unlabeled json file
        f = open(self.unlabeled_anno_fp)
        unlabeled_anno = json.load(f)

        # build dictionary of unlabeled images and their road_type
        image_names = [x['file_name'].split('/')[-1] for x in unlabeled_anno['images']]

        # random split in half
        imgs0 = image_names[0:int(len(image_names)/2)]
        imgs1 = image_names[int(len(image_names)/2):]

        # assign to dictionary
        self.cluster_imgs = dict(
            cluster0 = imgs0, 
            cluster1 = imgs1
        )

        # random priors
        self.cluster_priors = dict(
            cluster0 = dict(boxes_per_image = 13.915817093503156, cls_ratio = [0.0406, 0.7900, 0.0066, 0.1314, 0.0002, 0.0042, 0.0102, 0.0168]),
            cluster1 = dict(boxes_per_image = 13.915817093503156, cls_ratio = [0.0406, 0.7900, 0.0066, 0.1314, 0.0002, 0.0042, 0.0102, 0.0168])
        )

        if not os.path.isdir(f'./cluster_priors/{self.dataset}'):
            os.mkdir(f'./cluster_priors/{self.dataset}')

        with open(f'./cluster_priors/{self.dataset}/{self.method}_cluster_prior.pkl', 'wb') as handle:
            pickle.dump(self.cluster_priors, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(f'./cluster_priors/{self.dataset}/{self.method}_cluster_imgs.pkl', 'wb') as handle:
            pickle.dump(self.cluster_imgs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def road_type(self):
        # import unlabeled json file
        f = open(self.unlabeled_anno_fp)
        unlabeled_anno = json.load(f)

        # build dictionary of unlabeled images and their road_type
        img_names = [x['file_name'].split('/')[-1] for x in unlabeled_anno['images']]
        road_type_img_dict = {}
        count = 0
        for img in img_names:
            xml_file = img.replace('jpg', 'xml')
            count+=1
            tree = ET.parse(os.path.join(self.unabeled_anno_orig_fp, xml_file))
            root = tree.getroot()
            scene = root.findall('scene')[0].text
            img_name = root.findall('filename')[0].text

            # insufficient numbers for a meaningful prior, so combine to 'other' category
            if scene in ['undefined', 'tunnel', 'parking lot', 'gas stations']:
                scene = 'other'

            if scene not in road_type_img_dict.keys():
                road_type_img_dict[scene] = [img_name]
            else:
                road_type_img_dict[scene].append(img_name)
            
            # if count == 1000:
            #     break
        
        # for road_type in road_type_img_dict.keys():
        #     print(len(road_type_img_dict[road_type]))
        
        # calculate the class distribution prior for each road type
        road_type_prior_dict = {}
        for road_type in road_type_img_dict.keys():
            road_type_prior_dict[road_type] = [0 for cls in unlabeled_anno['categories']]
        
        for detection in unlabeled_anno['annotations']:
            img_id = detection['image_id']
            class_id = detection['category_id']
            img_name = unlabeled_anno['images'][img_id]['file_name'].split('/')[-1]

            for road_type, imgs in road_type_img_dict.items():
                if img_name in imgs:
                    road_type_prior_dict[road_type][class_id] +=1
                    break

        self.cluster_imgs = road_type_img_dict
        
        self.cluster_priors = {}
        for road_type, prior in road_type_prior_dict.items():
            try:
                prior_dict = dict(boxes_per_image=sum(prior)/len(self.cluster_imgs[road_type]), cls_ratio=[x/sum(prior) for x in prior])
            except ZeroDivisionError:
                prior_dict = dict(boxes_per_image=sum(prior)/len(self.cluster_imgs[road_type]), cls_ratio=[0 for x in prior])

            self.cluster_priors[road_type] = prior_dict

        # print(self.cluster_imgs)
        print(self.cluster_priors)

        if not os.path.isdir(f'./cluster_priors/{self.dataset}'):
            os.mkdir(f'./cluster_priors/{self.dataset}')

        print(f'./cluster_priors/{self.dataset}/{self.method}_cluster_prior.pkl')
        with open(f'./cluster_priors/{self.dataset}/{self.method}_cluster_prior.pkl', 'wb') as handle:
            pickle.dump(self.cluster_priors, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(f'./cluster_priors/{self.dataset}/{self.method}_cluster_imgs.pkl', 'wb') as handle:
            pickle.dump(self.cluster_imgs, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    # calculate clusters
    clusters = Clusters('C2B')
    clusters.manual_prior([0.0594598602428718, 0.6265260473232964, 0.013047970694580203, 0.20557211065699907, 0.01070782118060481, 0.016982466658595773, 0.015051533008159382, 0.05265219023489254], 17)
    