from lib2to3.pytree import LeafPattern
from logging.config import stopListening
from pydoc import cli
from re import L
from tkinter.font import names
import umap
from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
import plotly.express as px
import os
import pickle
import torch
from torch.utils.data import DataLoader
from multiprocessing import Process
from sklearn.cluster import KMeans
import json
import sys
import csv
import random
from scipy import stats

from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel, CLIPModel, CLIPTokenizer
from sklearn.linear_model import LinearRegression



class ClipDistance:
    def __init__(self, dataset, split, image_subset=False, plot=True, random_flag = False, data_name='default', path = '/home/nicolas/hpc-home/ssod/'):
        self.dataset = dataset
        self.split = split
        self.image_subset = image_subset
        self.plot = plot
        self.random_flag = random_flag
        self.data_name = data_name
        
        global DIR
        DIR = path

        self.parent_dataset = dataset
        for i in range(len(self.parent_dataset)-1, 0, -1):
            if self.parent_dataset[i].isdigit():
                self.parent_dataset = self.parent_dataset[0:i]
            else:
                break

        # get class names
        if self.parent_dataset in ['C2B', 'D2N', 'C2N', 'C2F']:
            self.classes = ['truck', 'car', 'rider', 'person', 'train', 'motorcycle', 'bicycle', 'bus']
        elif self.parent_dataset in ['OAK', 'V2O']:
            self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'monitor']
            if dataset =='OAK202':
                self.classes = ['bicycle', 'bus', 'car', 'chair', 'dining table', 'person', 'potted plant']
        elif self.parent_dataset in ["CLAD"]:
            self.classes = ['Pedestrian', 'Cyclist', 'Car', 'Truck', 'Tram', 'Tricycle']

    def calculate_clip_distances(self, normalized_text_output, labels):

        # clip_embedding, umap_embedding, names, dataset_list = clip_to_umap(dataset, ['labeled_data', 'unlabeled_data'])
        clip_embedding, umap_embedding, names, dataset_list = self.clip_to_umap(self.dataset, [self.split])

        # only include names in image_subset
        if self.image_subset:
            indeces = []
            for i in range(len(clip_embedding)):
                if names[i] in self.image_subset:
                    indeces.append(i)
            clip_embedding = np.array([clip_embedding[i] for i in indeces])
            umap_embedding = np.array([umap_embedding[i] for i in indeces])
            names = [names[i] for i in indeces]
            dataset_list = [dataset_list[i] for i in indeces]
        
        self.names = names
        self.clip_embeddings = clip_embedding

        ################### PERFORM CLASSIFICATION WITH CLIP ######################
        # import model
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # learnt scaling factor is multiplied by logits before applying softmax
        t = model.logit_scale.exp()

        # process image data, starting with the saved CLIP embedding
        pooled_vision_output = torch.FloatTensor(clip_embedding)
        # project into shared subspace
        projected_vision_output = model.visual_projection(pooled_vision_output)
        # normalise
        normalized_vision_output = projected_vision_output / projected_vision_output.norm(p=2, dim=-1, keepdim=True)

        # dot product of text embeddings and image embeddings
        manual_logits = torch.matmul(normalized_vision_output, normalized_text_output.t())*t
        # softmax to convert to a classification score
        manual_probs = torch.nn.functional.softmax(manual_logits)
        # convert to classifications
        cls_clip = torch.argmax(manual_probs, dim=1).detach().numpy()

        # store result
        self.clip_distance = manual_logits.detach().numpy()
        self.labels = labels

        # get the per-image class distribution
        self.calc_class_distribution()

        # produce dataframe
        self.distance_prior_df()

    def clip_to_umap(self, dataset, data_type, max_size = 100000):
        data = []
        image_names = []
        dataset_list = []

        print('import clip embeddings')
        for dt in data_type:
            # ignore processed images not in the annotation file
            f = open(f'{DIR}dataset/{dataset}/{dt}.json')
            anno = json.load(f)
            image_set = set()
            for image_details in anno['images']:
                image_set.add(image_details['file_name'].split('/')[-1])
            # sample set if larger than max size
            if len(image_set) > max_size:
                image_set = random.sample(image_set, max_size)

            clip_embeddings = {}
            with open(f'{DIR}cluster_priors/clip_embeddings/{self.parent_dataset}_{dt}_clip_saved.csv') as f:
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    print(row)
                    clip_embeddings[row[0]] = [float(i) for i in row[1:]]

            for name, embedding in clip_embeddings.items():
                # only assess data if it is in the annotations file
                # if name in image_set:
                if name in image_set:
                    image_names.append(name)
                    data.append(embedding)
                    dataset_list.append(dt)

        # these parameters have been tuned to better capture the global structure of the data, leading to a representation that 
        # best preserves clusters in the CLIP representation space
        # umap_embedding = umap.UMAP(n_neighbors=50,
        #                         min_dist=0.5,
        #                         metric='correlation').fit_transform(data)
        umap_embedding = data
        
        return data, umap_embedding, image_names, dataset_list
    
    def calc_class_distribution(self):
        f = open(f'{DIR}dataset/{self.dataset}/{self.split}.json')
        anno = json.load(f)
        
        # init data structure
        image_class_dist = {}
        for name in self.names:
            image_class_dist[name] = [0 for cls in anno['categories']]

        # overall distribution
        self.class_distribution = [0 for cls in anno['categories']]
        
        # sum detections for each class
        for detection in anno['annotations']:
            img_id = detection['image_id']
            class_id = detection['category_id']
            img_name = anno['images'][img_id]['file_name'].split('/')[-1]
            try:
                image_class_dist[img_name][class_id]+=1
                self.class_distribution[class_id]+=1
            except Exception as e:
                # print(e)
                continue
                
        self.image_class_distribution = []
        for name in self.names:
            try:
                # self.image_class_distribution.append([x/sum(image_class_dist[name]) for x in image_class_dist[name]])
                self.image_class_distribution.append([float(x) for x in image_class_dist[name]])
            except ZeroDivisionError:
                self.image_class_distribution.append([0.0 for x in image_class_dist[name]])
        
        # normalise overall dist
        self.class_distribution = [x/sum(self.class_distribution) for x in self.class_distribution]
    
    def cluster_class_distribution(self, clusters, names, dataset, split):
        f = open(f'{DIR}dataset/{dataset}/{split}.json')
        anno = json.load(f)
        
        # init data structure
        cluster_class_dist = {}
        for clust in range(max(clusters)+1):
            cluster_class_dist[clust] = [0 for cls in anno['categories']]
        
        # sum detections for each class
        for detection in anno['annotations']:
            img_id = detection['image_id']
            class_id = detection['category_id']
            img_name = anno['images'][img_id]['file_name'].split('/')[-1]
            try:
                i = names.index(img_name)
                cluster = clusters[i]
                cluster_class_dist[cluster][class_id]+=1
            except Exception as e:
                # print(e)
                continue
        
        # calculate prior
        cluster_prior = {}
        new_clusters = []
        priors = []
        for cluster, prior in cluster_class_dist.items():
            try:
                prior_dict = {'boxes_per_image': sum(prior)/len([x for x in clusters if x==cluster]), 'cls_ratio': [x/sum(prior) for x in prior]}
            except ZeroDivisionError:
                prior_dict = {'boxes_per_image': 0, 'cls_ratio': prior}
            cluster_prior[cluster] = prior_dict
            
            # for clustering in the class distribution space
            new_clusters.append(cluster)
            priors.append([cr for cr in prior_dict['cls_ratio']])
            
        return cluster_prior, new_clusters, priors
    
    def distance_prior_df(self):

        data = {}
        data['names'] = []
        data['dataset'] = []
        for i in range(len(self.names)):
            data['names'].append(self.names[i])
            data['dataset'] = self.data_name
        for c in range(len(self.classes)):
            data[self.classes[c]] = []
            for i in range(len(self.names)):
                data[self.classes[c]].append(self.image_class_distribution[i][c])
        for l in range(len(self.labels)):
            data[self.labels[l]] = []
            for i in range(len(self.names)):
                data[self.labels[l]].append(self.clip_distance[i][l])

        self.data_df = pd.DataFrame(data)
    
