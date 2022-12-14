from ast import parse
from lib2to3.pytree import LeafPattern
from logging.config import stopListening
from pydoc import cli
from re import L, X
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from tkinter.font import names
from xmlrpc.client import boolean
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
from clip_distances import ClipDistance
import argparse
import scipy
from sklearn.preprocessing import MinMaxScaler
import math
from create_cluster_priors import Clusters

from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel, CLIPModel, CLIPTokenizer


def text_to_projection(labels):
    # import model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # model.to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # process text data
    text_inputs = tokenizer(labels, padding=True, return_tensors="pt")
    # run models
    text_outputs = model.text_model(**text_inputs) 
    # extract pooled output
    pooled_text_output = text_outputs.pooler_output
    # project into shared representation space
    project_text_output = model.text_projection(pooled_text_output)
    # normalise
    normalized_text_output = project_text_output / project_text_output.norm(p=2, dim=-1, keepdim=True)

    return normalized_text_output

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert XML annotations to mmdetection format')
    parser.add_argument('--labeled_dataset', default='C2B')
    parser.add_argument('--labeled_data', default='labeled_data')
    parser.add_argument('--unlabeled_dataset', default='C2B')
    parser.add_argument('--unlabeled_data', default='unlabeled_data')
    parser.add_argument('--labeled_subset', default=False)
    parser.add_argument('--unlabeled_subset', default=False)
    parser.add_argument('--plot', default=1)
    parser.add_argument('--dir', default='/home/nicolas/hpc-home/ssod/')
    parser.add_argument('--prior', default='predicted')

    args = parser.parse_args()
    return args

def plot_class_distribution(cluster, prior, classes):
    data = {'cluster': [], 'class': [], 'obj_per_image': []}
    for i in range(len(cluster)):
        for j in range(len(prior[i])):
            data['cluster'].append(str(cluster[i]))
            data['class'].append(classes[j])
            data['obj_per_image'].append(prior[i][j])
    df = pd.DataFrame(data=data)
    fig = px.bar(df, x="class", y="obj_per_image",
                    color='cluster', barmode='group')
    fig.show()

def regress_distances(df, labeled_clip, unlabeled_clip):
    labeled_df = df[df['dataset']=='labeled']
    unlabeled_df = df[df['dataset']=='unlabeled']

    model = LinearRegression(fit_intercept=True)
    y = labeled_df[labeled_clip.classes].values.reshape(-1,len(labeled_clip.classes))
    y_scaler = MinMaxScaler().fit(y)
    y_norm = y_scaler.transform(y)

    x = labeled_df[labeled_clip.labels].values.reshape(-1,len(labeled_clip.labels))
    x_scaler = MinMaxScaler().fit(x)
    x_norm = x_scaler.transform(x)

    # alternatively we can use the clip embeddings
    # x = []
    # for name in labeled_df['names']:
        # i = labeled_clip.names.index(name)
        # x.append(labeled_clip.clip_embeddings[i])

    model.fit(x, y)
    # print('\n')
    # print(model.coef_)
    # print(model.intercept_)

    # pass the means through the regression model. This is the same as passing all images through individually, and then calculating the mean output.
    x = [[unlabeled_df.describe()[label]['mean'] for label in unlabeled_clip.labels]]
    # x = []
    # for i in range(len(unlabeled_df)):
        # x.append([unlabeled_df.iloc[i][label] for label in unlabeled_clip.labels])
    x_norm = x_scaler.transform(x)
    
    # x = []
    # for name in unlabeled_df['names']:
        # i = unlabeled_clip.names.index(name)
        # x.append(unlabeled_clip.clip_embeddings[i])

    y = model.predict(x)
    # y = y_scaler.inverse_transform(y)
  
    return y

def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


if __name__ == "__main__":

    args = parse_args()
    labeled_dataset = args.labeled_dataset
    labeled_data = args.labeled_data
    unlabeled_dataset = args.unlabeled_dataset
    unlabeled_data = args.unlabeled_data
    labeled_subset = int(args.labeled_subset)
    unlabeled_subset = int(args.unlabeled_subset)
    plot = int(args.plot)
    global DIR
    DIR = args.dir

    parent_dataset = labeled_dataset
    for i in range(len(parent_dataset)-1, 0, -1):
        if parent_dataset[i].isdigit():
            parent_dataset = parent_dataset[0:i]
        else:
            break

    # generate subsets
    if labeled_subset:
        with open(f'{DIR}cluster_priors/dataset_subsets/{labeled_dataset}_{labeled_data}_subsets.pkl', 'rb') as handle:
            subset = pickle.load(handle)
            names_l = subset[labeled_subset]
    else:
        names_l = False
    if unlabeled_subset:
        with open(f'{DIR}cluster_priors/dataset_subsets/{unlabeled_dataset}_{unlabeled_data}_subsets.pkl', 'rb') as handle:
            subset = pickle.load(handle)
            names_u = subset[unlabeled_subset]
    else:
        names_u = False

    # generate clusters using text labels
    if parent_dataset in ['C2B', 'D2N', 'C2N', 'C2F']:
        labels = ["a photo of cars", "a photo of people", "a photo of bicycles", "a photo of trucks", "a photo of riders", "a photo of motorcycles", "a photo of buses"]
    elif parent_dataset in ['OAK']:
        labels = ["a photo of bicycles", "a photo of buses", "a photo of cars", "a photo of chairs", "a photo of dining tables", "a photo of people", "a photo of potted plants"]
    
    # get text embedding in multi-modal space
    normalized_text_output = text_to_projection(labels)

    # run classification on labeled data
    labeled_clip = ClipDistance(labeled_dataset, labeled_data, image_subset=names_l, plot=plot, data_name='labeled', path=DIR)
    labeled_clip.calculate_clip_distances(normalized_text_output, labels)
    labeled_means = [labeled_clip.data_df.describe()[label]['mean'] for label in labeled_clip.labels]
    labeled_object_frequency = [labeled_clip.data_df.describe()[cls]['mean'] for cls in labeled_clip.classes]
    labeled_object_frequency = [0.01 if x<=0 else x for x in labeled_object_frequency]

    # run classification on unlabeled data
    unlabeled_clip = ClipDistance(unlabeled_dataset, unlabeled_data, image_subset=names_u, plot=plot, data_name='unlabeled', path=DIR)
    unlabeled_clip.calculate_clip_distances(normalized_text_output, labels)
    unlabeled_means = [unlabeled_clip.data_df.describe()[label]['mean'] for label in unlabeled_clip.labels]
    unlabeled_object_frequency = [unlabeled_clip.data_df.describe()[cls]['mean'] for cls in unlabeled_clip.classes]
    unlabeled_object_frequency = [0.01 if x<=0 else x for x in unlabeled_object_frequency]

    df = pd.concat([labeled_clip.data_df, unlabeled_clip.data_df])
    df = df.reset_index(drop=True)
    clip_distances = df[labeled_clip.labels].values

    # print(df)
    # print(df.describe())

    # predict the absolute occurence of objects
    predicted_object_frequency = regress_distances(df, labeled_clip, unlabeled_clip)
    predicted_object_frequency = np.mean(predicted_object_frequency, axis=0)
    predicted_object_frequency = [0.01 if x<=0 else x for x in predicted_object_frequency]
    scale_1 = [predicted_object_frequency[i]/labeled_object_frequency[i] for i in range(len(predicted_object_frequency))]

    # change df to probability
    df['sum'] = df[labeled_clip.classes].sum(axis=1)
    for i in range(len(df)):
        for cls in labeled_clip.classes:
            if df.loc[i]['sum'] != 0:
                df.at[i,cls] = df.loc[i][cls]/df.loc[i,'sum']
            else:
                df.at[i,cls] = 0
    
    # predict relative probability of object occurence
    prediction_relative_distribution = regress_distances(df, labeled_clip, unlabeled_clip)
    prediction_relative_distribution = np.mean(prediction_relative_distribution, axis=0)
    prediction_relative_distribution = [x*sum(labeled_object_frequency)/sum(prediction_relative_distribution) for x in prediction_relative_distribution]
    prediction_relative_distribution = [0.01 if x<=0 else x for x in prediction_relative_distribution]
    scale_2 = [prediction_relative_distribution[i]/labeled_object_frequency[i] for i in range(len(prediction_relative_distribution))]

    # merge each distribution using geometric mean
    predicted_distribution_mean = [labeled_object_frequency[i]*math.sqrt(scale_1[i]*scale_2[i]) for i in range(len(labeled_object_frequency))]
    predicted_distribution_mean = [0.01 if x<=0 else x for x in predicted_distribution_mean]
    predicted_distribution_mean = [x/sum(predicted_distribution_mean) for x in predicted_distribution_mean]

    # scale up via knowledge of number of images per class distribution
    # predicted_distribution_scaled = [predicted_distribution[i]*unlabeled_object_frequency[i]/labeled_object_frequency[i] if ((predicted_object_frequency[i]!=0) and (labeled_object_frequency[i]!=0)) else 0 for i in range(len(predicted_distribution))]
    predicted_distribution_squared = [labeled_object_frequency[i]*scale_1[i]*scale_2[i] for i in range(len(labeled_object_frequency))]
    predicted_distribution_squared = [0.01 if x<=0 else x for x in predicted_distribution_squared]
    predicted_distribution_squared = [x/sum(predicted_distribution_squared) for x in predicted_distribution_squared]

    # store class ratios as list for displaying results
    priors = [unlabeled_object_frequency, predicted_object_frequency, prediction_relative_distribution, labeled_object_frequency]
    names = ['unlabeled_distribution','instances_per_img_prediction','class_ratio_prediction', 'labeled_distribution']


    # normalize each prior
    norm_priors = []
    for prior in priors:
        norm_priors.append([x/sum(prior) for x in prior])
    
    # plot priors
    if plot:   
        plot_class_distribution(names, norm_priors, labeled_clip.classes)
        priors = [norm_priors[0], predicted_distribution_squared, predicted_distribution_mean, norm_priors[3]]
        names = ['unlabeled_distribution','squared prediction', 'merged prediction', 'labeled_distribution']   
        plot_class_distribution(names, priors, labeled_clip.classes)

    # calculate distances
    distribution_shift = KL(norm_priors[0], norm_priors[3])
    prediction_error = KL(norm_priors[0], predicted_distribution_mean)
    scaled_error = KL(norm_priors[0], predicted_distribution_squared)

    # print distances between distributions
    print('DIFFERENCE BETWEEN LABELED AND UNLABELED: ', distribution_shift)
    print('MERGED ERROR: ', prediction_error)
    print('SQUARED ERROR: ', scaled_error)

    # write the prediction to file for use in uda
    clusters = Clusters(labeled_dataset, root=DIR)
    if args.prior == 'predicted':
        clusters.manual_prior(predicted_distribution_squared, sum(labeled_object_frequency))
    elif args.prior == 'unlabeled':
        clusters.manual_prior(norm_priors[0], sum(unlabeled_object_frequency))
    elif args.prior == 'labeled':
        clusters.manual_prior(norm_priors[3], sum(labeled_object_frequency))



