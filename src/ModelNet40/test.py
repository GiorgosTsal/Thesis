# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 19:41:39 2021

@author: root
"""

import numpy as np
import math
import random
import os
import torch
from path import Path
from source import model
from source import dataset
from source import utils
from source.args import parse_args
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import source.model

random.seed = 42


def test(args):
    path = Path(args.root_dir)
    
    folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
    classes = {folder: i for i, folder in enumerate(folders)};
    
    train_transforms = transforms.Compose([
        utils.PointSampler(1024),
        utils.Normalize(),
        utils.RandRotation_z(),
        utils.RandomNoise(),
        utils.ToTensor()
    ])
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # as is
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    print(device)
    pointnet = model.PointNet()
    pointnet.to(device)
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=args.lr)
    
    train_ds = dataset.PointCloudData(path, transform=train_transforms)
    valid_ds = dataset.PointCloudData(path, valid=True, folder='test', transform=train_transforms)
    print('Train dataset size: ', len(train_ds))
    print('Valid dataset size: ', len(valid_ds))
    print('Number of classes: ', len(train_ds.classes))
    
    train_loader = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=args.batch_size*2)
    
    
    #%% Testing Phase 
    from sklearn.metrics import confusion_matrix
   
    pointnet = model.PointNet()
    pointnet.load_state_dict(torch.load('checkpoints/save_8.pth'))
    pointnet.eval();

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            print('Batch [%4d / %4d]' % (i+1, len(valid_loader)))
                       
            inputs, labels = data['pointcloud'].float(), data['category']
            outputs, __, __ = pointnet(inputs.transpose(1,2))
            _, preds = torch.max(outputs.data, 1)
            all_preds += list(preds.numpy())
            all_labels += list(labels.numpy())
            
    cm = confusion_matrix(all_labels, all_preds);
    print("=========")
    print("Confusion Matrix second:")
    print(cm)
    
    import itertools
    import matplotlib.pyplot as plt
    
    # function from https://deeplizard.com/learn/video/0LhiS6yu2qQ
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    
    plt.figure(figsize=(8,8))
    plot_confusion_matrix(cm, list(classes.keys()), normalize=True)
    
    
            
    cm = confusion_matrix(all_labels, all_preds);

if __name__ == '__main__':
    args = parse_args()
    test(args)

