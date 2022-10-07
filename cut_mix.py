from dataset import dataset
from torch.utils.data import DataLoader,Dataset
import torch
import torchvision
import os
import torchvision.transforms as transforms
import torch.optim as optim
from math import isnan
import numpy as np
import h5py
import cv2
from copy import deepcopy

BATCH_SIZE = 100

transform_test = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

beta = 1


random_seed = 3

np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    # cut_rat = np.sqrt(1. - lam)
    cut_rat = 0.2
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    while True:
        if (1/6)*W<cx<(5/6)*W or (1/6)*H<cy<(5/6)*H:
            cx = np.random.randint(W)
            cy = np.random.randint(H)
        else:
            break


    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

images = []
labels = []

imgs = []
labels = []


for i, (input, target) in enumerate(testloader):

    print(target)

    imgs1 = deepcopy(input)


    for i in range(1):
        # generate mixed sample
        imgs1 = deepcopy(input)
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(input.size()[0]).cuda()
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        print(bbx1,bby1,bbx2,bby2)
        imgs1[:, :, bbx1:bbx2, bby1:bby2] = imgs1[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        imgs.append(imgs1)
        labels.append(target)

    break


imgs = torch.cat(imgs,dim=0)
labels = torch.cat(labels,dim=0)
imgs = imgs.transpose(1,3)

imgs = np.array(imgs)
labels = np.array(labels)


print(imgs.shape)
print(labels.shape)

file1 = h5py.File('data/cut_mix_final.h5','w')
file1.create_dataset("/data",data=imgs)
file1.create_dataset("/label",data=labels)


