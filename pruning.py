import argparse
import sys
import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
import torch.nn as nn
from dataset import dataset
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import random
import h5py
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.prune as prune
import torch.utils.data as Data
from model_load import load_model
from copy import deepcopy


os.environ['CUDA_VISIBLE_DEVICES'] = '5'
dir = 'Fine-Pruning'

BATCH_SIZE = 256
LR = 0.001
prune_ratio = 0.90

def idx_change(idx, neuron_num):
    total = 0
    for i in range(neuron_num.shape[0]):
        total += neuron_num[i]
        if idx < total:
            layer_num = i
            layer_idx = idx - (total - neuron_num[i])
            break
    return layer_num, layer_idx


def prune_neuron(mask_list, idx, neuron_num):
    layer_num, layer_idx = idx_change(idx, neuron_num)
    mask_list[layer_num].weight_mask[layer_idx] = 0


class FeatureHook():

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):

        self.output = output

    def close(self):
        self.hook.remove()

def find_smallest_neuron(hook_list,prune_list):
    activation_list = []
    for j in range(len(hook_list)):
        activation = hook_list[j].output
        for i in range(activation.shape[1]):
            activation_channel = torch.mean(torch.abs(activation[:,i,:,:]))
            activation_list.append(activation_channel)

    activation_list1 = []
    activation_list2 = []

    for n, data in enumerate(activation_list):
        if n in prune_list:
            pass
        else:
            activation_list1.append(n)
            activation_list2.append(data)

    activation_list2 = torch.tensor(activation_list2)
    prune_num = torch.argmin(activation_list2)
    prune_idx = activation_list1[prune_num]

    return prune_idx

def finetune_step(model, dataloader, criterion):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    for i,(inputs,labels) in enumerate(dataloader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        labels=labels.long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i+1)*inputs.shape[0]>= 2056:
            break


def value(model, dataloader):
    model.eval()
    num = 0
    total_num = 0
    for i, (x, y) in enumerate(dataloader):
        y = y.long()
        b_x, b_y = x.cuda(), y.cuda()
        output = model(b_x)
        pred = torch.max(output, 1)[1].data.squeeze()
        num += (pred == b_y).sum().item()
        total_num += pred.shape[0]

    accu = num / total_num
    return accu

def run_model(model, dataloader):
    model.eval()
    for i, (x, y) in enumerate(dataloader):
        y = y.long()
        b_x, b_y = x.cuda(), y.cuda()
        output = model(b_x)
        pred = torch.max(output, 1)[1].data.squeeze()


def idx_change(idx, neuron_num):
    total = 0
    for i in range(neuron_num.shape[0]):
        total += neuron_num[i]
        if idx < total:
            layer_num = i
            layer_idx = idx - (total - neuron_num[i])
            break
    return layer_num, layer_idx


def prune_neuron(mask_list, idx, neuron_num):
    layer_num, layer_idx = idx_change(idx, neuron_num)
    mask_list[layer_num].weight_mask[layer_idx] = 0


def plot_figure(mem,length):
    plt.figure(1)
    acc = np.squeeze(mem)
    plt.plot(np.squeeze(np.array(acc)[:, 0])/length, np.squeeze(np.array(acc)[:, 1]), 'b',label='Clean Classification Accuracy')
    plt.xlabel("Ratio of Neurons Pruned")
    plt.ylabel("Rate")
    plt.legend()
    plt.show()


def fine_pruning(model, train_loader, test_loader):
    model = model.cuda()
    module_list = []
    neuron_num = []
    hook_list = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module_list.append(module)
            neuron_num.append(module.out_channels)
            hook_list.append(FeatureHook(module))

    neuron_num = np.array(neuron_num)
    max_id = np.sum(neuron_num)

    neuron_list = []
    mask_list = []
    for i in range(neuron_num.shape[0]):
        neurons = list(range(neuron_num[i]))
        neuron_list.append(neurons)
        prune_filter = prune.identity(module_list[i], 'weight')
        mask_list.append(prune_filter)

    prune_list = []
    init_val = value(model, test_loader)
    acc = []
    length = deepcopy(len(neuron_list))
    total_length = 0
    for i in range(length):
        total_length += len(neuron_list[i])
    print("Total number of neurons is",total_length)
    for i in range(int(np.floor(0.8*total_length))):
        if i % 20 == 0:
            run_model(model, train_loader)
        idx = find_smallest_neuron(hook_list, prune_list)
        prune_list.append(idx)
        prune_neuron(mask_list, idx, neuron_num)
        if i % 50 == 0:
            finetune_step(model, train_loader, criterion=torch.nn.CrossEntropyLoss())
        if i % 50 == 0:
            new_val = value(model, test_loader)
            print("neuron remove:", i, "init_value:", init_val, "new_value:", new_val)
            acc.append([i, new_val])

        if (np.floor(20*i/total_length)-np.floor(20*(i-1)/total_length)) == 1:
            iter = int(np.floor(20*i/total_length))
            # torch.save(model, "Fine-Pruning/prune_model_" + str(iter)+".pth")
            print("Saving model! Model number is:",iter)

    # mem = np.array([acc])
    # np.save("Fine-Pruning/fp", mem)
    return mem,length


if __name__ == "__main__":
    if os.path.exists(dir) == 0:
        os.mkdir(dir)
        print("Making directory!")

    teacher = load_model(0, "teacher")
    teacher = teacher.cuda()
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    train_data = dataset('dataset_attack_1.h5', train=False)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    mem,length = fine_pruning(teacher, train_loader, test_loader)
    # plot_figure(mem, length)
