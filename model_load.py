import numpy as np
from torch.utils.data import DataLoader,Dataset
import torch
import torchvision
import os
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import time
import h5py
from dataset import  dataset1
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
BATCH_SIZE = 128


def load_model(num,mode):
    if mode == 'teacher':
        model = torchvision.models.vgg16_bn(pretrained=False)
        in_feature = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(os.path.join("model", "vgg_model.pth")))


    elif mode == 'student':

        if 5<=num<10:
            model = torchvision.models.resnet18(pretrained=False)
            in_feature = model.fc.in_features
            model.fc = torch.nn.Linear(in_feature, 10)
        elif num<5:
            model = torchvision.models.vgg13(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)
        elif 15>num>=10:
            model = torchvision.models.densenet121(pretrained=False)
            in_feature = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_feature, 10)

        elif 20>num>=15:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)

        model.load_state_dict(torch.load(os.path.join("model", "student_model_1_" + str(num) + ".pth")))


    elif mode == "student_kd":

        if 5<=num<10:
            model = torchvision.models.resnet18(pretrained=False)
            in_feature = model.fc.in_features
            model.fc = torch.nn.Linear(in_feature, 10)
        elif num<5:
            model = torchvision.models.vgg13(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)

        elif 15>num>=10:
            model = torchvision.models.densenet121(pretrained=False)
            in_feature = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_feature, 10)

        elif 20>num>=15:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)

        model.load_state_dict(torch.load(os.path.join("model", "student_model_kd_" + str(num) + ".pth")))



    elif mode == "teacher_kd":

        if num>=3:
            model = torchvision.models.resnet34(pretrained=False)
            in_feature = model.fc.in_features
            model.fc = torch.nn.Linear(in_feature, 10)
        else:
            model = torchvision.models.vgg19_bn(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)

        model.load_state_dict(torch.load(os.path.join("model", "student_model_tk_" + str(num) + ".pth")))


    elif mode == "irrelevant":
        if 10 >num>=5:
            model = torchvision.models.resnet18(pretrained=False)
            in_feature = model.fc.in_features
            model.fc = torch.nn.Linear(in_feature, 10)
        elif num < 5:
            model = torchvision.models.vgg13(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)

        elif 15 > num >= 10:
            model = torchvision.models.densenet121(pretrained=False)
            in_feature = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_feature, 10)

        elif 20 > num >= 15:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)


        model.load_state_dict(torch.load(os.path.join("model", "clean_model_" + str(num) + ".pth")))


    elif mode == 'finetune-100':
        model = torchvision.models.vgg16_bn(pretrained=False)
        in_feature = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(os.path.join("finetune_model", "finetune_C_" + str(num) + ".pth")))

    elif mode == 'CIFAR100':
        model = torchvision.models.vgg16_bn(pretrained=False)
        in_feature = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(os.path.join("finetune_model", "CIFAR100" + str(num) + ".pth")))

    elif mode == 'finetune-10C':
        model = torchvision.models.vgg16_bn(pretrained=False)
        in_feature = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(os.path.join("finetune_model", "finetune_cifar10c_" + str(num) + ".pth")))

    elif mode == 'CIFAR10C':
        if num >= 5:
            model = torchvision.models.resnet18(pretrained=False)
            in_feature = model.fc.in_features
            model.fc = torch.nn.Linear(in_feature, 10)
        elif num < 5:
            model = torchvision.models.vgg16_bn(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)

        model.load_state_dict(torch.load(os.path.join("finetune_model", "CIFAR10C_" + str(num) + ".pth")))

    elif mode == 'adv_train':

        if 5<=num<10:
            model = torchvision.models.resnet18(pretrained=False)
            in_feature = model.fc.in_features
            model.fc = torch.nn.Linear(in_feature, 10)
        elif num<5:
            model = torchvision.models.vgg13(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)
        elif 15>num>=10:
            model = torchvision.models.densenet121(pretrained=False)
            in_feature = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_feature, 10)

        elif 20>num>=15:
            model = torchvision.models.mobilenet_v2(pretrained=False)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)

        model.load_state_dict(torch.load(os.path.join("adv_train", "adv_" + str(num) + ".pth")))

    elif mode == "fine-pruning":
        model = torch.load("Fine-Pruning/prune_model_"+str(num)+".pth")


    elif mode == 'finetune_normal':
        model = torchvision.models.vgg16_bn(pretrained=False)
        in_feature = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(torch.load(os.path.join("finetune_10", "finetune" + str(num) + ".pth")))


    return model


