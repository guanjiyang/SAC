from dataset import dataset
from torch.utils.data import DataLoader,Dataset
import torch
import torchvision
import os
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from model_load import load_model
import numpy as np
from copy import deepcopy
from torch.autograd import Variable
import h5py
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
dir = "adv_train"
BATCH_SIZE = 256


def denormalize(image):
    image_data = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]
    image_data = np.array(image_data)

    img_copy = torch.zeros(image.shape).cuda()
    for i in range(3):
        img_copy[:,i,:,:] = image[:,i,:,:]*image_data[1,i] + image_data[0,i]

    return img_copy

def normalize(image):
    image_data = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]
    image_data = np.array(image_data)
    img_copy = torch.zeros(image.shape).cuda()
    for i in range(3):
        img_copy[:, i, :, :] = (image[:, i, :, :] - image_data[0, i])/image_data[1,i]

    return img_copy

def PGD(model,image,label):
    label = label.cuda()
    loss_func1 = torch.nn.CrossEntropyLoss()
    image_de = denormalize(deepcopy(image))
    image_attack = deepcopy(image)
    image_attack = image_attack.cuda()
    image_attack = Variable(image_attack, requires_grad=True)
    alpha = 1/256
    epsilon = 4/256

    for iter in range(30):
        image_attack = Variable(image_attack, requires_grad=True)
        output = model(image_attack)
        loss = -loss_func1(output,label)
        loss.backward()
        grad = image_attack.grad.detach().sign()
        image_attack = image_attack.detach()
        image_attack = denormalize(image_attack)
        image_attack -= alpha*grad
        eta = torch.clamp(image_attack-image_de,min=-epsilon,max=epsilon)
        image_attack = torch.clamp(image_de+eta,min=0,max=1)
        image_attack = normalize(image_attack)
    pred_prob = output.detach()
    pred = torch.argmax(pred_prob, dim=-1)
    acc_num = torch.sum(label == pred)
    num = label.shape[0]
    acc = acc_num/num
    acc = acc.data.item()

    return image_attack.detach(), acc


def adv_train(model,teacher,train_loader,test_loader,iter):
    model = model.cuda()
    model.train()
    teacher.eval()
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)
    for epoch in range(2):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, _ = x.cuda(), y.cuda()
            teacher_output = teacher(b_x)
            pred = torch.max(teacher_output, 1)[1].data.squeeze().detach()
            x_adv,acc = PGD(model,b_x,pred)
            output = model(b_x)
            output_adv = model(x_adv)
            loss = loss_func(output, pred) + loss_func(output_adv, pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i % 20 == 0):
                print("Epoch:", epoch + 1, "iteration:", i, "loss:", loss.data.item(),"ASR:",1-acc)


        model.eval()
        num = 0
        total_num = 0

        for i, (x, y) in enumerate(test_loader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, b_y = x.cuda(), y.cuda()
            output = model(b_x)
            pred = torch.max(output, 1)[1].data.squeeze()
            num += (pred == b_y).sum().item()
            total_num += pred.shape[0]

        accu1 = num / total_num
        print("Epoch:", epoch + 1, "accuracy:", accu1)
    # torch.save(model.state_dict(), os.path.join(dir, "adv_" + str(iter) + ".pth"))
    model = model.cpu()

    return accu1


if __name__ == "__main__":

    if os.path.exists(dir)==0:
        os.mkdir(dir)

    teacher = load_model(0, "teacher")
    teacher = teacher.cuda()
    teacher.eval()

    models = []
    accus = []
    for i in range(20):
        iters = i
        globals()['student' + str(iters)] = load_model(iters, "student")
        models.append(globals()['student' + str(iters)])
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False)
    train_data = dataset('dataset_attack_1.h5', train=False)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    for i in range(len(models)):
        accu = adv_train(models[i], teacher, train_loader, test_loader, i)
        accus.append(accu)
    print(accus)