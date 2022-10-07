import numpy as np
from dataset import dataset,dataset3,dataset4,dataset1
from torch.utils.data import DataLoader,Dataset
import torch
import torchvision
import os
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from model_load import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
BATCH_SIZE = 128

class FeatureHook():

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):

        self.output = output

    def close(self):
        self.hook.remove()



def cal_accu(model,dataloader):
    acc = 0
    num = 0
    for i,(x,y) in enumerate(dataloader):
        x, y = x.cuda(), y.cuda()
        output = model(x)
        output = output.detach()
        pred = torch.argmax(output,dim=-1)
        acc += torch.sum(y==pred)
        num += y.shape[-1]

    accu = acc/num

    return accu

def test_acc(models):
    accuracy = []
    for i in range(len(models)):
        model = models[i]
        model = model.cuda()
        model.eval()
        accu = cal_accu(model, test_loader).cpu().numpy()
        accuracy.append(accu)

        model = model.cpu()

    accuracy = np.array(accuracy)
    return accuracy



if __name__ == '__main__':

    models = []
    cor_mats = []

    # transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    test_data = dataset4("CIFAR10C_test.h5", train=False)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE)

    # test_data = dataset3("CIFAR100_test.h5", train=False)
    # test_loader = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE)

    accus = []


    for i in range(10):
        globals()['student_model' + str(i)] = load_model(i, "CIFAR10C")
        models.append(globals()['student_model' + str(i)])

    accuracy = test_acc(models)
    accus.append(accuracy)
    accuracy = np.array(accuracy)


    print(accuracy)
    print(np.average(accuracy))


