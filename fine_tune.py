from dataset import dataset
from torch.utils.data import DataLoader,Dataset
import torch
import torchvision
import os
import torchvision.transforms as transforms
import torch.optim as optim
from math import isnan
os.environ["CUDA_VISIBLE_DEVICES"]='0'
BATCH_SIZE = 512
EPOCH = 30
dir ='finetune_10'


def finetune_model(iter,teacher,cls):

    teacher = teacher.cuda()
    teacher.train()
    accu_best = 0

    train_data = dataset('dataset_attack_1.h5', train=False)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False)

    loss_func = torch.nn.CrossEntropyLoss()

    if cls == 'all':
        optimizer = optim.SGD(teacher.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)
    elif cls == 'last':
        optimizer = optim.SGD(teacher.classifier.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)

    for epoch in range(EPOCH):

        teacher.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, b_y = x.cuda(), y.cuda()
            teacher_output = teacher(b_x)
            loss = loss_func(teacher_output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i % 20 == 0):
                print("Epoch:", epoch + 1, "iteration:", i, "loss:", loss.data.item())
        teacher.eval()
        num = 0
        total_num = 0
        for i, (x, y) in enumerate(testloader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, b_y = x.cuda(), y.cuda()
            output = teacher(b_x)
            pred = torch.max(output, 1)[1].data.squeeze()
            num += (pred == b_y).sum().item()
            total_num += pred.shape[0]


        accu1 = num / total_num

        print("Epoch:", epoch + 1, "accuracy:", accu1)

        # if accu1 > accu_best:
        #     torch.save(teacher.state_dict(), os.path.join(dir, "finetune" + str(iter) + ".pth"))
        #     accu_best = accu1


    return accu_best

if __name__ == "__main__":

    if os.path.exists(dir) == 0:
        os.mkdir(dir)

    teacher = torchvision.models.vgg16_bn(pretrained=False)
    in_feature = teacher.classifier[-1].in_features
    teacher.classifier[-1] = torch.nn.Linear(in_feature, 10)
    teacher.load_state_dict(torch.load("model/vgg_model.pth"))
    teacher.eval()


    for iter in range(20):
        iters = iter

        if iters<10:
            cls = 'all'
        elif 10<=iters<20:
            cls = 'last'

        print("Beigin training model:", iters,"Model:",cls)
        accu = finetune_model(iters,teacher,cls)
        teacher.load_state_dict(torch.load("model/vgg_model.pth"))
        print("Model {} has been trained and the accuracy is {}".format(iters, accu))

    # iter = 1
    # accu = train_student_model(iter,teacher)
    # print("Model {} has been trained and the accuracy is {}".format(iter, accu))
