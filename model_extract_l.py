from dataset import dataset
from torch.utils.data import DataLoader,Dataset
import torch
import torchvision
import os
import torchvision.transforms as transforms
import torch.optim as optim
from math import isnan
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
BATCH_SIZE = 512
EPOCH = 200
dir ='model'


def reset(cls):
    if cls == 'resnet':
        model = torchvision.models.resnet18(pretrained=False)
        in_feature = model.fc.in_features
        model.fc = torch.nn.Linear(in_feature, 10)
    elif cls == "vgg":
        model = torchvision.models.vgg13(pretrained=False)
        in_feature = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_feature, 10)

    elif cls == 'dense':
        model = torchvision.models.densenet121(pretrained=False)
        in_feature = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_feature, 10)

    elif cls == 'mobile':
        model = torchvision.models.mobilenet_v2(pretrained=False)
        in_feature = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_feature, 10)


    model.cuda()
    return model


def train_student_model(iter,teacher,cls):

    teacher = teacher.cuda()
    teacher.eval()

    accu_best = 0
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False)

    train_data = dataset('dataset_attack_1.h5', train=False)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    model = reset(cls)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    count = 0

    for epoch in range(EPOCH):

        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, _ = x.cuda(), y.cuda()
            teacher_output = teacher(b_x)
            pred = torch.max(teacher_output, 1)[1].data.squeeze().detach()
            output = model(b_x)
            loss = loss_func(output, pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (i % 10 == 0):
                print("Epoch:", epoch + 1, "iteration:", i, "loss:", loss.data.item())

        if isnan(loss.data.item())==1:
            model = reset(cls)

        model.eval()
        num = 0
        total_num = 0

        for i, (x, y) in enumerate(testloader):
            x = x.type(torch.FloatTensor)
            y = y.long()
            b_x, b_y = x.cuda(), y.cuda()
            output = model(b_x)
            pred = torch.max(output, 1)[1].data.squeeze()
            num += (pred == b_y).sum().item()
            total_num += pred.shape[0]


        accu1 = num / total_num

        print("Epoch:", epoch + 1, "accuracy:", accu1)

        # if accu1 > accu_best:
        #     torch.save(model.state_dict(), os.path.join(dir, "student_model_1_" + str(iter) + ".pth"))
        #     accu_best = accu1


        if accu_best<0.12:
            count += 1

        if count>15:
            model = reset(cls)
            count = 0


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

        if iters < 5:
            cls = 'vgg'
        elif 5 <= iters < 10:
            cls = 'resnet'
        elif 10 <= iters < 15:
            cls = 'dense'
        elif 15 <= iters < 20:
            cls = 'mobile'
        print("Beigin training model:", iters,"student model:",cls)
        accu = train_student_model(iters,teacher,cls)
        print("Model {} has been trained and the accuracy is {}".format(iters, accu))

    # iter = 1
    # accu = train_student_model(iter,teacher)
    # print("Model {} has been trained and the accuracy is {}".format(iter, accu))