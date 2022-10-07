from dataset import dataset1,dataset4
from torch.utils.data import DataLoader,Dataset
import torch
import torchvision
import os
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from model_load import load_model
from sklearn.metrics import roc_curve,auc



os.environ["CUDA_VISIBLE_DEVICES"] = '3'
BATCH_SIZE = 100


def calculate_auc(list_a, list_b):
    l1,l2 = len(list_a),len(list_b)
    y_true,y_score = [],[]
    for i in range(l1):
        y_true.append(0)
    for i in range(l2):
        y_true.append(1)
    y_score.extend(list_a)
    y_score.extend(list_b)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return auc(fpr, tpr)


class FeatureHook():

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.output = output
    def close(self):
        self.hook.remove()



def correlation(m,n):
    m = F.normalize(m,dim=-1)
    n = F.normalize(n,dim=-1).transpose(0,1)
    cose = torch.mm(m,n)
    matrix = 1-cose
    matrix = matrix/2
    return matrix

def pairwise_euclid_distance(A):
    sqr_norm_A = torch.unsqueeze(torch.sum(torch.pow(A, 2),dim=1),dim=0)
    sqr_norm_B = torch.unsqueeze(torch.sum(torch.pow(A, 2), dim=1), dim=1)
    inner_prod = torch.matmul(A, A.transpose(0,1))
    tile1 = torch.reshape(sqr_norm_A,[A.shape[0],1])
    tile2 = torch.reshape(sqr_norm_B,[1,A.shape[0]])
    return tile1+tile2 - 2*inner_prod


def correlation_dist(A):
    A = F.normalize(A,dim=-1)
    cor = pairwise_euclid_distance(A)
    cor = torch.exp(-cor)

    return cor


def cal_cor(model,dataloader):
    model.eval()
    model = model.cuda()
    outputs = []
    for i,(x,y) in enumerate(dataloader):
        # if i!=0:
        #    break

        x, y = x.cuda(), y.cuda()
        output = model(x)
        outputs.append(output.cpu().detach())

    output = torch.cat(outputs,dim=0)
    cor_mat = correlation(output,output)

    # cor_mat = correlation_dist(output)

    model = model.cpu()
    return cor_mat

def cal_cor_onehot(model,dataloader):
    model.eval()
    model = model.cuda()
    outputs = []
    for i,(x,y) in enumerate(dataloader):
        x, y = x.cuda(), y.cuda()
        output = model(x)
        output = output_to_label(output)
        outputs.append(output.cpu().detach())
    output = torch.cat(outputs,dim=0)
    cor_mat = correlation(output,output)


    model = model.cpu()
    return cor_mat

def output_to_label(output):
    shape = output.shape
    pred = torch.argmax(output,dim=1)
    preds = 0.01 * torch.ones(shape)

    for i in range(shape[0]):
        preds[i,pred[i]]=1

    preds = torch.softmax(preds,dim=-1)

    # print(preds[0,:])
    return preds




def cal_correlation(models,i):

    cor_mats = []

    # SAC-normal
    # transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # train_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    # SAC-w
    train_data = dataset1('dataset_common.h5', train=False)
    train_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)

    # SAC-m (with flip)
    # train_data = dataset1('cut_mix_100.h5', train=False)
    # train_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)

    # SAC-m (without flip)
    # train_data = dataset1('cut_mix_100_nf.h5', train=False)
    # train_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)

    # SAC-m (50 samples)
    # train_data = dataset1('cut_mix_50.h5', train=False)
    # train_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)

    # SAC-source
    # train_data = dataset1('source_wrong_final.h5', train=False)
    # train_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)

    # train_data = dataset1('cut_mix_final.h5', train=False)
    # train_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)


    for i in range(len(models)):

        model = models[i]
        cor_mat = cal_cor(model, train_loader)
        cor_mats.append(cor_mat)

    print(len(cor_mats), cor_mat.shape)

    diff = torch.zeros(len(models))



    for i in range(len(models) - 1):
        iter = i + 1
        diff[i] = torch.sum(torch.abs(cor_mats[iter] - cor_mats[0])) / (cor_mat.shape[0] * cor_mat.shape[1])
        # print(cor_mat.shape[0],cor_mat.shape[1])


    print("Correlation difference is:", diff[:20])
    print("Correlation difference is:", diff[20:40])
    print("Correlation difference is:", diff[40:60])
    print("Correlation difference is:", diff[60:70])
    print("Correlation difference is:", diff[70:80])
    print("Correlation difference is:", diff[80:90])
    print("Correlation difference is:", diff[90:100])
    print("Correlation difference is:", diff[100:120])
    print("Correlation difference is:", diff[120:135])
    print("Correlation difference is:", diff[135:155])

    list1 = diff[:20]
    list2 = diff[20:40]
    list3 = diff[40:60]
    list4 = diff[60:70]
    list5 = diff[70:80]
    list6 = diff[80:90]
    list7 = diff[90:100]
    list8 = diff[100:120]
    list9 = diff[120:135]
    list10 = diff[135:155]

    auc_p = calculate_auc(list1, list3)
    auc_l = calculate_auc(list2, list3)
    auc_finetune = calculate_auc(list10, list3)
    auc_adv = calculate_auc(list8, list3)
    auc_prune = calculate_auc(list9[:10], list3)
    auc_100 = calculate_auc(list4, list5)
    auc_10C = calculate_auc(list6, list7)

    print("Calculating AUC:")

    print("AUC_P:",auc_p,"AUC_L:", auc_l, "AUC_Finetune:",auc_finetune,"AUC_Prune:", auc_prune,"AUC_Adv:",auc_adv, "AUC_100:", auc_100,"AUC_10C:", auc_10C)


if __name__ == '__main__':




    models = []


    for i in [0]:
        globals()['teacher' + str(i)] = load_model(i, "teacher")
        models.append(globals()['teacher' + str(i)])


    for i in range(20):
        globals()['student_kd' + str(i)] = load_model(i, "student_kd")
        models.append(globals()['student_kd' + str(i)])

    for i in range(20):
        globals()['student' + str(i)] = load_model(i, "student")
        models.append(globals()['student' + str(i)])

    for i in range(20):
        globals()['clean' + str(i)] = load_model(i, "irrelevant")
        models.append(globals()['clean' + str(i)])
    #

    for i in range(10):
        globals()['finetune' + str(i)] = load_model(i, "finetune-100")
        models.append(globals()['finetune' + str(i)])

    for i in range(10):
        globals()['CIFAR10C' + str(i)] = load_model(i, "CIFAR100")
        models.append(globals()['CIFAR10C' + str(i)])

    for i in range(10):
        globals()['finetune' + str(i)] = load_model(i, "finetune-10C")
        models.append(globals()['finetune' + str(i)])
    #
    for i in range(10):
        globals()['CIFAR10C' + str(i)] = load_model(i, "CIFAR10C")
        models.append(globals()['CIFAR10C' + str(i)])


    for i in range(20):
        globals()['adv' + str(i)] = load_model(i, "adv_train")
        models.append(globals()['adv' + str(i)])

    for i in range(15):
        globals()['fp' + str(i)] = load_model(i, "fine-pruning")
        models.append(globals()['fp' + str(i)])

    for i in range(20):
        globals()['finetune_normal' + str(i)] = load_model(i, 'finetune_normal')
        models.append(globals()['finetune_normal' + str(i)])

    for i in range(1):
        iter = i
        print("Iter:", iter)
        cal_correlation(models,iter)








