import ipaddress
import sys, argparse
import numpy as np
import torch
from torch.nn.functional import relu, avg_pool2d
from buffer import Buffer
# import utils
import datetime
from torch.nn.functional import relu
import torch
import torch.nn as nn
import torch.nn.functional as F
from CSL import tao as TL
from CSL import classifier as C
from CSL.utils import normalize
from CSL.contrastive_learning import get_similarity_matrix, NT_xent, Supervised_NT_xent, SupConLoss
import torch.optim.lr_scheduler as lr_scheduler
from CSL.shedular import GradualWarmupScheduler
import torch
import torchvision.transforms as transforms
import torchvision

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
parser.add_argument('--experiment', default='cifar-10', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--lr', default=0.02, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--parameter', type=str, default='', help='(default=%(default)s)')
parser.add_argument('--dataset', type=str, default='cifar', help='(default=%(default)s)')
parser.add_argument('--input_size', type=str, default=[3, 32, 32], help='(default=%(default)s)')
parser.add_argument('--buffer_size', type=int, default=1000, help='(default=%(default)s)')
parser.add_argument('--gen', type=str, default=True, help='(default=%(default)s)')
parser.add_argument('--p1', type=float, default=0.1, help='(default=%(default)s)')
parser.add_argument('--cuda', type=str, default='1', help='(default=%(default)s)')
parser.add_argument('--n_classes', type=int, default=512, help='(default=%(default)s)')
parser.add_argument('--buffer_batch_size', type=int, default=64, help='(default=%(default)s)')
args = parser.parse_args()
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda  # use gpu0,1


def flip_inner(x, flip1, flip2):
    num = x.shape[0]

    # print(num)
    a = x  # .permute(0,1,3,2)
    a = a.view(num, 3, 2, 16, 32)
    #  imshow(torchvision.utils.make_grid(a))
    a = a.permute(2, 0, 1, 3, 4)
    s1 = a[0]  # .permute(1,0, 2, 3)#, 4)
    s2 = a[1]  # .permute(1,0, 2, 3)
    # print("a",a.shape,a[:63][0].shape)
    if flip1:
        s1 = torch.flip(s1, (3,))  # torch.rot90(s1, 2*rot1, (2, 3))
    if flip2:
        s2 = torch.flip(s2, (3,))  # torch.rot90(s2, 2*rot2, (2, 3))

    s = torch.cat((s1.unsqueeze(2), s2.unsqueeze(2)), dim=2)
    # imshow(torchvision.utils.make_grid(s[2]))
    #   print("s",s.shape)
    # S = s.permute(0,1, 2, 3, 4)  # .view(3,32,32)
    # print("S",S.shape)
    S = s.reshape(num, 3, 32, 32)
    # S =S.permute(0,1,3,2)
    # imshow(torchvision.utils.make_grid(S[2]))
    #    print("S", S.shape)
    return S


def RandomFlip(x, num):
    # print(x.shape)
    #aug_x = simclr_aug(x)
    x=simclr_aug(x)
    X = []
    # print(x.shape)

    # for i in range(4):
    X.append(x)
    X.append(flip_inner(x, 1, 1))

    X.append(flip_inner(x, 0, 1))

    X.append(flip_inner(x, 1, 0))
    # else:
    #   x1=rot_inner(x,0,1)

    return torch.cat([X[i] for i in range(num)], dim=0)


def rot_inner(x, rot1, rot2):
    num = x.shape[0]

    # print(num)
    a = x.permute(0, 1, 3, 2)
    a = a.view(num, 3, 2, 16, 32)
    #  imshow(torchvision.utils.make_grid(a))
    a = a.permute(2, 0, 1, 3, 4)
    s1 = a[0]  # .permute(1,0, 2, 3)#, 4)
    s2 = a[1]  # .permute(1,0, 2, 3)
    # print("a",a.shape,a[:63][0].shape)
    s1 = torch.rot90(s1, 2 * rot1, (2, 3))
    s2 = torch.rot90(s2, 2 * rot2, (2, 3))

    s = torch.cat((s1.unsqueeze(2), s2.unsqueeze(2)), dim=2)

    S = s.reshape(num, 3, 32, 32)
    S = S.permute(0, 1, 3, 2)

    return S


def Rotation(x, r):
    # print(x.shape)
    x = torch.rot90(x, r, (2, 3))
    X = []
    # print(x.shape)

    X.append(rot_inner(x, 0, 0))

    X.append(rot_inner(x, 1, 1))

    X.append(rot_inner(x, 1, 0))

    X.append(rot_inner(x, 0, 1))


    return x


oop = 4
print('=' * 100)
print('Arguments =')
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print('=' * 100)
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  ' + os.environ["CUDA_VISIBLE_DEVICES"])
print('=' * 100)
########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]')
    sys.exit()
import cifar as dataloader

from Resnet18 import resnet18 as b_model
from buffer import Buffer as buffer
# imagenet200 import SequentialTinyImagenet as STI
from torch.optim import Adam, SGD  # ,SparseAdam
import torch.nn.functional as F
from copy import deepcopy
import matplotlib.pyplot as plt



def test_model_cur(loder, i):
    test_loss = 0
    correct = 0
    num = 0
    for batch_idx, (data, target) in enumerate(loder):
        data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        Basic_model.eval()
        pred = Basic_model.forward(data)[:,2*(i):2*(i+1)]
        Pred = pred.data.max(1, keepdim=True)[1]
        num += data.size()[0]
        target=target-2*i

        #    print("final", Pred, target.data.view_as(Pred))
        # print(target,"True",pred)

        correct += Pred.eq(target.data.view_as(Pred)).cpu().sum()

    test_accuracy = 100. * correct / num  # len(data_loader.dataset)
    print(
        'Test set{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
            .format(i,
                    test_loss, correct, num,
                    100. * correct / num, ))
    return test_accuracy

def test_model_past(loder, i):
    test_loss = 0
    correct = 0
    num = 0
    for batch_idx, (data, target) in enumerate(loder):
        data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        Basic_model.eval()
        pred = Basic_model.forward(data)[:,:2*(i+1)]
        Pred = pred.data.max(1, keepdim=True)[1]
        num += data.size()[0]
        target=target

        #    print("final", Pred, target.data.view_as(Pred))
        # print(target,"True",pred)

        correct += Pred.eq(target.data.view_as(Pred)).cpu().sum()

    test_accuracy = 100. * correct / num  # len(data_loader.dataset)
    print(
        'Test set{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
            .format(i,
                    test_loss, correct, num,
                    100. * correct / num, ))
    return test_accuracy
def test_model_future(loder, i):
    test_loss = 0
    correct = 0
    num = 0
    for batch_idx, (data, target) in enumerate(loder):
        data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        Basic_model.eval()
        pred = Basic_model.forward(data)[:,2*i:]
        Pred = pred.data.max(1, keepdim=True)[1]
        num += data.size()[0]
        target=target-2*i

        correct += Pred.eq(target.data.view_as(Pred)).cpu().sum()

    test_accuracy = 100. * correct / num  # len(data_loader.dataset)
    print(
        'Test set{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
            .format(i,
                    test_loss, correct, num,
                    100. * correct / num, ))
    return test_accuracy


def test_model(loder, i):
    test_loss = 0
    correct = 0
    num = 0
    for batch_idx, (data, target) in enumerate(loder):
        data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        Basic_model.eval()
        pred = Basic_model.forward(data)
        Pred = pred.data.max(1, keepdim=True)[1]
        num += data.size()[0]

        correct += Pred.eq(target.data.view_as(Pred)).cpu().sum()

    test_accuracy = 100. * correct / num  # len(data_loader.dataset)
    print(
        'Test set{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
            .format(i,
                    test_loss, correct, num,
                    100. * correct / num, ))
    return test_accuracy



def get_true_prob(x, y, llabel):
    num = x.size()[0]
    true = []
    true2 = []
    for i in range(num):

        if y[i] in llabel:
            true.append(1)
        else:
            true.append(0)
        # true.append(x[i][y[i]])
        # true2.append(0.5)

        # true.append(x[i][y[i]])
    return torch.FloatTensor(true).cuda()  # ,#torch.FloatTensor(true2).cuda()


def get_prob_rate(x, logits, label):
    num = x.size()[0]
    logits = F.softmax(logits, dim=1)
    rate = []
    # true2=[]
    for i in range(num):
        true_prob = logits[i][label[i]].item()
        max_prob = torch.max(logits[i])
        rate.append(true_prob / max_prob)
    return torch.FloatTensor(rate).cuda()


def get_prob_rate_cross( logits, label, t):
    logits = F.softmax(logits, dim=1)
    rate = []
    num = logits.size()[0]
    # true2=[]
    # import pdb
    # pdb.set_trace()
    for i in range(num):
        true_prob = logits[i][label[i]].item()
        # import pdb
        # pdb.set_trace()
        max_prob = torch.max(logits[i, :-t])
        rate.append(true_prob / max_prob)
    return torch.FloatTensor(rate).cuda()
def get_mean_rate_cross( logits, label, t):
    logits = F.softmax(logits, dim=1)
    rate = []
    num = logits.size()[0]
    # true2=[]
    # import pdb
    # pdb.set_trace()
    for i in range(num):
        true_prob = logits[i][label[i]].item()
        # import pdb
        # pdb.set_trace()
        max_prob = torch.max(logits[i, :-t])
        rate.append(true_prob / max_prob)
    return torch.FloatTensor(rate).cuda()

print('Load data...')
num_class_per_task=10
data, taskcla, inputsize, Loder, test_loder = dataloader.get_cifar100_10(seed=args.seed)
data2, taskcla2, inputsize2, Loder2, test_loder2 = dataloader.get_cifar100_100d(seed=args.seed)
print('Input size =', inputsize, '\nTask info =', taskcla)
buffero = buffer(args).cuda()
Basic_model = b_model(num_class_per_task).cuda()
llabel = {}
Optimizer = Adam(Basic_model.parameters(), lr=0.001, betas=(0.9, 0.99),
                 weight_decay=1e-4)  # SGD(Basic_model.parameters(), lr=0.02, momentum=0.9)
from apex import amp
Basic_model, Optimizer = amp.initialize(Basic_model, Optimizer,opt_level="O1")
hflip = TL.HorizontalFlipLayer().cuda()
cutperm = TL.CutPerm().cuda()
with torch.no_grad():
    resize_scale = (0.6, 1.0)  # resize scaling factor,default [0.08,1]
    # if P.resize_fix: # if resize_fix is True, use same scale
    #    resize_scale = (P.resize_factor, P.resize_factor)

    # Align augmentation
   # color_jitter = TL.ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8).cuda()
    color_gray = TL.RandomColorGrayLayer(p=0.2).cuda()
    resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=[32, 32, 3]).cuda()
    simclr_aug = transform = torch.nn.Sequential(color_gray, resize_crop,
        # color_jitter,  # 这个不会变换大小，但是会变化通道值，新旧混杂
        #  resize_crop,
        )
    #color_gray,  # 这个也不会，混搭
    #    resize_crop,
# for n,w in Basic_model.named_parameters():
#   print(n,w.shape)
Max_acc = []
print('=' * 100)
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  ' + os.environ["CUDA_VISIBLE_DEVICES"])
print('=' * 100)
class_holder = []
class_prototype = {}
buffer_per_class = 7
flip_num = 2
negative_logits_SUM = None
positive_logits_SUM = None
num_SUM = 0
Category_sum=None
import pdb
#pdb.set_trace()
for run in range(1):
    # rank = torch.randperm(len(Loder))
    rank = torch.arange(0,10)#tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    for i in range(len(Loder)):
        new_class_holder = []
        print(i)
        task_id = i
        prev_index=True
        if i > 0:
            Basic_model.change_output_dim(num_class_per_task)
            Category_sum = torch.cat((Category_sum, torch.zeros(num_class_per_task)))
            negative_logits_SUM = torch.cat(
                (negative_logits_SUM, torch.zeros(num_class_per_task).cuda()))
            positive_logits_SUM = torch.cat(
                (positive_logits_SUM, torch.zeros(num_class_per_task).cuda()))

           # Category_sum = torch.cat((Category_sum, torch.zeros(num_class_per_task)))
           # negative_logits_SUM = torch.cat((negative_logits_SUM, torch.zeros(num_class_per_task).cuda()))
           # positive_logits_SUM = torch.cat((positive_logits_SUM, torch.zeros(num_class_per_task).cuda()))
        #if task_id>=2:
         #   for name,param in Basic_model.named_parameters():
          #      if "layer1.0" in name:
           #         param.requires_grad=False
           #     if "layer2.0" in name:
            #        param.requires_grad=False
             #   if "layer3.0" in name:
              #      param.requires_grad=False


        train_loader = Loder[rank[i].item()]['train']
        negative_logits_sum=None
        positive_logits_sum=None
        sum_num=0
        category_sum = None
        for epoch in range(1):
            Basic_model.train()
            num_d = 0
            for batch_idx, (x, y) in enumerate(train_loader):
              #  if batch_idx>=10:
               #     continue

                num_d += x.shape[0]
                if num_d % 5000 == 0:
                    print(num_d, num_d / 10000)
                llabel[i] = []

                Y = deepcopy(y)
                for j in range(len(Y)):
                    if Y[j] not in class_holder:

                        class_holder.append(Y[j].detach())
                        class_prototype[Y[j].detach()] = 0
                        new_class_holder.append(Y[j].detach())
                        #if i > 0:
                           # Basic_model.change_output_dim(num_class_per_task)


                       # if i > 0:
                            #Basic_model.change_output_dim(1)


                Optimizer.zero_grad()
                # if args.cuda:
                x, y = x.cuda(), y.cuda()
                ori_x = x.detach()
                ori_y = y.detach()
                x = x.requires_grad_()
              #  import pdb
               # pdb.set_trace()

                if batch_idx==0&task_id==0:
                    cur_x, cur_y,_ = torch.zeros(1),torch.zeros(1),torch.zeros(1)#,None,None#buffero.onlysample(22, task=task_id)
                else:

                    cur_x, cur_y, _,_ = buffero.onlysample(22, task=task_id)
                if len(cur_x.shape) > 3:
                    x = torch.cat((x, cur_x), dim=0)
                    y = torch.cat((y, cur_y))
                images1 = torch.cat([torch.rot90(x, rot, (2, 3)) for rot in range(1)])  # 4B
                images2 = torch.cat([torch.rot90(x, rot, (2, 3)) for rot in range(1)])  # 4B

                images_pair = torch.cat([images1, simclr_aug(images2)], dim=0)  # 8B

                labels1 = y.cuda()
                # print("LLLL",labels1.shape)
                rot_sim_labels = torch.cat([labels1 + 100 * i for i in range(1)], dim=0)
                Rot_sim_labels = torch.cat([labels1 + 0 * i for i in range(1)], dim=0)

                rot_sim_labels = rot_sim_labels.cuda()

                outputs_aux = Basic_model(images_pair, is_simclr=True)

                simclr = normalize(outputs_aux)  # normalize

                sim_matrix = get_similarity_matrix(simclr)

                loss_sim1 = Supervised_NT_xent(sim_matrix, labels=rot_sim_labels,
                                               temperature=0.07)

                if not buffero.is_empty():

                    buffer_batch_size = 64

                    # x = x.requires_grad_()
                    x = RandomFlip(x, flip_num)
                    y = y.repeat(flip_num)
                    x = x.requires_grad_()
                    hidden_pred = Basic_model.f_train(simclr_aug(x))
                    pred_y = Basic_model.linear(hidden_pred)
                    #

                    t = num_class_per_task#len(new_class_holder)
                    if task_id>0:
                        pred_y_new = pred_y[:, -t:]#torch.cat([Basic_model.linear(hidden_pred)[:, :-t].data.detach(),pred_y[:, -t:]],dim=1)
                        loss_balance = (pred_y[:,:-t]**2).mean()
                    else:
                        pred_y_new=pred_y
                        loss_balance=0
                    min_y = min(new_class_holder)
                    y_new = y - num_class_per_task*i#min_y

                    num_x=ori_y.size()[0]
                    rate=len(new_class_holder)/len(class_holder)

                    mem_x, mem_y, logits, bt = buffero.sample(int(buffer_batch_size*(1-rate))*1, exclude_task=task_id)
                    #if task_id>0:
                        #distribution = torch.ones(2 * task_id).cuda()
                        #distribution /= distribution.sum()
                       # pdb.set_trace()
                        # if task_id>=3:
                        #    pdb.set_trace()
                       # mem_x, mem_y, _, bt = buffero.pro_class_sample(int(buffer_batch_size*(1-rate))*1, distribution=distribution)
                      #  index_only = torch.randperm(mem_y_only.size()[0])
                   # mem_x_only = mem_x_only[index_only][:]
                    #mem_y_only = mem_y_only[index_only][:]
                    index_x=ori_x
                    index_y=ori_y
                    if len(cur_x.shape) > 3:
                        index_x = torch.cat((index_x, cur_x), dim=0)
                        index_y = torch.cat((index_y, cur_y))

                    all_x = torch.cat((mem_x, index_x), dim=0)
                    all_y = torch.cat((mem_y, index_y))
                      #  index_cur = torch.randperm(index_y.size()[0])
                       # index_x = index_x[index_cur][:]
                        #index_y = index_y[index_cur][:]
                  #  if len(class_holder)>len(new_class_holder):
                     #   prev_hiddens=Previous_model.forward(mem_x)
                      #  cur_hiddens=Basic_model.forward(mem_x)[:,:-len(new_class_holder)]
                #        cur_logits=torch.sum(F.softmax(Basic_model.forward(mem_x))[:,:-len(new_class_holder)],dim=1)
              #          _,idx_cur=torch.sort(cur_logits)
               #         mem_x=mem_x[idx_cur]
                #        mem_y=mem_y[idx_cur]
                      #  import pdb
                       # pdb.set_trace()
                 #       logits_cur=F.softmax(Basic_model.forward(ori_x))
                  #      logits_pre=torch.sum(logits_cur[:,:-len(new_class_holder)],dim=1)
                   #     _,idx_pre=torch.sort(logits_pre,descending=True)
                    #    ori_x=ori_x[idx_pre]
                    #    ori_y=ori_y[idx_pre]




                    mem_x = torch.cat((mem_x[:int(buffer_batch_size*(1-rate))],index_x[:int(buffer_batch_size*rate)]),dim=0)
                    mem_y = torch.cat((mem_y[:int(buffer_batch_size*(1-rate))],index_y[:int(buffer_batch_size*rate)]))
                    logits = torch.cat((logits[:int(buffer_batch_size*(1-rate))],Basic_model.f_train(index_x[:int(buffer_batch_size*rate)])),dim=0)
                    index = torch.randperm(mem_y.size()[0])
                    mem_x=mem_x[index][:]
                    mem_y=mem_y[index][:]
                    logits=logits[index][:]

                    mem_dif = torch.zeros_like(mem_x)
                    mem_dif.data = deepcopy(mem_x.data)
                    loss_div = 0
                    with torch.no_grad():
                        from utils import feat_normalized

                        feat = feat_normalized(Basic_model, mem_x)
                        feat_all = feat_normalized(Basic_model, all_x)

                        num = mem_x.shape[0]

                        # repeat_num=2
                        # mem_x = mem_x.repeat(repeat_num, 1, 1, 1)
                        mask_object = feat > 0.5#args.p2
                        mask_object_2 = feat_all > 0.5#0.5args.p2

                    for ii in range((task_id) * 2):
                        # index_mix=[]
                        index = mem_y == ii
                        index_dif = all_y != ii  # .float()#

                        if index.sum() > 0:
                            # for tt in range(repeat_num-1):
                            #     index_mix.append(mem_y==ii+1)
                            # pdb.set_trace()
                            random_id = torch.from_numpy(
                                np.random.choice(index_dif.sum().cpu().item(), index.sum().cpu().item(),
                                                 replace=True)).cuda()  # torch.randperm(index.sum())

                            mask_background1 = ((mask_object[index]).float() + (
                                ~mask_object_2[index_dif][random_id]).float() == 2)
                            mask_background2 = mask_object[index].float() - mask_background1.float()
                            # pdb.set_trace()
                            mem_dif[index] = mem_x[:num][index] * (
                                        1 - mask_object[index].float() + mask_background2.float()) + all_x[index_dif][
                                                 random_id] * mask_background1

                        # pdb.set_trace()
                    # mem_y=mem_y.repeat(repeat_num)
                    teacher_temperature = 0.1
                    student_temperature = 0.07
                    # mem_x = mem_x.requires_grad_()
                    with  torch.no_grad():
                        hidden_normal = normalize(Basic_model.simclr(Basic_model.f_train(mem_x)))
                        hidden_same_normal = normalize(Basic_model.simclr(Basic_model.f_train(mem_x)))
                        hidden_same_batch = torch.matmul(hidden_same_normal, hidden_normal.t()) / teacher_temperature

                        relation_sam = F.softmax(hidden_same_batch, dim=0)
                    mem_dif = mem_dif.requires_grad_()
                    hidden_dif_normal = normalize(Basic_model.simclr(Basic_model.f_train(mem_dif)))
                    hidden_dif_batch = torch.matmul(hidden_dif_normal, hidden_normal.t()) / student_temperature
                    relation_dif = F.log_softmax(hidden_dif_batch, dim=0)
                    loss_dif = F.kl_div(relation_dif, relation_sam,
                                        reduction='batchmean')  # -(relation_sam * torch.nn.functional.log_softmax(relation_dif, 1)).sum()/relation_dif.shape[0]

                    mem_y = mem_y.reshape(-1)

                    mem_x = mem_x.requires_grad_()

                    images1_r = torch.cat([Rotation(mem_x, r) for r in range(1)])
                    images2_r = torch.cat([Rotation(mem_x, r) for r in range(1)])

                    images_pair_r = torch.cat([images1_r, simclr_aug(images2_r)], dim=0)

                    u = Basic_model(images_pair_r, is_simclr=True)

                    images_out_r = u

                    simclr_r = normalize(images_out_r)

                    rot_sim_labels_r = torch.cat([mem_y.cuda() + 100 * i for i in range(1)], dim=0)

                    sim_matrix_r = get_similarity_matrix(simclr_r)

                    loss_sim_r = Supervised_NT_xent(sim_matrix_r, labels=rot_sim_labels_r, temperature=0.07)

                    lo1 = 1 * loss_sim_r + 1*loss_sim1

                    hidden = Basic_model.f_train(mem_x)

                  #  if len(class_holder) > len(new_class_holder):
                   #     T=2

                    #    loss_kd= 1.0*((hidden-logits)**2).mean()+2.0*((prev_hiddens-cur_hiddens)**2).mean()
                    #else:
                     #   loss_kd = 1.0*((hidden-logits)**2).mean()
                   # if len(class_holder) > len(new_class_holder):
                    #   import pdb
                     #  pdb.set_trace()
                    mem_x = RandomFlip(mem_x, flip_num)
                    mem_y = mem_y.repeat(flip_num)
                    y_pred = Basic_model.forward(mem_x)
                    y_pred_hidden=Basic_model.f_train(mem_x)
                    loss_old=0
                    #if i >0:
                     #   pdb.set_trace()
                     #   prev_logits= Previous_model.linear(y_pred_hidden)
                      #  loss_old=F.mse_loss(prev_logits,y_pred[:,:-2])

                    y_pred_new = y_pred
                    loss_only=0


                    # category_matrix_new = torch.zeros(logits_new.shape)
                    exp_new = torch.exp(y_pred_new)
                 #   positive_matrix = torch.ones_like(exp_new)
                #    Negative_matrix = torch.ones_like(exp_new)

                   # for i_v in range(int(exp_new.shape[0])):
                        # category_matrix_new[i_v][mem_y[i_v]] = 1
                  #      Negative_matrix[i_v][:-len(new_class_holder)] = 1 / (torch.exp(-NT[:-len(new_class_holder)] - 0.1))
                    #    if mem_y[i_v]  in new_class_holder:
                     #       continue
                            #1 / NT[:-len(new_class_holder)]
                     #   else:

                      #      positive_matrix[i_v][mem_y[i_v]] = 1#1/(NT[mem_y[i_v]])
                       # if mem_y[i_v] in new_class_holder:
                            # Negative_matrix[i_v][:-len(new_class_holder)] = 1 / NT[:-len(new_class_holder)]
                        #    positive_matrix[i_v][mem_y[i_v]] = 1  # 1 / (NT[mem_y[i_v]])
                        #else:

                         #   positive_matrix[i_v][mem_y[i_v]] = 1 / (torch.exp(-ANT[mem_y[i_v]] - 0.1))
                    # pdb.set_trace()
                 #   if task_id > 0:
                  #      print(Negative_matrix)
                    exp_new = exp_new# * Negative_matrix
                    # pdb.set_trace()
                    exp_new_sum = torch.sum(exp_new, dim=1)
                    logits_new = (exp_new / exp_new_sum.unsqueeze(1))
                    category_matrix_new = torch.zeros(logits_new.shape)
                    for i_v in range(int(logits_new.shape[0])):
                        category_matrix_new[i_v][mem_y[i_v]] = 1
                    #     positive_matrix[i_v][mem_y[i_v]]=0

                    # if task_id>0:
                    #    import pdb
                    #   pdb.set_trace()

                    #  import pdb
                    # pdb.set_trace()
                    positive_prob = torch.zeros(logits_new.shape)
                    false_prob = deepcopy(logits_new.detach())
                    for i_t in range(int(logits_new.shape[0])):
                        false_prob[i_t][mem_y[i_t]] = 0
                        positive_prob[i_t][mem_y[i_t]] = logits_new[i_t][mem_y[i_t]].detach()
                    if negative_logits_sum is None:
                        negative_logits_sum = torch.sum(false_prob, dim=0)
                        positive_logits_sum = torch.sum(positive_prob, dim=0)
                        if i == 0:
                            Category_sum = torch.sum(category_matrix_new, dim=0)
                        else:
                            Category_sum += torch.sum(category_matrix_new, dim=0)  # .cuda()

                        category_sum = torch.sum(category_matrix_new, dim=0)
                    else:
                        Category_sum += torch.sum(category_matrix_new, dim=0)  # .cuda()
                        negative_logits_sum += torch.sum(false_prob, dim=0)
                        positive_logits_sum += torch.sum(positive_prob, dim=0)
                        category_sum += torch.sum(category_matrix_new, dim=0)
                    if negative_logits_SUM is None:
                        negative_logits_SUM = torch.sum(false_prob, dim=0).cuda()
                        positive_logits_SUM = torch.sum(positive_prob, dim=0).cuda()
                    else:
                        negative_logits_SUM += torch.sum(false_prob, dim=0).cuda()
                        positive_logits_SUM += torch.sum(positive_prob, dim=0).cuda()

                    sum_num += int(logits_new.shape[0])
                    if batch_idx < 5:
                        ANT = torch.ones(len(class_holder))
                        NT = torch.ones(len(class_holder))
                    else:
                        #   pdb.set_trace()
                        ANT = (Category_sum.cuda() - positive_logits_SUM).cuda()/negative_logits_SUM.cuda() #/ (Category_sum.cuda() - positive_logits_SUM).cuda()
                        NT = negative_logits_sum.cuda() / (category_sum - positive_logits_sum).cuda()

                    ttt = torch.zeros(logits_new.shape)
                    for qqq in range(mem_y.shape[0]):
                        if mem_y[qqq]>=len(ANT):
                            ttt[qqq][mem_y[qqq]] = 1
                        else:
                            ttt[qqq][mem_y[qqq]] = 2 / (1+torch.exp(1-(ANT[mem_y[qqq]])))
                       # if mem_y[qqq] in new_class_holder:
                        #    ttt[qqq][mem_y[qqq]] = 1  # (ANT[mem_y[qqq]])
                        #else:
                         #   ttt[qqq][mem_y[qqq]] = 1 / (1+torch.exp(-ANT[mem_y[qqq]] - 1))

                   # logits_new==logits_new_p
                    #import pdb
                    #pdb.set_trace()

                   # if len(class_holder) > len(new_class_holder):


                   # identity_matrix_new=torch.ones(logits_new.shape)
                   # logits_=F.softmax(y_pred_new,dim=1)
                    #if batch_idx>0:
                 #   ANT=negative_logits_SUM.cuda() / (Category_sum.cuda() - positive_logits_SUM).cuda()

                    #.detach()

                 #   aaa=F.nll_loss(torch.log(logits_new),mem_y)
                   # if batch_idx>3:
                    #    pdb.set_trace()
                    #+0.05#1+torch.exp(-mem_y[qqq].float())
                   # print(ttt)
                    loss_n=-torch.sum(torch.log(logits_new)*ttt.cuda())/mem_y.shape[0]
                    loss =2* loss_n + 1 * F.cross_entropy(
                        pred_y_new, y_new)#+loss_balance#+2*loss_sim_r+loss_sim1#+loss_dif#+loss_old#+2*loss_only

                else:

                    x = RandomFlip(x, flip_num)
                    y = y.repeat(flip_num)
                    x = x.requires_grad_()
                    hidden_pred = Basic_model.f_train(simclr_aug(x))
                    pred_y = Basic_model.linear(hidden_pred)

                    t = num_class_per_task#len(new_class_holder)
                    pred_y_new = pred_y[:, -t:]
                    min_y = num_class_per_task*i#min(new_class_holder)
                    y_new = y - min_y

                    loss = F.cross_entropy(pred_y_new, y_new)
                copy_x = ori_x
                copy_y = ori_y.unsqueeze(1)
                copy_hidden = Basic_model.f_train(copy_x).detach()
                with amp.scale_loss(loss, Optimizer) as scaled_loss:
                    scaled_loss.backward()
             #   loss.backward()
                Optimizer.step()


                buffero.add_reservoir(x=copy_x.detach(), y=copy_y.squeeze(1).detach(), logits=copy_hidden.float().detach(),
                                      t=i)
        weights_path = 'weights_pre.pt'
        torch.save(Basic_model.state_dict(), weights_path)
        Previous_model = deepcopy(Basic_model)
        print(len(class_holder))

   #     import pdb
    #    pdb.set_trace()
        #if task_id>0:
        print(negative_logits_SUM.cuda(),(Category_sum.cuda()-positive_logits_SUM).cuda(),category_sum,sum_num,negative_logits_SUM.cuda()/(Category_sum.cuda()-positive_logits_SUM).cuda())
        for j in range(i + 1):
            print("ori", rank[j].item())
            a = test_model(Loder[rank[j].item()]['test'], j)

            if j == i:
                Max_acc.append(a)
            if a > Max_acc[j]:
                Max_acc[j] = a
     #   if task_id>=1:
      #      import pdb
       #     pdb.set_trace()


    import pdb
    class_acc=[]
    for j in range(100):

        acc = test_model(Loder2[j]['test'], j)
        class_acc.append(acc)

    print(class_acc,'!')
    pdb.set_trace()

    print('=' * 100)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  ' + os.environ["CUDA_VISIBLE_DEVICES"])
    print('=' * 100)
    import pdb
    test_loss = 0
    correct = 0
    num = 0
    for batch_idx, (data, target) in enumerate(test_loder):
        data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        Basic_model.eval()
        pred = F.softmax(Basic_model.forward(data),dim=1)
        Pred = pred.data.max(1, keepdim=True)[1]
        num += data.size()[0]

        #    print("final", Pred, target.data.view_as(Pred))
        # print(target,"True",pred)

        correct += Pred.eq(target.data.view_as(Pred)).cpu().sum()

    test_accuracy = 100. * correct / num  # len(data_loader.dataset)
    print(
        'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
            .format(
            test_loss, correct, num,
            100. * correct / num, ))
    print(Max_acc)
    import pdb

    pdb.set_trace()
    n = 0
    sum = 0
    for m in range(len(Max_acc)):
        sum += Max_acc[m]
        n += 1
    print(sum / n)


