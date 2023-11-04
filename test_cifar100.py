import ipaddress
import sys, argparse
import numpy as np
import torch
from torch.nn.functional import relu, avg_pool2d
from buffer import Buffer
# import utils
import datetime
from torch.nn.functional import relu
import torch.nn as nn
import torch.nn.functional as F
from CSL import tao as TL
from CSL import classifier as C
from CSL.utils import normalize
import torch.optim.lr_scheduler as lr_scheduler
from CSL.shedular import GradualWarmupScheduler
import torchvision.transforms as transforms
import torchvision
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
from utils import *
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
oop = 4
print('=' * 100)
print('Arguments =')
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print('=' * 100)
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  ' + os.environ["CUDA_VISIBLE_DEVICES"])
print('=' * 100)
########################################################################################################################
print('Load data...')
num_class_per_task=10
data, taskcla, inputsize, Loder, test_loder = dataloader.get_cifar100_10(seed=args.seed)
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

    color_gray = TL.RandomColorGrayLayer(p=0.2).cuda()
    resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=[32, 32, 3]).cuda()
    simclr_aug = transform = torch.nn.Sequential(color_gray, resize_crop,
        )

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
            print('Adding dimension !')
            Basic_model.change_output_dim(num_class_per_task)
            Category_sum = torch.cat((Category_sum, torch.zeros(num_class_per_task)))
            negative_logits_SUM = torch.cat(
                (negative_logits_SUM, torch.zeros(num_class_per_task).cuda()))
            positive_logits_SUM = torch.cat(
                (positive_logits_SUM, torch.zeros(num_class_per_task).cuda()))

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

                Optimizer.zero_grad()
                # if args.cuda:
                x, y = x.cuda(), y.cuda()
                ori_x = x.detach()
                ori_y = y.detach()
                x = x.requires_grad_()

                if batch_idx==0&task_id==0:
                    cur_x, cur_y,_ = torch.zeros(1),torch.zeros(1),torch.zeros(1)#,None,None#buffero.onlysample(22, task=task_id)
                else:

                    cur_x, cur_y, _,_ = buffero.onlysample(22, task=task_id)
                if len(cur_x.shape) > 3:
                    x = torch.cat((x, cur_x), dim=0)
                    y = torch.cat((y, cur_y))

                if not buffero.is_empty():

                    buffer_batch_size = 64

                    # x = x.requires_grad_()
                    x = RandomFlip(x, flip_num)
                    y = y.repeat(flip_num)
                    x = x.requires_grad_()
                    hidden_pred = Basic_model.f_train(simclr_aug(x))
                    pred_y = Basic_model.linear(hidden_pred)
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
                    #balance sampling
                    mem_x, mem_y, logits, bt = buffero.sample(int(buffer_batch_size*(1-rate))*1, exclude_task=task_id)

                    index_x=ori_x
                    index_y=ori_y
                    if len(cur_x.shape) > 3:
                        index_x = torch.cat((index_x, cur_x), dim=0)
                        index_y = torch.cat((index_y, cur_y))

                    all_x = torch.cat((mem_x, index_x), dim=0)
                    all_y = torch.cat((mem_y, index_y))

                    mem_x = torch.cat((mem_x[:int(buffer_batch_size*(1-rate))],index_x[:int(buffer_batch_size*rate)]),dim=0)
                    mem_y = torch.cat((mem_y[:int(buffer_batch_size*(1-rate))],index_y[:int(buffer_batch_size*rate)]))
                    logits = torch.cat((logits[:int(buffer_batch_size*(1-rate))],Basic_model.f_train(index_x[:int(buffer_batch_size*rate)])),dim=0)
                    index = torch.randperm(mem_y.size()[0])
                    mem_x=mem_x[index][:]
                    mem_y=mem_y[index][:]
                    logits=logits[index][:]

                    mem_y = mem_y.reshape(-1)

                    mem_x = mem_x.requires_grad_()
                    hidden = Basic_model.f_train(mem_x)
                    mem_x = RandomFlip(mem_x, flip_num)
                    mem_y = mem_y.repeat(flip_num)
                    y_pred = Basic_model.forward(mem_x)
                    y_pred_hidden=Basic_model.f_train(mem_x)
                    #Calculating Rate
                    y_pred_new = y_pred
                    loss_only=0
                    exp_new = torch.exp(y_pred_new)

                    exp_new = exp_new# * Negative_matrix
                    exp_new_sum = torch.sum(exp_new, dim=1)
                    logits_new = (exp_new / exp_new_sum.unsqueeze(1))
                    category_matrix_new = torch.zeros(logits_new.shape)
                    for i_v in range(int(logits_new.shape[0])):
                        category_matrix_new[i_v][mem_y[i_v]] = 1

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
        print('Calculating Performance')
        for j in range(i + 1):
            print("ori", rank[j].item())
            a = test_model(Loder[rank[j].item()]['test'], j)
            if j == i:
                Max_acc.append(a)
            if a > Max_acc[j]:
                Max_acc[j] = a
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


        correct += Pred.eq(target.data.view_as(Pred)).cpu().sum()

    test_accuracy = 100. * correct / num  # len(data_loader.dataset)
    print(
        'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
            .format(
            test_loss, correct, num,
            100. * correct / num, ))



