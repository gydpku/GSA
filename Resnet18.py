# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from typing import List
#from modified_linear import *
from torch.nn import functional as F




def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.num_classes=num_classes
        self.linear = nn.Linear(nf * 8 * block.expansion, self.num_classes)#nn.utils.weight_norm(nn.Linear(nf * 8 * block.expansion, self.num_classes))
      #  torch.nn.init.xavier_uniform(self.linear.weight)
        self.out_dim = nf * 8 * block.expansion
        self.drop = nn.Dropout(p=0.2)
       # self.drop2 = nn.Dropout(p=0.3)
        self.simclr=nn.Linear(nf * 8 * block.expansion, 128)
        self.simclr2 = nn.Linear(nf * 8 * block.expansion, 128)
        self._features = nn.Sequential(self.conv1,
                                       self.bn1,
                                       self.layer1,
                                       self.layer2,
                                       self.layer3,
                                       self.layer4
                                       )
        self.classifier = self.linear
    def f_train_feat_map(self, x: torch.Tensor,mask=None) -> torch.Tensor:

        out = relu(self.bn1(self.conv1(x)))
       # pdb.set_trace()
        out = self.layer1(out)#,None)#,mask)  # 64, 32, 32
        out = self.layer2(out)#,None)#,mask)  # 128, 16, 16
        out = self.layer3(out)#,None)  # 256, 8, 8
       # pdb.set_trace()

        #out = self.layer4.BasicBlock0
        out = self.layer4(out)#,None)  # 512, 4, 4
        #out = avg_pool2d(out, out.shape[2])  # 512, 1, 1
        #out = out.view(out.size(0), -1)  # 512
        return out
    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def f_train(self, x: torch.Tensor) -> torch.Tensor:

        out = relu(self.bn1(self.conv1(x)))
       # out = self.drop(out)
        out = self.layer1(out)  # 64, 32, 32
       # out = self.drop(out)
        out = self.layer2(out)  # 128, 16, 16
       # out = self.drop(out)
        out = self.layer3(out)  # 256, 8, 8
       # out = self.drop(out)
        out = self.layer4(out)  # 512, 4, 4
       # out = self.drop(out)
        out = avg_pool2d(out, out.shape[2])  # 512, 1, 1

        out = out.view(out.size(0), -1)  # 512
        return out
    def f_inter(self, x: torch.Tensor) -> torch.Tensor:

        out = relu(self.bn1(self.conv1(x)),inplace=True)
        out = self.layer1(out)  # 64, 32, 32
        out = self.layer2(out)  # 128, 16, 16
          # 512, 1, 1
        out = self.layer3(out)  # 256, 8, 8
        out = self.layer4(out)
        out = out.view(out.size(0), -1)  # 512
        return out

    def forward(self, x: torch.Tensor, is_simclr=False,is_simclr2=False,is_drop=False) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        ''' 
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  # 64, 32, 32
        out = self.layer2(out)  # 128, 16, 16
        out = self.layer3(out)  # 256, 8, 8
        '''
        out = self.f_train(x)
        #out = self.drop(out)
        ''' 
        out = self.layer4(out)  # 512, 4, 4
        out = avg_pool2d(out, out.shape[2]) # 512, 1, 1
        out = out.view(out.size(0), -1)  # 512
        '''
        if is_simclr:
          #  out=self.drop2(out)
            out = self.simclr(out)
        elif is_drop:
            #out=nn.dropout
            out=self.drop(out)
            out = self.linear(out)
          #  out=out.detach()
           # out = self.drop(out)
        else:
           # out=out / (out.norm(dim=1, keepdim=True) + 1e-8)
           # out = self.drop(out)
            out = self.linear(out)

        return out

    def change_output_dim(self, new_dim, second_iter=False):
        self.prev_weights = nn.Linear(self.out_dim, self.num_classes+new_dim)
        in_features = self.out_dim
        out_features = self.num_classes+new_dim
       # old_embedding_weights = self.embedding.weight.data
        # create a new embedding of the new size
        #nn.Embedding(new_vocab_size, embedding_dim)
        # initialize the values for the new embedding. this does random, but you might want to use something like GloVe
        new_weights =nn.Linear(in_features,out_features)#nn.Linear(in_features,out_features,bias=False)
        # as your old values may have been updated, you want to retrieve these updates values
       # new_weights[:old_vocab_size] = old_embedding_weights



        print("in_features:", in_features, "out_features:", out_features)
   ##    self.weight_new =Parameter(torch.Tensor(out_features,in_features))
    #    new_out_features = new_dim
     #   num_new_classes = new_dim - out_features

        #new_fc = SplitCosineLinear(in_features, out_features, num_new_classes)
      #  new_fc= nn.Linear(in_features,out_features)
       # torch.nn.init.xavier_uniform(new_fc.weight)


     #   self.weight_new.data[:self.num_classes] = self.linear.weight.data

        new_weights.weight.data[:self.num_classes] = self.linear.weight.data
        new_weights.bias.data[:self.num_classes] = self.linear.bias.data
      #  self.prev_weights.weight.data[:self.num_classes] = self.linear.weight.data
     #   self.prev_weights.bias.data[:self.num_classes] = self.linear.bias.data
    #    self.linear.weight = self.weight_new#nn.Linear(in_features, out_features)
        #self.linear.weight.data.copy_(new_weights.weight.data)
        #elf.linear.bias.data.copy_(new_weights.bias.data)
        #new_fc.sigma.data = self.fc.sigma.data
        from torch.nn.parameter import Parameter
        self.linear = new_weights.cuda()
        self.linear.requires_grad=True
        self.num_classes = out_features
       # return prev_weights

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        out = self._features(x)
        out = avg_pool2d(out, out.shape[2])
        feat = out.view(out.size(0), -1)
        return feat
    def prev_logit(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        out = self.prev_weights(x)

        return out

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                                               torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)


def resnet18(nclasses: int, nf: int = 64) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf=64)
