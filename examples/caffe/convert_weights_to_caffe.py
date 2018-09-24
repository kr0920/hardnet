import numpy as np
# Make sure that caffe is on the python path:
caffe_root = '/home/ubuntu/dev/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe


import sys
import argparse
import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import os
import sys
sys.path.insert(0, '/home/ubuntu/dev/opencv-3.1/build/lib')
import cv2
import math
import numpy as np
from copy import deepcopy
import random
import time
import numpy as np
import glob
import os

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim = 1) + self.eps
        x= x / norm.expand_as(x)
        return x

class HardNet(nn.Module):
    """HardNet model definition
    """
    def __init__(self):
        super(HardNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8,stride=2),
            nn.Tanh(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=6),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.Tanh()
        )
           
        #self.features.apply(weights_init)

    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        x=self.classifier(x)
        return L2Norm()(x)
gg = {}
gg['counter'] = 1
def copy_weights(m,):
    if isinstance(m, nn.Conv2d):
        counter = gg['counter']
        l_name = 'conv' + str(counter)
        print( l_name,  m.weight.data.cpu().numpy().shape)
        net.params[l_name][0].data[:] = m.weight.data.cpu().numpy();
        gg['counter']+=1
        #try:
        #    net.params[l_name][1].data[:] = m.bias.data.cpu().numpy();
        #except:
        #    pass
    if isinstance(m, nn.BatchNorm2d):
        counter = gg['counter']
        l_name = 'conv' + str(counter) + '_BN'
        print( l_name)
        net.params[l_name][0].data[:] = m.running_mean.cpu().numpy();
        net.params[l_name][1].data[:] = m.running_var.cpu().numpy();
        net.params[l_name][2].data[:] = 1.
        gg['counter'] += 1
    #if isinstance(m,nn.Linear):
    #    l_name='fc'
    #    print(l_name)
    #    net.params[l_name][0].data[:]=m.weight.data.cpu.numpy()
    #    net.params[l_name][1].data[:]=m.bias.data.cpu().numpy()
    #if isinstance(m,nn.MaxPool2d):
    #    l_name='pool1'
    #    print(net.params)
    #    print(net.blobs)
    #    net.params[l_name][0].data[:]=m.kernel_size;
    #    net.params[l_name][1].data[:]=m.stride;
    #    print(l_name,m.stride,m.kernel_size)
model = HardNet()

model.cuda()
    
mws = [
"./checkpoint_8_withoutpool.pth"
#"./checkpoint_tfeat_001.pth"
#"../../pretrained/train_liberty_with_aug/checkpoint_liberty_with_aug.pth",
#"../../pretrained/train_yosemite/checkpoint_yosemite_no_aug.pth",
#"../../pretrained/train_liberty/checkpoint_liberty_no_aug.pth",
#"../../pretrained/train_notredame_with_aug/checkpoint_notredame_with_aug.pth",
#"../../pretrained/train_notredame/checkpoint_notredame_no_aug.pth",
#"../../pretrained/train_yosemite_with_aug/checkpoint_yosemite_with_aug.pth",
#"../../pretrained/pretrained_all_datasets/HardNet++.pth"
#"../../pretrained/6Brown/hardnetBr6.pth"
    ]
#import pdb
#pdb.set_trace()
for model_weights in mws:
    gg['counter'] = 1
    print(model_weights)
    checkpoint = torch.load(model_weights)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    caffe.set_mode_cpu()
    net = None
    net = caffe.Net('TFeat_withoutpool.prototxt', caffe.TEST)
    model.features.apply(copy_weights)
    caffe_weights_fname = model_weights.split('/')[-1].replace('.pth', '.caffemodel')
    net.save(caffe_weights_fname)

    
