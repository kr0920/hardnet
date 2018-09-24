#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import os
import cv2
import math
import numpy as np

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
            nn.Conv2d(1, 32, kernel_size=7),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=6),
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.Tanh()
        )

    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        x = self.classifier(x)
        return L2Norm()(x)
    

if __name__ == '__main__':
    DO_CUDA = True
    try:
          input_img_fname = sys.argv[1]
          output_fname = sys.argv[2]
          if len(sys.argv) > 3:
              DO_CUDA = sys.argv[3] != 'cpu'
    except:
          print("Wrong input format. Try ./extract_hardnet_desc_from_hpatches_file.py imgs/ref.png out.txt gpu")
          sys.exit(1)
    #model_weights = '../pretrained/train_liberty_with_aug/checkpoint_liberty_with_aug.pth'
    #model_weights='/home/kangrong/HardNet/hardnet/code/data/models/withoutpool/liberty_train_withoutpool/_liberty_min_as_fliprot/checkpoint_8.pth'
    #model_weights='/home/kangrong/HardNet/hardnet/code/data/models/tfeat_whole/liberty_train_tfeat_whole/_liberty_min_as_fliprot/checkpoint_0.pth'
    model_weights='/unsullied/sharefs/kangrong/home/hardnet/data/models/model_HPatches_HardTFeat_a_lr01_trimar/all_min_as/checkpoint_8.pth'
    model = HardNet()
    checkpoint = torch.load(model_weights,map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    #model=torch.load(model_weights)
    model.eval()
    if DO_CUDA:
        model.cuda()
        print('Extracting on GPU')
    else:
        print('Extracting on CPU')
        model = model.cpu()
    image = cv2.imread(input_img_fname,0)
    h,w = image.shape
    print(h,w)

    n_patches =  int(h/w)

    print('Amount of patches: {}'.format(n_patches))

    t = time.time()
    patches = np.ndarray((n_patches, 1, 32, 32), dtype=np.float32)
    for i in range(n_patches):
        patch =  image[i*(w): (i+1)*(w), 0:w]
        patches[i,0,:,:] = cv2.resize(patch,(32,32)) / 255.
    patches -= 0.443728476019
    patches /= 0.20197947209
    bs = 128
    outs = []
    n_batches = int(n_patches / bs) + 1
    t = time.time()
    descriptors_for_net = np.zeros((len(patches), 128))
    for i in range(0, len(patches), bs):
        data_a = patches[i: i + bs, :, :, :].astype(np.float32)
        data_a = torch.from_numpy(data_a)
        if DO_CUDA:
            data_a = data_a.cuda()
        data_a = Variable(data_a)
        # compute output
        with torch.no_grad():
            out_a = model(data_a)
        descriptors_for_net[i: i + bs,:] = out_a.data.cpu().numpy().reshape(-1, 128)
    print(descriptors_for_net.shape)
    assert n_patches == descriptors_for_net.shape[0]
    et  = time.time() - t
    print('processing', et, et/float(n_patches), ' per patch')
    np.savetxt(output_fname, descriptors_for_net, delimiter=' ', fmt='%10.5f')
