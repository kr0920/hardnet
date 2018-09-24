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
import pdb


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x


class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim=1) + self.eps
        x = x / norm.expand_as(x)
        return x


class TFeat(nn.Module):
    """TFeat model definition
    """

    def __init__(self):
        super(TFeat, self).__init__()

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

        # self.features.apply(weights_init)

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(
            -1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        x = self.classifier(x)
        return L2Norm()(x)


if __name__ == '__main__':
    DO_CUDA = True
    try:
        input_img = sys.argv[1]
        # output_fname = sys.argv[2]
        if len(sys.argv) > 2:
            DO_CUDA = sys.argv[2] != 'cpu'
    except:
        print(
            "Wrong input format. Try ./extract_hardnet_desc_from_hpatches_file.py /home/mitc/3dconstruction/ORB_SLAM2-master/Examples/rgbd_dataset_freiburg1_xyz/rgb gpu")
        sys.exit(1)
    # model_weights = '../pretrained/train_liberty_with_aug/checkpoint_liberty_with_aug.pth'
    model_weights = '/unsullied/sharefs/kangrong/home/hardnet/data/models/model_HPatches_HardTFeat_a_lr01_trimar/all_min_as/checkpoint_8.pth'
    # model_weights = '../code/data/models/margin_siftliberty_train_sift/_liberty_min_as_fliprot/checkpoint_9.pth'
    # model_weights='checkpoint_9.pth'
    # pdb.set_trace()
    model = TFeat()
    checkpoint = torch.load(model_weights)  # 权重
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()  # 将模型设置成evaluation模式。仅仅当模型中有Dropout和BatchNorm是才会有影响。
    if DO_CUDA:
        model.cuda()
        print('Extracting on GPU')
    else:
        print('Extracting on CPU')
        model = model.cpu()
    # pdb.set_trace()
    for filename in os.listdir(input_img):
        if os.path.splitext(filename)[1] == '.png':
            liststr = ['../output_hpatchesfast_tfeat_trainedonhpatches/', input_img, '/', filename, '.txt']
            output_filename = ''.join(liststr)
            img_dir = input_img + '/' + filename
            print(img_dir)
            image = cv2.imread(img_dir, 0)
            h, w = image.shape
            print(h, w)

            n_patches = int(h / w)
            print('Amount of patches: {}'.format(n_patches))

            t = time.time()
            patches = np.ndarray((n_patches, 1, 32, 32), dtype=np.float32)
            for i in range(n_patches):
                patch = image[i * (w): (i + 1) * (w), 0:w]
                patches[i, 0, :, :] = cv2.resize(patch, (32, 32)) / 255.
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
                # with torch.no_grad():
                out_a = model(data_a)
                descriptors_for_net[i: i + bs, :] = out_a.data.cpu().numpy().reshape(-1, 128)
            print(descriptors_for_net.shape)
            assert n_patches == descriptors_for_net.shape[0]
            et = time.time() - t
            print('processing', et, et / float(n_patches), ' per patch')
            output_filename = str(output_filename.rsplit('/', 1)[0])
            if not os.path.exists(output_filename):
                os.makedirs(str(output_filename))
            print(output_filename)
            np.savetxt(''.join(output_filename), descriptors_for_net, delimiter=' ', fmt='%10.5f')
