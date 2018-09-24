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

class HardNet(nn.Module):
    """HardNet model definition
    """
    def __init__(self):
        super(HardNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=8, bias = False),
            nn.BatchNorm2d(128, affine=False),

        )

    def forward(self, input):
        x_features = self.features(input)
        x = x_features.view(x_features.size(0), -1)
        #x = self.classifier(x)
        return (x)
if __name__ == '__main__':
    '''
    model_weights='./checkpoint_2_orimodel_test.pth'
    model = HardNet()
    checkpoint = torch.load(model_weights,map_location='cpu')
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    save_dir='./checkpoint_2_orimodel_test_sim.pth'
    torch.save(model,save_dir)
    '''
    save_dir='./checkpoint_2_orimodel_test_sim.pth'
    model_dump=torch.load(save_dir,map_location='cpu')
    from mmdnn.conversion.pytorch.pytorch_parser import PytorchParser
    size=32
    pytorchparser = PytorchParser(model_dump, [1, size, size])
    IR_file = 'HardNet_ori'
    pytorchparser.run(IR_file)


