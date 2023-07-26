import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import glob
import os

class DenseNet(torch.nn.Module):
    def __init__(self, inputs, num_hid, outputs):
        super(DenseNet, self).__init__()
        self.layer1 = nn.Linear(inputs, num_hid)
        self.layer1to2 = nn.Linear(inputs, num_hid)
        self.layer1to3 = nn.Linear(inputs, outputs)
        self.layer2 = nn.Linear(num_hid, num_hid)
        self.layer2to3 = nn.Linear(num_hid, outputs)
        self.layer3 = nn.Linear(num_hid, outputs)

        # first 2 use tan
        self.tanh = nn.Tanh()
        # last use sigmoid
        self.sig = nn.Sigmoid()

    def forward(self, input):
        output = self.layer1(input)
        inputTo2 = self.layer1to2(input)
        inputTo3 = self.layer1to3(input)
        output = self.tanh(output)
        self.hid1 = output
        hid1To3 = self.layer2to3(self.hid1)

        output = self.layer2(output)
        output = self.tanh(output + inputTo2)
        self.hid2 = output
        # output
        output = self.layer3(output)
        output = self.sig(output + inputTo3 + hid1To3)
        return output
