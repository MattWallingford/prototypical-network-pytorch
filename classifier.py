import argparse
import os.path as osp

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet

class simpleClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.classifier = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.classifier(x)
