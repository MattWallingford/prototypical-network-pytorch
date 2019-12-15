import os.path as osp
from PIL import Image
import numpy as np
import pandas as pd

import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms


ROOT_PATH = './materials/'


class MiniImageNet(Dataset):

    def __init__(self, setname):
        csv_path = osp.join(ROOT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label

        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)




def splitImageNet(sup_ratio):
    setname = 'train'
    csv_path = osp.join(ROOT_PATH, setname + '.csv')
    unsup_path = osp.join(ROOT_PATH,'unsup'  + '.csv')
    sup_path = osp.join(ROOT_PATH,'sup'  + '.csv')
    df = pd.read_csv(csv_path)
    msk = np.random.rand(len(df)) < 0.4
    print('Number of supervised samples: {}'.format(sum(msk)))
    print('Number of unsupervised samples: {}'.format(sum(~msk)))
    df[~msk].to_csv(unsup_path)
    df[msk].to_csv(sup_path)


class SSMiniImageNet(Dataset):
    def __init__(self):
        setname = 'sup'
        sup_path = osp.join(ROOT_PATH, setname + '.csv')
        setname = 'unsup'
        usup_path = osp.join(ROOT_PATH, setname + '.csv')
        slines = [x.strip() for x in open(sup_path, 'r').readlines()][1:]
        ulines = [x.strip() for x in open(usup_path, 'r').readlines()][1:]

        sdata = []
        slabel = []
        udata = []
        ulabel = []
        lb = -1

        self.wnids = []

        for l in slines:
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            sdata.append(path)
            slabel.append(lb)

        for l in ulines:
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            udata.append(path)
            ulabel.append(lb)

        self.sdata = sdata
        self.slabel = slabel
        self.udata = udata
        self.ulabel = ulabel

        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        i = i[0]
        u_index = random.randint(0, len(self.udata) - 1)
        path, label = self.sdata[i], self.slabel[i]
        upath, ulabel = self.udata[u_index], self.ulabel[u_index]
        image = self.transform(Image.open(path).convert('RGB'))
        uimage = self.transform(Image.open(upath).convert('RGB'))
        return image, uimage

