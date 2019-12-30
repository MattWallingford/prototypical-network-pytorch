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
    df[~msk].to_csv(unsup_path, index = False)
    df[msk].to_csv(sup_path, index = False)


def splitIm(splits):
    setname = 'train'
    csv_path = osp.join(ROOT_PATH, setname + '.csv')
    df = pd.read_csv(csv_path)
    df_np = np.array(df)
    np.random.shuffle(df_np)
    s_df = pd.DataFrame(df_np)
    increment = int(len(df)/splits)
    for i in range(splits-1):
        d_i = s_df.iloc[increment*i: increment*(i+1)]
        d_i = d_i.sort_values(by=1)
        d_i.to_csv(osp.join(ROOT_PATH, setname + str(i) + '.csv'), index = False)
    d_i = s_df.iloc[increment*(splits-1):]
    d_i = d_i.sort_values(by=1)
    d_i.to_csv(osp.join(ROOT_PATH, setname + str(splits-1) + '.csv'), index = False)

def splitIm2(splits):
    setname = 'train'
    csv_path = osp.join(ROOT_PATH, setname + '.csv')
    df = pd.read_csv(csv_path)
    df_np = np.array(df)
    np.random.shuffle(df_np)
    s_df = pd.DataFrame(df_np)
    increment = int(len(df)/splits)
    s_df = pd.DataFrame(df_np)
    s_df = s_df.sort_values(by=1)
    for i in range(splits):
        d_i = s_df[i::splits]
        d_i.to_csv(osp.join(ROOT_PATH, setname + str(i) + '.csv'), index=False)

def splitSS2(classes):
    setname = 'train'
    csv_path = osp.join(ROOT_PATH, setname + '.csv')
    df = pd.read_csv(csv_path)
    unsup = df.iloc[600*classes:, :]
    sup = df.iloc[:600*classes, :]
    unsup.to_csv(osp.join(ROOT_PATH, 'unsup2' + '.csv'), index = False)
    sup.to_csv(osp.join(ROOT_PATH, 'sup2' + '.csv'), index = False)



class SplitMiniImageNet(Dataset):
    def __init__(self, split):
        setname = 'train'
        self.data = []
        self.label = []
        lb = -1
        for i in range(split):
            path = osp.join(ROOT_PATH, setname + str(i) + '.csv')
            line = [x.strip() for x in open(path, 'r').readlines()][1:]
            self.wnids = []
            for l in line:
                name, wnid = l.split(',')
                path = osp.join(ROOT_PATH, 'images', name)
                if wnid not in self.wnids:
                    lb += 1
                    self.wnids.append(wnid)
                self.data.append(path)
                self.label.append(lb)

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
        #print(label)
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


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
        self.wnid_lab = dict()

        for l in slines:
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                lb += 1
                self.wnid_lab[wnid] = lb
                self.wnids.append(wnid)
            sdata.append(path)
            slabel.append(lb)

        for l in ulines:
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH, 'images', name)
            # if wnid not in self.wnids:
            #     self.wnids.append(wnid)
            #     lb += 1
            lb = self.wnid_lab[wnid]
            udata.append(path)
            ulabel.append(lb)

        self.sdata = sdata
        self.slabel = slabel
        self.udata = udata
        self.ulabel = np.array(ulabel)

        self.m_ind = []
        for i in range(max(self.ulabel) + 1):
            ind = np.argwhere(self.ulabel == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

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

        #u_index = random.randint(0, len(self.udata) - 1)
        path, label = self.sdata[i], self.slabel[i]
        idxs = self.m_ind[label]
        u_index = np.random.choice(idxs)
        upath, ulabel = self.udata[u_index], self.ulabel[u_index]

        image = self.transform(Image.open(path).convert('RGB'))
        uimage = self.transform(Image.open(upath).convert('RGB'))
        return image, uimage


class SS2MiniImageNet(Dataset):
    def __init__(self):
        setname = 'sup2'
        sup_path = osp.join(ROOT_PATH, setname + '.csv')
        setname = 'unsup2'
        usup_path = osp.join(ROOT_PATH, setname + '.csv')
        slines = [x.strip() for x in open(sup_path, 'r').readlines()][1:]
        ulines = [x.strip() for x in open(usup_path, 'r').readlines()][1:]

        sdata = []
        slabel = []
        udata = []
        ulabel = []
        lb = -1

        self.wnids = []
        self.wnid_lab = dict()

        for l in slines:
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                lb += 1
                self.wnid_lab[wnid] = lb
                self.wnids.append(wnid)
            sdata.append(path)
            slabel.append(lb)
        lb += 1
        for l in ulines:
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH, 'images', name)
            #print(wnid, lb)
            udata.append(path)
            ulabel.append(lb)

        self.sdata = sdata
        self.slabel = slabel
        self.udata = udata
        self.ulabel = np.array(ulabel)

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
        path, label = self.sdata[i], self.slabel[i]
        u_index = np.random.choice(len(self.ulabel))
        upath, ulabel = self.udata[u_index], self.ulabel[u_index]

        image = self.transform(Image.open(path).convert('RGB'))
        uimage = self.transform(Image.open(upath).convert('RGB'))
        return image, uimage