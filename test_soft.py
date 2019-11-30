import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet
from utils import pprint, set_gpu, count_acc, Averager, euclidean_metric
from extensions import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--load', default='./save/proto-1/max-acc.pth')
    parser.add_argument('--batch', type=int, default=2000)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=30)
    parser.add_argument('--folds', type=int, default=2)
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)

    dataset = MiniImageNet('test')
    sampler = CategoriesSampler(dataset.label,
                                args.batch, args.way, args.folds * args.shot + args.query)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=8, pin_memory=True)

    model = Convnet().cuda()
    model.load_state_dict(torch.load(args.load))
    model.eval()

    ave_acc = Averager()
    s_label = torch.arange(args.train_way).repeat(args.shot).view(args.shot * args.train_way)
    s_onehot = torch.zeros(s_label.size(0), 20)
    s_onehot = s_onehot.scatter_(1, s_label.unsqueeze(dim=1), 1).cuda()

    for i, batch in enumerate(loader, 1):
        data, _ = [_.cuda() for _ in batch]
        k = args.way * args.shot
        data_shot, meta_support, data_query = data[:k], data[k:2*k], data[2*k:]

        #p = inter_fold(model, args, data_shot)

        x = model(data_shot)
        x = x.reshape(args.shot, args.way, -1).mean(dim=0)
        p = x

        lam = 0.01
        proto = model(meta_support)
        meta_logits = euclidean_metric(proto, p)
        soft_labels = (F.sigmoid(meta_logits, dim=1) + lam * s_onehot) / (1 + lam)
        #soft_labels_norm2 = soft_labels / soft_labels.sum(dim=0)
        proto = torch.mm(soft_labels.permute((1, 0)), proto)

        logits = euclidean_metric(model(data_query), proto)

        label = torch.arange(args.way).repeat(args.query)
        label = label.type(torch.cuda.LongTensor)

        acc = count_acc(logits, label)
        ave_acc.add(acc)
        print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

        x = None;
        p = None;
        logits = None