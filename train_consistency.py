import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np

from mini_imagenet import MiniImageNet, SSMiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=20)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./save/proto-1')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--load', default='na')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--noise', type=int, default=.02)
    parser.add_argument('--meta-size', type=int, default=5)
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    ensure_path(args.save_path)
    writer = SummaryWriter()
    # noise_sample = torch.distributions.normal(loc=0, scale=.02)
    noise = torch.distributions.Normal(loc=0, scale=.02)
    # trainset = MiniImageNet('train')
    # train_sampler = CategoriesSampler(trainset.label, 100,
    #                                   args.train_way, 2*args.shot + args.query)
    # train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
    #                           num_workers=args.num_workers, pin_memory=True)

    ssdata = SSMiniImageNet()
    ss_sampler = CategoriesSampler(ssdata.slabel, 100,
                                      args.train_way, 2*args.shot + args.query)
    ss_loader = DataLoader(dataset=ssdata, batch_sampler=ss_sampler,
                                num_workers=args.num_workers, pin_memory=True)

    valset = MiniImageNet('val')
    val_sampler = CategoriesSampler(valset.label, 400,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=True)

    model = Convnet().cuda()
    if args.load is not 'na':
        print('Loading Model')
        model.load_state_dict(torch.load(args.load))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))


    timer = Timer()
    # s_label = torch.arange(args.train_way).repeat(args.shot).view(args.shot * args.train_way)
    # s_onehot = torch.zeros(s_label.size(0), 20)
    # s_onehot = s_onehot.scatter_(1, s_label.unsqueeze(dim=1), 1).cuda()
    #acc_label = label.type(torch.cuda.LongTensor)
    #q_onehot = torch.zeros(label.size(0), 20)
    #q_onehot = q_onehot.scatter_(1, label.unsqueeze(dim=1), 1).cuda()
    for epoch in range(args.start_epoch, args.max_epoch + 1):
        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(ss_loader, 1):
            data,_, udata, _ = [_.cuda() for _ in batch]
            p = args.shot * args.train_way
            m = args.meta_size * args.train_way
            q = args.query * args.train_way
            data_shot, data_shot2, data_query = data[:p], data[p:2*p], data[2*p:2*p + q]
            meta_proto = model(data_shot2)
            meta_proto = meta_proto.reshape(args.shot, args.train_way, -1).mean(dim=0)
            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)
            #meta_logits = euclidean_metric(proto, meta_proto)
            #lam = .01
            lam = .5
            lam_labs = 1
            #soft_labels = ((F.sigmoid(meta_logits)) + lam*s_onehot) / (1 + lam)
            #soft_labels = F.softmax(meta_logits, dim=1) #* lam + s_onehot) / (1 + lam)
            #soft_labels = soft_labels / soft_labels.sum(dim=0)
            #proto = torch.mm(soft_labels.permute((1, 0)), proto)

            # proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)
            #label = torch.arange(args.train_way).repeat(args.query)
            #label = label.type(torch.cuda.LongTensor)
            logits = euclidean_metric(model(udata), proto)
            logits2 = euclidean_metric(model(udata), meta_proto)
            #loss = F.binary_cross_entropy_with_logits(logits, q_onehot)
            #loss = lam_labs*F.cross_entropy(euclidean_metric(model(data_query), proto), label) + lam_labs*F.cross_entropy(logits2, label) + lam * F.kl_div(F.log_softmax(logits, dim=1), F.softmax(logits2, dim=1))
            loss = F.kl_div(F.log_softmax(logits, dim=1), F.softmax(logits2, dim=1))
            acc = count_acc(logits, label)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                  .format(epoch, i, len(ss_loader), loss.item(), acc))
            tl.add(loss.item())
            ta.add(acc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            proto = None;
            logits = None;
            meta_proto = None
            meta_logits = None
            loss = None
        if epoch % args.save_epoch == 0:
            pass
            #torch.save(soft_labels, args.save_path + "/soft_labels_" + str(epoch))
        tl = tl.item()
        ta = ta.item()

        model.eval()
        vl = Averager()
        va = Averager()
        if epoch % args.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))
            for i, batch in enumerate(val_loader, 1):
                data, _ = [_.cuda() for _ in batch]
                p = args.shot * args.test_way
                data_shot, data_query = data[:p], data[p:]

                proto = model(data_shot)
                proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

                label = torch.arange(args.test_way).repeat(args.query)
                label = label.type(torch.cuda.LongTensor)
                logits = euclidean_metric(model(data_query), proto)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)

                vl.add(loss.item())
                va.add(acc)

                proto = None;
                logits = None;
                loss = None

            vl = vl.item()
            va = va.item()
            writer.add_scalar('Loss/Val', vl, epoch)
            writer.add_scalar('Accuracy/Val', va, epoch)
            print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        writer.add_scalar('Loss/train', tl, epoch)
        writer.add_scalar('Accuracy/train', ta, epoch)

        # save_model('epoch-last')

        if epoch % args.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
