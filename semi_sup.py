import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np

from mini_imagenet import MiniImageNet, ConcatDataset, SSMiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--uquery', type=int, default=5)
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
    supset = MiniImageNet('sup')
    ssdata = SSMiniImageNet()
    ss_sampler = CategoriesSampler(ssdata.slabel, 100,
                                      args.train_way, 2*args.shot + args.query)
    ss_loader = DataLoader(dataset=ssdata, batch_sampler=ss_sampler,
                                num_workers=args.num_workers, pin_memory=True)
    # train_sampler = CategoriesSampler(supset.label, 100,
    #                                   args.train_way, 2*args.shot + args.query)
    # train_loader = DataLoader(dataset=supset, batch_sampler=train_sampler,
    #                           num_workers=args.num_workers, pin_memory=True)
    #
    # unsup = MiniImageNet('unsup')
    # unsup_sampler = CategoriesSampler(unsup.label, 100,
    #                                   args.train_way, args.uquery)
    # unsup_loader = DataLoader(dataset=unsup, batch_sampler=train_sampler,
    #                           num_workers=args.num_workers, pin_memory=True)


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
    lam = 1
    increment = 1/args.max_epoch
    for epoch in range(args.start_epoch, args.max_epoch + 1):
        lr_scheduler.step()
        #if epoch > 100:
            #lam = 1
        model.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(ss_loader, 1):
            data,udata = batch#[_.cuda() for _ in batch]
            data = data.cuda()
            p = args.shot * args.train_way
            m = args.meta_size * args.train_way
            q = args.query * args.train_way
            data_shot, data_shot2, data_query = data[:p],data[p:2*p], data[2*p:2*p + q]
            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)
            proto2 = model(data_shot2)
            proto2 = proto2.reshape(args.shot, args.train_way, -1).mean(dim=0)
            label = torch.arange(args.train_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)
            logits = euclidean_metric(model(data_query), proto)
            #loss = F.binary_cross_entropy_with_logits(logits, q_onehot)

            #load unsupervised
            q = args.uquery * args.train_way
            udata_query = udata
            udata_query = udata_query.cuda()
            ulogits1 = euclidean_metric(model(udata_query), proto)
            ulogits2 = euclidean_metric(model(udata_query), proto2)
            agree = torch.argmax(ulogits1, dim = 1) == torch.argmax(ulogits2, dim = 1)
            #loss_const = F.kl_div(F.log_softmax(ulogits1, dim=1), F.softmax(ulogits2, dim=1))
            feats =model(data_query)
            loss = F.cross_entropy(euclidean_metric(feats, proto), label) + F.cross_entropy(euclidean_metric(feats, proto2), label) + lam * F.kl_div(F.log_softmax(ulogits1[agree], dim=1), F.softmax(ulogits2[agree], dim=1)) #+ lam_labs*F.cross_entropy(logits2, label) + lam * F.kl_div(F.log_softmax(logits, dim=1), F.softmax(logits2, dim=1))
            acc = count_acc(logits, label)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                  .format(epoch, i, len(ss_loader), loss.item(), acc))
            tl.add(loss.item())
            ta.add(acc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            ulogits1 = None
            ulogits2 = None
            proto2 = None
            proto = None;
            logits = None;
            meta_proto = None
            meta_logits = None
            loss = None
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