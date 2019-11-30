import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from classifier import simpleClassifier
from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet
from learner import Learner
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric

if __name__ == '__main__':
    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', [])
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=30)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./save/proto-1')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--folds', type=int, default=2)
    args = parser.parse_args()
    pprint(vars(args))

    #set_gpu(args.gpu)
    ensure_path(args.save_path)
    testset = MiniImageNet('test')
    test_sampler = CategoriesSampler(testset.label,
                                args.batch, args.way, args.folds * args.shot + args.query)
    test_loader = DataLoader(testset, batch_sampler=test_sampler,
                        num_workers=args.num_workers, pin_memory=True)

    trainset = MiniImageNet('train')
    train_sampler = CategoriesSampler(trainset.label, 100,
                                      args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True)

    valset = MiniImageNet('val')
    val_sampler = CategoriesSampler(valset.label, 400,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    #model = Convnet().cuda()
    #TODO get out_dim of proto
    classifier = simpleClassifier(10, args.way)
    model = Learner(config, 3, 84).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))


    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()
    update_lr = .01
    for epoch in range(1, args.max_epoch + 1):
        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()
        for j, test_batch in enumerate(test_loader):
            test_data, _ = [_.cuda() for _ in batch]
            test_shot, test_query = test_data[:p], test_data[p:]
            for i, batch in enumerate(train_loader, 1):
                data, _ = [_.cuda() for _ in batch]
                p = args.shot * args.train_way
                train_shot, train_query = data[:p], data[p:]
                proto = model(train_shot, vars=None, bn_training=True)
                unsup_logits = classifier(proto)

                soft_labels = F.softmax(unsup_logits, dim=1)
                soft_labels = soft_labels / soft_labels.sum(dim=0)
                proto = torch.mm(soft_labels.permute((1, 0)), proto)

                label = torch.arange(args.train_way).repeat(args.query)
                label = label.type(torch.cuda.LongTensor)

                logits = euclidean_metric(model(train_query), proto)
                loss = F.cross_entropy(logits, label)
                grad = torch.autograd.grad(loss, model.parameters())

                fast_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grad, model.parameters())))

            acc = count_acc(logits, label)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                  .format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            proto = None;
            logits = None;
            loss = None

        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()

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
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-last')

        if epoch % args.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))