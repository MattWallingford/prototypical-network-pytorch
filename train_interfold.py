import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from options import TrainOptions

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric, log_settings
from extensions import inter_fold

if __name__ == '__main__':
    parser = TrainOptions()
    args = parser.parse_args()
    pprint(vars(args))
    log_settings(args)
    set_gpu(args.gpu)
    ensure_path(args.save_path)
    writer = SummaryWriter()
    #noise = torch.distributions.Normal(loc=0, scale=.02)
    trainset = MiniImageNet('train')
    train_sampler = CategoriesSampler(trainset.label, 100,
                                      args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True)

    valset = MiniImageNet('val')
    val_sampler = CategoriesSampler(valset.label, 400,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=True)

    model = torch.nn.DataParallel(Convnet())
    if args.load is not 'na':
        model.load_state_dict(torch.load(args.load))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))


    timer = Timer()
    s_label = torch.arange(args.train_way).repeat(args.shot).view(args.shot * args.train_way)
    s_onehot = torch.zeros(s_label.size(0), 20)
    s_onehot = s_onehot.scatter_(1, s_label.unsqueeze(dim=1), 1).cuda()
    for epoch in range(args.start_epoch, args.max_epoch + 1):
        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.train_way
            data_shot, data_query = data[:p], data[p:]

            proto = inter_fold(model, args, data_shot)
            #proto = model(data_shot)
            proto = proto.reshape(int(args.shot), args.train_way, -1).mean(dim=0)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            # proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)
            label = torch.arange(args.num_slices*args.train_way).repeat(int(args.query/args.num_slices))
            label = label.type(torch.cuda.LongTensor)
            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
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
