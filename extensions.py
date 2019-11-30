import torch
import torch.nn.functional as F
from utils import euclidean_metric

def fold(model, args, meta_support, true_labels):
    p = args.shot * args.train_way
    q = args.query * args.train_way
    for j in range(1, args.folds):
        next_proto = model(meta_support[p * i:p * (i + 1)])
        meta_logits = euclidean_metric(next_proto, meta_proto)
        soft_labels = (F.softmax(meta_logits, dim=1) + args.lam * s_onehot) / (1 + lam)
        meta_proto = torch.mm(soft_labels.permute((1, 0)), next_proto)

def inter_fold(model, args, meta_support):
    p = args.shot * args.train_way
    q = args.query * args.train_way
    meta_protos = []
    features = model(meta_support)
    for i in range(1, args.folds):
        proto = torch.cat([features[0:i], features[i+1:-1]])
        proto.reshape(args.shot-1, args.train_way, -1).mean(dim=0)
        current_ex = features[i]
        meta_logits = euclidean_metric(current_ex, proto)
        soft_labels = (F.softmax(meta_logits, dim=1))# + args.lam * true_labels) / (1 + args.lam)
        meta_proto = torch.mm(soft_labels.permute((1, 0)), current_ex)
        meta_protos.append(meta_proto)
    return torch.tensor(meta_protos)

def slice_data(data, num_ways, num_support, num_slices):
    labels = torch.arange(num_ways).repeat(num_support)
    p = int(num_ways * num_support / num_slices)
    sliced_labels = labels * num_slices
    for i in range(0, num_slices):
        sliced_labels[i * p:(i + 1) * p] = sliced_labels[i * p:(i + 1) * p] + i



    return sliced_data, sliced_labels

