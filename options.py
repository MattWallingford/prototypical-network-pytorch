import argparse
import os

def TrainOptions():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--max-epoch', type=int, default=200)
        parser.add_argument('--save-epoch', type=int, default=20)
        parser.add_argument('--shot', type=int, default=1)
        parser.add_argument('--query', type=int, default=15)
        parser.add_argument('--train-way', type=int, default=30)
        parser.add_argument('--test-way', type=int, default=5)
        parser.add_argument('--save-path', default='./save/proto-1')
        parser.add_argument('--gpu', default='0')
        parser.add_argument('--load', default='na')
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--start_epoch', type=int, default=1)
        parser.add_argument('--noise', type=int, default=.02)
        parser.add_argument('--folds', type=int, default=2)
        parser.add_argument('--lam', type=int, default=0)
        parser.add_argument('--num_slices', type=int, default=1)
        self.parser = parser

    def parse_args(self):
        return self.parser.parse_args()