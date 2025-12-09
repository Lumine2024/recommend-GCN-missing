import argparse
from argparse import Namespace

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Go GCN")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of GCN")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of GCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='lastfm',
                        help="available datasets: [lastfm]")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    parser.add_argument('--epochs', type=int,default=500)
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    return parser.parse_args()
