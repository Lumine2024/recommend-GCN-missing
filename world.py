import os
from os.path import join
import torch
from parse import parse_args

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = "D:\\文档\\Visual Studio 2022\\Python\\recommend-GCN-missing"
DATA_PATH = join(ROOT_PATH, 'data')

config = {}
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['GCN_n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob

config['test_u_batch_size'] = args.testbatch
config['lr'] = args.lr
config['decay'] = args.decay

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
seed = args.seed

dataset = args.dataset

TRAIN_epochs = args.epochs
topks = eval(args.topks)
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

def cprint(words: str) -> None:
    print(f"\033[0;30;43m{words}\033[0m")
