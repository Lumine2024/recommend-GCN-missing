import argparse
import torch
from torch import optim
import numpy as np
import time
from scipy.sparse import csr_matrix
import pandas as pd
from torch.utils.data import Dataset
from torch import nn
from sklearn.metrics import roc_auc_score
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ==================== Configuration ====================
def parse_args():
    parser = argparse.ArgumentParser(description="GCN Recommendation")
    parser.add_argument('--bpr_batch', type=int, default=2048, help="batch size for BPR loss")
    parser.add_argument('--recdim', type=int, default=64, help="embedding size")
    parser.add_argument('--layer', type=int, default=3, help="number of GCN layers")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--decay', type=float, default=1e-4, help="weight decay")
    parser.add_argument('--dropout', type=int, default=0, help="use dropout")
    parser.add_argument('--keepprob', type=float, default=0.6, help="keep probability")
    parser.add_argument('--testbatch', type=int, default=100, help="test batch size")
    parser.add_argument('--dataset', type=str, default='lastfm', help="dataset name")
    parser.add_argument('--topks', type=str, default="[20]", help="@k test list")
    parser.add_argument('--epochs', type=int, default=500, help="training epochs")
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    return parser.parse_args()

# Global config
args = parse_args()
config = {
    'bpr_batch_size': args.bpr_batch,
    'latent_dim_rec': args.recdim,
    'GCN_n_layers': args.layer,
    'dropout': args.dropout,
    'keep_prob': args.keepprob,
    'test_u_batch_size': args.testbatch,
    'lr': args.lr,
    'decay': args.decay
}
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
TRAIN_epochs = args.epochs
topks = eval(args.topks)

def cprint(words: str) -> None:
    print(f"\033[0;30;43m{words}\033[0m")

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# ==================== Utils ====================
class BPRLoss:
    def __init__(self, recmodel, config: dict) -> None:
        self.model = recmodel
        self.weight_decay = config['decay']
        self.opt = optim.Adam(recmodel.parameters(), lr=config['lr'])

    def stageOne(self, users, pos, neg) -> float:
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        loss = loss + reg_loss * self.weight_decay
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.cpu().item()

def UniformSample_original(dataset) -> np.ndarray:
    """Sample user-positive-negative triplets for BPR training"""
    users = np.random.randint(0, dataset.n_users, dataset.trainDataSize)
    allPos = dataset.allPos
    S = []
    for user in users:
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        positem = posForUser[np.random.randint(0, len(posForUser))]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem not in posForUser:
                break
        S.append([user, positem, negitem])
    return np.array(S)

def minibatch(*tensors, batch_size=None):
    """Yield mini-batches from tensors"""
    if batch_size is None:
        batch_size = config['bpr_batch_size']
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

class timer:
    """Simple timer context manager"""
    NAMED_TAPE = {}
    
    def __init__(self, name=None):
        self.named = name
        if name:
            timer.NAMED_TAPE[name] = timer.NAMED_TAPE.get(name, 0.)
    
    @staticmethod
    def dict():
        return "|" + "|".join(f"{k}:{v:.2f}" for k, v in timer.NAMED_TAPE.items()) + "|"
    
    @staticmethod
    def zero():
        for key in timer.NAMED_TAPE:
            timer.NAMED_TAPE[key] = 0
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += time.time() - self.start

# Metrics
def RecallPrecision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / k
    return {'recall': recall, 'precision': precis}

def NDCGatK_r(test_data, r, k):
    pred_data = r[:, :k]
    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = min(k, len(items))
        test_matrix[i, :length] = 1
    idcg = np.sum(test_matrix / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = np.sum(pred_data / np.log2(np.arange(2, k + 2)), axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = np.array([x in groundTrue for x in predictTopK], dtype=float)
        r.append(pred)
    return np.array(r, dtype=float)

# ==================== Dataset ====================
class BasicDataset(Dataset):
    def __init__(self) -> None:
        print("init dataset")

class LastFM(BasicDataset):
    def __init__(self, path: str = "./data/lastfm") -> None:
        cprint("loading [last fm]")
        
        trainData = pd.read_table(path + '/train1.txt', header=None)
        testData = pd.read_table(path + '/test1.txt', header=None)
        
        self.trainUser = np.array(trainData[0])
        self.trainItem = np.array(trainData[1])
        self.testUser = np.array(testData[0])
        self.testItem = np.array(testData[1])
        
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.n_users = int(max(self.trainUser.max(), self.testUser.max())) + 1
        self.m_items = int(max(self.trainItem.max(), self.testItem.max())) + 1
        
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_users, self.m_items))
        
        print(f"LastFM Sparsity: {(len(self.trainUser) + len(self.testUser)) / self.n_users / self.m_items:.6f}")
        
        self.allPos = [self.UserItemNet[user].nonzero()[1] for user in range(self.n_users)]
        
        self.testDict = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if user in self.testDict:
                self.testDict[user].append(item)
            else:
                self.testDict[user] = [item]
        
        self.Graph = None
    
    @property
    def trainDataSize(self) -> int:
        return len(self.trainUser)
    
    def getUserPosItems(self, users: list) -> list:
        return [self.allPos[user] for user in users]
    
    def getSparseGraph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)
            
            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim + self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1), dtype=torch.float32)
            
            size = torch.Size([self.n_users + self.m_items, self.n_users + self.m_items])
            self.Graph = torch.sparse_coo_tensor(index, data, size=size).coalesce().to(device)
            
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D == 0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(0)
            dense = dense / D_sqrt / D_sqrt.t()
            
            index2 = dense.nonzero().t()
            data2 = dense[dense != 0]
            self.Graph = torch.sparse_coo_tensor(index2, data2, size=size).coalesce().to(device)
        
        return self.Graph

# ==================== Model ====================
class GCN(nn.Module):
    def __init__(self, config: dict, dataset) -> None:
        super(GCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.n_layers = config['GCN_n_layers']
        self.keep_prob = config['keep_prob']
        
        self.embedding_user = nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        
        self.f = nn.Sigmoid()
        self.Graph = dataset.getSparseGraph()
        print(f"GCN ready (dropout:{config['dropout']})")
    
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = (torch.rand(len(values)) + keep_prob).int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        return torch.sparse_coo_tensor(index.t(), values, size)
    
    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        
        g_droped = self.__dropout_x(self.Graph, self.keep_prob) if (self.config['dropout'] and self.training) else self.Graph
        
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        rating = self.f(torch.matmul(users_emb, all_items.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0 = self.getEmbedding(
            users.long(), pos.long(), neg.long())
        
        reg_loss = (userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)) / (2 * len(users))
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss

# ==================== Training & Testing ====================
def BPR_train_original(dataset, recommend_model, loss_class, neg_k: int = 1) -> str:
    recommend_model.train()
    
    with timer(name="Sample"):
        S = UniformSample_original(dataset)
    
    users = torch.LongTensor(S[:, 0]).to(device)
    posItems = torch.LongTensor(S[:, 1]).to(device)
    negItems = torch.LongTensor(S[:, 2]).to(device)
    
    indices = torch.randperm(len(users))
    users, posItems, negItems = users[indices], posItems[indices], negItems[indices]
    
    total_batch = len(users) // config['bpr_batch_size'] + 1
    aver_loss = 0.
    for batch_users, batch_pos, batch_neg in minibatch(users, posItems, negItems, batch_size=config['bpr_batch_size']):
        aver_loss += loss_class.stageOne(batch_users, batch_pos, batch_neg)
    
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"

def Test(dataset, Recmodel) -> dict:
    u_batch_size = config['test_u_batch_size']
    testDict = dataset.testDict
    Recmodel.eval()
    max_K = max(topks)
    
    results = {
        'precision': np.zeros(len(topks)),
        'recall': np.zeros(len(topks)),
        'ndcg': np.zeros(len(topks))
    }
    
    with torch.no_grad():
        users = list(testDict.keys())
        rating_list = []
        groundTrue_list = []
        
        for batch_users in minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.LongTensor(batch_users).to(device)
            
            rating = Recmodel.getUsersRating(batch_users_gpu)
            
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            
            _, rating_K = torch.topk(rating, k=max_K)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        
        for rating_K, groundTrue in zip(rating_list, groundTrue_list):
            r = getLabel(groundTrue, rating_K.numpy())
            for i, k in enumerate(topks):
                ret = RecallPrecision_ATk(groundTrue, r, k)
                results['recall'][i] += ret['recall']
                results['precision'][i] += ret['precision']
                results['ndcg'][i] += NDCGatK_r(groundTrue, r, k)
        
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        
        print(results)
        return results

# ==================== Main ====================
if __name__ == "__main__":
    set_seed(args.seed)
    print(">>SEED:", args.seed)
    
    dataset = LastFM()
    Recmodel = GCN(config, dataset).to(device)
    bpr = BPRLoss(Recmodel, config)
    
    for epoch in range(TRAIN_epochs):
        info = BPR_train_original(dataset, Recmodel, bpr, neg_k=1)
        print(f"EPOCH[{epoch + 1}/{TRAIN_epochs}] {info}")
        if (epoch + 1) % 10 == 0:
            cprint("[TEST]")
            Test(dataset, Recmodel)
