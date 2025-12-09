import torch
from typing import Tuple, List, Optional
from dataloader import BasicDataset
from torch import nn

torch.nn.functional.softplus

class BasicModel(nn.Module):    
    def __init__(self) -> None:
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self) -> None:
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError

""" 请完善GCN类中缺失的代码"""
""" 缺失的代码包括:"""
"""（1）完善computer函数中的代码，设计一个3层的图卷积网络层，输入为g_droped和all_emb""" # 吐槽：其实是 n_layers 层
"""推荐使用torch.sparse.mm进行计算，将每一层的输出存入embs的列表中"""

"""（2）完善bpr_loss函数中的代码，根据BPR损失（loss）函数的定义，设计一个loss函数，输入为pos_scores和neg_scores"""
"""loss函数输出然后通过torch.nn.functional.softplus的激活函数，最后使用torch.mean进行平均处理"""

"""（3）完善getUsersRating函数中的代码，根据推荐模型输出的定义，计算用户与物品之间的评分"""
"""输出的评分然后通过模型中定义的self.f激活函数"""
"""在下方编写代码处编写代码"""
class GCN(PairWiseModel):
    def __init__(self, 
                 config: dict, 
                 dataset: BasicDataset) -> None:
        super(GCN, self).__init__()
        self.config = config
        self.dataset =  dataset
        self.__init_weight()

    def __init_weight(self) -> None:
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['GCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"gcn is already to go(dropout:{self.config['dropout']})")

    # 随机删几条边
    def __dropout_x(self, x: torch.Tensor, keep_prob: float) -> torch.Tensor:
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse_coo_tensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob: float) -> torch.Tensor:
        graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        propagate methods for GCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                # print("droping") # 太吵了！！
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph

        """##################################"""
        """##################################"""

        """     此处编写代码   """

        """##################################"""
        """##################################"""

        # 迭代 n_layers 次，计算 X_(i+1) = A X_i
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        # 取平均值，并输出二分图
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()

        """##################################"""
        """##################################"""

        """     此处编写代码   """

        """##################################"""
        """##################################"""
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))

        return rating
    
    def getEmbedding(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    # BPR 损失函数
    def bpr_loss(self, users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) +
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        """##################################"""
        """##################################"""

        """     此处编写代码   """

        """##################################"""
        """##################################"""
        _sum = torch.nn.functional.softplus((neg_scores - pos_scores).float())
        loss = torch.mean(_sum)

        return loss, reg_loss
