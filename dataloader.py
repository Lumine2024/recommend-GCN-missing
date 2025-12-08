from os.path import join
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import world
from world import cprint

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):

        raise NotImplementedError
    
    def getSparseGraph(self):
        raise NotImplementedError


""" 请完善LastFM类中缺失的代码"""
""" 缺失的代码包括:"""
"""（1）LastFM数据的加载（推荐使用pandas的read_table）"""
"""LastFM中的train1.txt的数据格式：第一列为user-Id；第二列为item-Id；第三列不需要使用"""
"""由于计算机的存储都是从0开始，得到的用户和物品的size需要-1"""
"""（2）用户与物品之间的交互图的构建（推荐使用scipy.csr_matrix）,交互图请以self.UserItemNet命名"""
"""UserItemNet是一个1892*4489的稀疏矩阵"""
"""在下方编写代码处编写代码"""
class LastFM(BasicDataset):
    def __init__(self, path="./data/lastfm"):
        # train or test
        cprint("loading [last fm]")
        self.mode_dict = {'train':0, "test":1}
        self.mode = self.mode_dict['train']
        # self.n_users = 1892 self.m_items = 4489

        """##################################"""
        """##################################"""

        """     此处编写代码   """
        """     训练和测试数据加载   """

        """##################################"""
        """##################################"""
        trainData = pd.read_table(path + '/train1.txt', header=None)
        testData = pd.read_table(path + '/test1.txt', header=None)


        self.trainData = trainData  ####trainData为加载的训练数据
        self.testData  = testData   ####testData为加载的测试数据
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        self.testUser  = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = np.array(testData[:][1])
        self.Graph = None

        self._n_users = int(max(self.trainUser.max(), self.testUser.max())) + 1
        self._m_items = int(max(self.trainItem.max(), self.testItem.max())) + 1

        """##################################"""
        """##################################"""

        """     此处编写代码   """
        """     UserItemNet的构建   """

        """##################################"""
        """##################################"""
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)), shape=(self.n_users,self.m_items)) 


        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser)) / self.n_users / self.m_items}")
        
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return self._n_users
    
    @property
    def m_items(self):
        return self._m_items
    
    @property
    def trainDataSize(self):
        return len(self.trainUser)
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)
            
            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim+self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            # index: shape (2, N) long tensor, data: shape (N,) tensor
            # 使用 float dtype 更方便后续归一化计算
            index = index.long()          # 确保索引为 long
            data = torch.ones(index.size(-1), dtype=torch.float32, device='cpu')
            
            size = torch.Size([self.n_users + self.m_items, self.n_users + self.m_items])
            # 创建稀疏 COO 张量
            self.Graph = torch.sparse_coo_tensor(index, data, size=size, dtype=torch.float32)
            
            # coalesce 合并重复索引，放到设备上
            self.Graph = self.Graph.coalesce().to(world.device)
            
            # 如果后续需要密集矩阵进行 D^(-1/2) * A * D^(-1/2) 计算，
            # 目前代码是把 sparse 转 dense 再做归一化（注意内存）
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D == 0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(0)
            dense = dense / D_sqrt
            dense = dense / D_sqrt.t()
            
            # 从归一化后的 dense 恢复成稀疏
            index2 = dense.nonzero().t()
            data2 = dense[dense != 0]
            self.Graph = torch.sparse_coo_tensor(index2, data2, size=size, dtype=torch.float32).coalesce().to(world.device)

        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems

    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user
    
    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']
    
    def __len__(self):
        return len(self.trainUniqueUsers)

