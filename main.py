import world
import utils
from world import cprint
import Procedure
import dataloader
import model

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================

dataset = dataloader.LastFM()

Recmodel = model.GCN(world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

Neg_k = 1

""" 构建一个模型的训练和测试过程"""
""" 缺失的代码包括:"""
"""（1）利用Procedure.py文件中的BPR_train_original函数，构建一个训练过程"""
"""（2）利用Procedure.py文件中的Test函数，构建一个测试过程"""
"""要求：每训练10个epoch，进行一次测试"""
"""在下方编写代码处编写代码"""

for epoch in range(world.TRAIN_epochs):

    """##################################"""
    """##################################"""

    """     此处编写代码   """

    """##################################"""
    """##################################"""
    info = Procedure.BPR_train_original(dataset, Recmodel, bpr, Neg_k)
    print(f"EPOCH[{epoch + 1}/500]{info}")
    if (epoch + 1) % 10 == 0:
        cprint("[TEST]")
        Procedure.Test(dataset, Recmodel)


