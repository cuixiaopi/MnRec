import os
import sys

os.chdir('/NEWS/Codes/News-Recommendation')
sys.path.append('/NEWS/Codes/News-Recommendation')


import torch
from utils.utils import  prepare, load_hparams
from models.NPLM3 import NPLMModel
import random
import numpy as np

if __name__ == "__main__":
    hparams = {
        'name': 'nplm',  # NPA模型
        'epochs': 6,  # 跑十轮5
        'scale': 'demo',  # demo模型
        'mode': 'tune',  # 喂进去的数据是训练数据 对应prepare()不同策略
        'batch_size':20,  # 批量大小设置为100  200
        'title_size': 20,  # 新闻标题的最大长度设置（论文中设置为30）
        'his_size': 50,  # 用于学习用户表示的最大新闻点击数设置为50
        'npratio': 4,  # 负采样的新闻个数
        'dropout_p': 0.2,  # 减少过拟合 设置丢弃率 为0.2
        'embedding_dim': 300,  # 单词嵌入维数为300 em
        'user_dim': 50,  # 用户嵌入维数设置为50 eu
        'preference_dim': 200,  # 单词和新闻偏好查询向量q维度设置大小为200
        'filter_num': 150,  # 通道个数
        'metrics': 'auc,mean_mrr,ndcg@5,ndcg@10',  # 实验评价指标
        'attrs': ['title'],
        'device': 'cuda:0',  # gpu
        # 'device':'cpu',
        'k': 0,
        'select': None,
        'save_step': [0],#196
        # 'save_each_epoch': False,  # 是否保存每一步输出
        'train_embedding': True,  # 对embedding进行训练
        'news_id': False,
        'validate': False,
        'multiview': False,
        'abs_size': 40,
        'onehot': False,
        'learning_rate': 1e-3,
        'spadam': False,
        'schedule': None,
        'interval': 10,
        'val_freq': 1,
    }


    # def setup_seed(seed):
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     # torch.backends.cudnn.benchmark = False
    #     # torch.backends.cudnn.deterministic = True
    #
    # seed = 2000
    # setup_seed(seed)

    device = torch.device(hparams['device'])

    vocab, loaders = prepare(hparams)
    nplmModel = NPLMModel(vocab=vocab, hparams=hparams, user_num=len(loaders[0].dataset.uid2index)).to(hparams['device'])

    if hparams['mode'] == 'dev':
        # evaluate_dev(npfkModel, hparams, loaders[0], load=True,)
        nplmModel.evaluate(hparams, loaders[0], load=True)

    elif hparams['mode'] == 'train':
        # train(npfkModel, hparams, loaders)
         nplmModel.fit(hparams, loaders)

    elif hparams['mode'] == 'test':
        # test(npfkModel, hparams, loaders[0])
         nplmModel.test(hparams, loaders[0])
    elif hparams['mode'] == 'tune':
         nplmModel.tune(hparams, loaders)