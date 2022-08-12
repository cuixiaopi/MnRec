import os
import sys
os.chdir('/NEWS/Codes/News-Recommendation')
sys.path.append('/NEWS/Codes/News-Recommendation')

import torch
from utils.utils import evaluate,train,prepare,load_hparams,test
from models.KNRM import KNRMModel

if __name__ == "__main__":    
    hparams = {
        'name':'knrm',
        'embedding_dim':300,
        'kernel_num':11,
        'epochs': 1,  # 跑十轮
        'scale': 'demo',  # demo模型
        'batch_size': 5,  # 批量大小设置为100
        'title_size': 20,  # 新闻标题的最大长度设置（论文中设置为30）
        'his_size': 50,  # 用于学习用户表示的最大新闻点击数设置为50
        'npratio': 4,  # 负采样的新闻个数
        'metrics': 'auc,mean_mrr,ndcg@5,ndcg@10',  # 实验评价指标
        'attrs': ['title'],
        'device': 'cuda:0',  # gpu
        # 'device':'cpu',
        'k': 0,
        'select': None,
        'save_step': [0],
        'save_each_epoch': False,  # 是否保存每一步输出
        'train_embedding': True,  # 对embedding进行训练
        'mode': 'dev',  # 喂进去的数据是训练数据 对应prepare()不同策略
        'news_id': False,
        'validate': False,
    }


    device = torch.device(hparams['device'])

    vocab, loaders = prepare(hparams)
    knrmModel = KNRMModel(vocab=vocab,hparams=hparams).to(device)

    if hparams['mode'] == 'dev':
        evaluate(knrmModel,hparams,loaders[0],load=True)

    elif hparams['mode'] == 'train':
        train(knrmModel, hparams, loaders)
    
    elif hparams['mode'] == 'test':
        test(knrmModel, hparams, loaders[0])