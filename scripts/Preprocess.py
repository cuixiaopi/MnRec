#预数据处理
import os
import sys
os.chdir('../')
sys.path.append('../')

import torch
from utils.utils import prepare,analyse,constructBasicDict,tailorData

if __name__ == '__main__':
    hparams = {
    'npratio':4,#新闻负采样本的个数
    'mode':'train',
    'scale':'demo',#demo模型用来训练
    'batch_size':10,#一批次10个
    'his_size':50,#用户最大的历史点击新闻数目
    'title_size':20,#新闻标题的最大长度
    'device':'cuda:0',
    # 'device':'cpu',
    'attrs': ['title'],
    'news_id':True,
    'k': 0,
    'validate':False,
    'multiview': False,
    'abs_size': 40,
    'onehot': False,
    'learning_rate': 1e-3,
    'spadam': False,
    'schedule': None,
    'interval': 10,
    'val_freq': 1,
    }
    #torch.cuda.set_device(device)
    #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(hparams['device'])
    """
   torch.device代表将torch.Tensor分配到的设备的对象。
   torch.device包含一个设备类型（'cpu'或'cuda'设备类型）和可选的设备的序号。
    """
    #构造字典
    constructBasicDict(attrs=['title'],path='/NPA/Data/MIND')
    #view data
    vocab, loaders = prepare(hparams)

    # loader_train
    #a = next(iter(loaders[0]))
    #print(a)
    # loader_dev
    #b = next(iter(loaders[1]))
    #print(b)

    #Tailor Data to demo size
    # tailor 2000 impressions from MINDsmall_train to form MINDdemo_trai
    tailorData('/NEWS/Data/MIND/MINDsmall_train/behaviors.tsv',8000)#2000/500

    tailorData('/NEWS/Data/MIND/MINDsmall_dev/behaviors.tsv',2000)


    # analyse(hparams)
