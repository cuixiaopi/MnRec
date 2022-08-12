
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .Encoders import NPA_Encoder
from models.base_model import BaseModel
import math

class NPFFModel(BaseModel):
    def __init__(self,hparams,vocab,user_num):
        super().__init__(hparams)
        self.name = hparams['name'] #模型的名称
        self.cdd_size = (hparams['npratio'] + 1) if hparams['npratio'] > 0 else 1 #负采样新闻个数
        self.dropout_p = hparams['dropout_p'] #减少过拟合 设置丢弃率 为0.2
        self.metrics = hparams['metrics'] #评价指标
        self.batch_size = hparams['batch_size'] #传入数据的批量大小
        self.signal_length = hparams['title_size'] #新闻标题的长度
        self.his_size =hparams['his_size']#用于学习用户表示的最大新闻点击数设置为50

        self.Tanh = nn.Tanh()
        self.filter_num = hparams['filter_num']#通道个数//150
        self.embedding_dim = hparams['embedding_dim']#word编码以后的向量维度300
        self.user_dim = hparams['user_dim'] #编码用户的向量维度，学习用户的表示
        # self.preference_dim =hparams['preference_dim'] #单词和新闻偏好查询向量q设置大小为200
        self.device = torch.device(hparams['device']) #gpu加载程序和数据
        self.encoder = NPA_Encoder(hparams, vocab, user_num)
        self.hidden_dim = self.encoder.hidden_dim#400
        self.preference_dim = self.encoder.query_dim
        self.user_dim = self.encoder.user_dim
        self.user_embedding_news = nn.Parameter(torch.randn((self.hidden_dim, self.hidden_dim)).requires_grad_(True))
        self.device = torch.device(hparams['device'])
        # trainable lookup layer for user embedding, important to have len(uid2idx) + 1 rows because user indexes start from 1
        self.user_embedding = self.encoder.user_embedding
        # project e_u to word news preference vector of preference_dim
        self.newsPrefProject = nn.Linear(self.user_dim, self.preference_dim)
        # project preference query to vector of filter_num
        self.newsQueryProject = nn.Linear(self.preference_dim, self.hidden_dim)
        # pretrained embedding 预训练嵌入
        # self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=False)
        self.embedding = nn.Parameter(vocab.vectors.clone().detach().requires_grad_(True).to(self.device))
        # elements in the slice along dim will sum up to 1
        self.softmax = nn.Softmax(dim=-1)
        self.CNN = nn.Conv1d(in_channels=self.embedding_dim,out_channels=self.filter_num,kernel_size=3,padding=1)
        self.RELU = nn.ReLU()
        self.DropOut = nn.Dropout(p=self.dropout_p)

    def _scaled_dp_attention(self, query, key, value):
        """ calculate scaled attended output of values

        Args:
            query: tensor of [batch_size, *, query_num, key_dim]
            key: tensor of [batch_size, *, key_num, key_dim]
            value: tensor of [batch_size, *, key_num, value_dim]

        Returns:
            attn_output: tensor of [batch_size, *, query_num, value_dim]
        """

        # make sure dimension matches
        assert query.shape[-1] == key.shape[-1]
        key = key.transpose(-2, -1)

        attn_weights = torch.matmul(query, key) / math.sqrt(query.shape[-1])
        # print(attn_weights.shape)
        attn_weights = self.softmax(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output

    def _attention_news(self, hidden,b):
        """ apply original attention mechanism over news in user history

        Args:
            query: tensor of [batch_size, preference_dim] 【100，200】
            keys: tensor of [batch_size, filter_num, his_size] 【100，400，50】

        Returns:
            attn_aggr: tensor of [batch_size, filter_num], which is batch of user embedding 【100，400】
        """

        qt = self.linear_t(hidden)
        beta = F.softmax(b @ qt.transpose(1, 2), -1)  # batch_size x n_nodes x seq_length
        target = beta @ hidden  # batch_size x n_nodes x latent_size


        return target

    def _click_predictor(self, cdd_news_repr, user_repr):
        """ calculate batch of click probability

        Args:
            cdd_news_repr: tensor of [batch_size, cdd_size, hidden_dim]
            user_repr: tensor of [batch_size, 1, hidden_dim]

        Returns:
            score: tensor of [batch_size, cdd_size]
        """
        # score_t = torch.bmm(cdd_news_repr, user_repr.transpose(-2, -1)).squeeze(dim=-1)
        scores = torch.bmm(cdd_news_repr, user_repr.transpose(-2, -1)).squeeze(dim=-1)
        if self.cdd_size > 1:
            score = nn.functional.log_softmax(scores, dim=1)
        else:
            score = torch.sigmoid(scores)
        return score

    def forward(self,x):
        if x['candidate_title'].shape[0] != self.batch_size:
            self.batch_size = x['candidate_title'].shape[0]

        user_index = x['user_index'].long().to(self.device)

        cdd_news = x['candidate_title'].long().to(self.device)
        _, cdd_news_repr = self.encoder( cdd_news, user_index=user_index)

        his_news = x['clicked_title'].long().to(self.device)
        _, his_news_repr = self.encoder( his_news,user_index=user_index)


        e_u = self.DropOut(self.user_embedding(user_index))
        news_query = self.Tanh(self.newsQueryProject(
            self.RELU(self.newsPrefProject(e_u))))



        user_repr = self._scaled_dp_attention(news_query, his_news_repr, his_news_repr)



        # scores1 = torch.bmm(cdd_news_repr, user_repr.transpose(-2, -1)).squeeze(dim=-1)


        # score = self._click_(scores1)

        score = self._click_predictor(cdd_news_repr, user_repr)




        return score