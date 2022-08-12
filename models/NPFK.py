'''
Author: Pt
Date: 2020-11-05 18:05:03
LastEditTime: 2020-12-09 15:41:02
'''
#0.4 0.6
def kernel_mus(n_kernels):
    """
    get the mu for each guassian kernel. Mu is the middle of each bin
    :param n_kernels: number of kernels (including exact match). first one is exact match
    :return: l_mu, a list of mu.
    """
    mus  = [1]
    if n_kernels == 1:
        return mus

    bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    mus .append(1 - bin_size / 2)  # mu: middle of the bin
    for i in range(1, n_kernels - 1):
        mus .append(mus [i] - bin_size)
    return mus

def kernel_sigmas(n_kernels):

    l_sigma = [0.001]
    if n_kernels == 1:
        return l_sigma

    l_sigma += [0.1] * (n_kernels - 1)
    return l_sigma

import torch
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
from .Encoders import NPAU_Encoder
from models.base_model import BaseModel
import math
#目前主流
class NPFKModel(BaseModel):
    def __init__(self,hparams,vocab,user_num):
        super().__init__(hparams)
        self.name = hparams['name'] #模型的名称
        self.cdd_size = (hparams['npratio'] + 1) if hparams['npratio'] > 0 else 1
        self.dropout_p = hparams['dropout_p']
        self.metrics = hparams['metrics']
        self.batch_size = hparams['batch_size']
        self.signal_length = hparams['title_size']
        self.his_size =hparams['his_size']
        self.filter_num = hparams['filter_num']
        self.embedding_dim = hparams['embedding_dim']
        self.device = torch.device(hparams['device'])

        self.ascore = torch.nn.Parameter(torch.rand(()))
        self.bscore = torch.nn.Parameter(torch.rand(()))

        # self.n = hparams['n']
        # self.m = hparams['m']

        self.softmax = nn.Softmax(dim=-1)
        self.Tanh = nn.Tanh()
        self.RELU = nn.ReLU()
        self.DropOut = nn.Dropout(p=self.dropout_p)

        self.encoder = NPAU_Encoder(hparams, vocab, user_num)
        # self.embedding = nn.Parameter(vocab.vectors.clone().detach().requires_grad_(True).to(self.device))
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=False)
        #GL
        self.user_dim = hparams['user_dim']
        self.user_embedding = self.encoder.user_embedding

        self.preference_dim = hparams['preference_dim']
        self.newsPrefProject = nn.Linear(self.user_dim, self.preference_dim)
        self.newsQueryProject = nn.Linear(self.preference_dim, self.filter_num)

        #TRAGET
        self.linear_t = nn.Linear(self.filter_num, self.filter_num)  # target attention

        #LOCAL

        self.kernel_num = hparams['kernel_num']  # 核的长度是11
        mus = torch.tensor(kernel_mus(hparams['kernel_num']), dtype=torch.float)  # 新添加
        self.mus = nn.Parameter(mus.view(1, 1, 1, 1, 1, hparams['kernel_num']), requires_grad=False).to(self.device)


        sigmas=torch.tensor(kernel_sigmas(hparams['kernel_num']))
        self.sigmas = nn.Parameter(sigmas.view(1, 1, 1, 1, 1, hparams['kernel_num']), requires_grad=False).to(self.device)

        self.learningToRank = nn.Linear(55, 1)


        self.CNN0 = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.filter_num, kernel_size=3,padding=1)
        self.SeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=11, out_channels=11, kernel_size=3, padding=1),#11---->1
            nn.ReLU(),
	        nn.MaxPool1d(kernel_size=3, stride=3),

	        nn.Conv1d(in_channels=11, out_channels=11, kernel_size=3, padding=1),#11---->1
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),


        )


    def _scaled_dp_attention(self, query, key, value):

        assert query.shape[-1] == key.shape[-1]
        key = key.transpose(-2, -1)
        attn_weights = torch.matmul(query, key) / math.sqrt(query.shape[-1])
        attn_weights = self.softmax(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
        return attn_output


    def _attention_news(self, hidden,b):

        qt = self.linear_t(hidden)
        attn=(b @ qt.transpose(1, 2))/ math.sqrt(qt.shape[-1])
        print(attn.size())
        beta = self.softmax(attn)
        target = beta @ hidden  # batch_size x n_nodes x latent_size


        return target

    def _kernel_pooling(self, matrices, mask_cdd, mask_his):

        pooling_matrices = torch.exp(-((matrices - self.mus) ** 2) / (2 * (self.sigmas ** 2)))
        print(pooling_matrices.size())
        pooling_matrices = pooling_matrices * mask_his
        pooling_sum = torch.sum(pooling_matrices, dim=-2)

        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10))
        log_pooling_sum = log_pooling_sum * mask_cdd * 0.01

        pooling_vectors = torch.sum(log_pooling_sum, dim=-2)
        return pooling_vectors

    def _news_encoder_knrm(self, news_batch):


        cdd_title_embedding = self.RELU(self.CNN0(news_batch))
        return cdd_title_embedding

    def _fusion(self, cdd_news_batch, his_news_batch):


        cdd_news_embedding=cdd_news_batch.view(self.batch_size,-1,self.signal_length,self.filter_num).unsqueeze(dim=2)
        cdd_news_embedding = F.normalize(cdd_news_embedding, dim=-1, p=2)
        # cdd_news_embedding=torch.nan_to_num(cdd_news_embedding / torch.linalg.norm(cdd_news_embedding, 2, dim=-1, keepdims=True))

        his_news_embedding=his_news_batch.view(self.batch_size,self.his_size,self.signal_length,self.filter_num).unsqueeze(dim=1)
        his_news_embedding = F.normalize(his_news_embedding, dim=-1, p=2).transpose(-1,-2)
        #his_news_embedding = torch.nan_to_num(his_news_embedding / torch.linalg.norm(his_news_embedding, 2, dim=-1, keepdims=True)).transpose(-1,-2)
        fusion_matrices = torch.matmul(cdd_news_embedding, his_news_embedding)

        sim = (0.5000 + 0.5000 * fusion_matrices).unsqueeze(dim=-1)
        #


        # fusion_matrices = sim

        return sim

    def _click_(self, score_t):

        if self.cdd_size > 1:
            score = nn.functional.log_softmax(score_t, dim=1)
        else:
            score = torch.sigmoid(score_t)
        return score

    def _click_t(self, score_t):

        if self.cdd_size > 1:
            score = nn.functional.log_softmax(score_t, dim=1)
        else:
            score = score_t
        return score



    def forward(self,x):
        if x['candidate_title'].shape[0] != self.batch_size:
            self.batch_size = x['candidate_title'].shape[0]
        user_index = x['user_index'].long().to(self.device)


        cdd_news = x['candidate_title'].long().to(self.device)
        _, cdd_news_repr, _, cpembedding = self.encoder(cdd_news,user_index=user_index)
        can_embedding = self._news_encoder_knrm(cpembedding ).transpose(-1, -2)


        his_news = x['clicked_title'].long().to(self.device)
        _, his_news_repr, e_u, hpembedding = self.encoder(his_news,user_index=user_index)
        cli_embedding = self._news_encoder_knrm( hpembedding ).transpose(-1, -2)


        news_query = self.Tanh(self.newsQueryProject(self.RELU(self.newsPrefProject(e_u))))

        #GL
        user_repr = self._scaled_dp_attention(news_query, his_news_repr, his_news_repr)
        scores1 = torch.bmm(cdd_news_repr, user_repr.transpose(-2, -1)).squeeze(dim=-1)



        #TARGET(观测和完善)
        user_repr1 = self._attention_news(his_news_repr, cdd_news_repr)  # [10,5,256]
        scores = torch.sum(user_repr1 * cdd_news_repr, -1) # b,n)


        # local 局部训练（mask的观测，代码比较与改善）
        #cdd_news_knrm = x['candidate_title'].long().to(self.device)
        #his_news_knrm = x['clicked_title'].long().to(self.device)


        #是否需要進行留待觀測
        #can_embedding = self._news_encoder_knrm(cdd_news ).transpose(-1, -2)
        #cli_embedding = self._news_encoder_knrm( his_news ).transpose(-1, -2)


        fusion_matrices = self._fusion(can_embedding, cli_embedding).to(self.device)

        mask_cdd = x['candidate_title_pad'].float().to(self.device).view(self.batch_size, self.cdd_size, 1,self.signal_length, 1)
        mask_his = x['clicked_title_pad'].float().to(self.device).view(self.batch_size, 1, self.his_size, 1,self.signal_length, 1)

        pooling_vectors = self._kernel_pooling(fusion_matrices, mask_cdd, mask_his).view(-1, self.kernel_num, self.his_size)  # [5,5,50,11]



        #方案一
        fusion_tensor = self.SeqCNN1D(pooling_vectors).view(self.batch_size, self.cdd_size, -1)
        # print(fusion_tensor.size())


        score_z = torch.sigmoid(self.learningToRank(fusion_tensor)).squeeze(dim=-1)  # [得分】【5，5，80】
        # score_2 = self._click_t(score_z)

        score0=self.ascore*scores+self.bscore*score_z+(1-self.bscore-self.ascore)*scores1

        # score0 = scores + score_z +  scores1
        # score0=s * scores + (1-self.a-self.b)scores1+ self.b
        score_1 = self._click_(score0)


        return score_1