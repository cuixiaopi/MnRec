import torch
import math
import torch.nn as nn
from models.base_model import BaseModel
import torch.nn.functional as F
from .Encoders import NPA_Encoder
#encode 建模新闻表示 NPAmodel 实现模型实现
class NPAModel(BaseModel):
    def __init__(self,hparams,vocab,user_num):
        super().__init__(hparams)
        self.name = 'npa'

        self.cdd_size = (hparams['npratio'] + 1) if hparams['npratio'] > 0 else 1
        self.batch_size = hparams['batch_size']
        self.signal_length = hparams['title_size']
        self.his_size =hparams['his_size']

        self.encoder = NPA_Encoder(hparams, vocab, user_num)

        self.hidden_dim = self.encoder.hidden_dim
        self.preference_dim =self.encoder.query_dim
        self.user_dim = self.encoder.user_dim
        self.user_embedding_news = nn.Parameter(torch.randn((self.hidden_dim, self.hidden_dim)).requires_grad_(True))

        self.device = torch.device(hparams['device'])

        # trainable lookup layer for user embedding, important to have len(uid2idx) + 1 rows because user indexes start from 1
        self.user_embedding = self.encoder.user_embedding
        # project e_u to word news preference vector of preference_dim
        self.newsPrefProject = nn.Linear(self.user_dim,self.preference_dim)
        # project preference query to vector of filter_num
        self.newsQueryProject = nn.Linear(self.preference_dim,self.hidden_dim)

        self.RELU = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.Tanh = nn.Tanh()
        self.DropOut = self.encoder.DropOut

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

        attn_weights = torch.matmul(query, key)/math.sqrt(query.shape[-1])
        # print(attn_weights.shape)
        attn_weights = self.softmax(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output

    def _user_encoder(self, his_news_batch, news_query, word_query):
        """ encode batch of user history clicked news to user representations of [batch_size,filter_num]

        Args:
            his_news_batch: tensor of [batch_size, his_size, title_size]
            news_query: tensor of [batch_size, preference_dim]
            word_query: tensor of [batch_size, preference_dim]

        Returns:
            user_repr: tensor of [batch_size, filter_num]
        """
        his_news_reprs = self._news_encoder(his_news_batch,word_query).view(self.batch_size,self.his_size,self.filter_num).permute(0,2,1)
        user_reprs = self._attention_news(news_query,his_news_reprs)

        return user_reprs

    def _attention_news(self, keys, cdd_news_reprs):
        """ apply original attention mechanism over news in user history

        Args:
            query: tensor of [batch_size, preference_dim] 【100，200】
            keys: tensor of [batch_size, filter_num, his_size] 【100，400，50】

        Returns:
            attn_aggr: tensor of [batch_size, filter_num], which is batch of user embedding 【100，400】
        """

        w=self.user_embedding_news
        cdd_news_reprs=cdd_news_reprs.view(self.batch_size, -1, self.hidden_dim) #[100,5,400]

        a = torch.matmul(cdd_news_reprs, w)
        # print(a.size())
        # print(keys.size())
        a1 = torch.matmul(a, keys)

        # print(a1.size())
        softmax = torch.nn.Softmax(dim=-1)
        att = softmax(a1) #[5,5,50]

        value = keys.permute(0, 2, 1)
        attn_aggr = torch.matmul(att, value) #[100,5,400] //候选项目表示


        return attn_aggr


    def _click_predictor(self,cdd_news_repr,user_repr):
        """ calculate batch of click probability

        Args:
            cdd_news_repr: tensor of [batch_size, cdd_size, hidden_dim]
            user_repr: tensor of [batch_size, 1, hidden_dim]

        Returns:
            score: tensor of [batch_size, cdd_size]
        """
        score = torch.bmm(cdd_news_repr,user_repr.transpose(-2,-1)).squeeze(dim=-1)
        if self.cdd_size > 1:
            score = nn.functional.log_softmax(score,dim=1)
        else:
            score = torch.sigmoid(score)
        return score

    def forward(self,x):
        if x['candidate_title'].shape[0] != self.batch_size:
            self.batch_size = x['candidate_title'].shape[0]

        user_index = x['user_index'].long().to(self.device)

        cdd_news = x['candidate_title'].long().to(self.device)
        _, cdd_news_repr = self.encoder(
            cdd_news,
            user_index=user_index)

        his_news = x['clicked_title'].long().to(self.device)
        _, his_news_repr = self.encoder(
            his_news,
            user_index=user_index)

        e_u = self.DropOut(self.user_embedding(user_index))
        news_query = self.Tanh(self.newsQueryProject(
            self.RELU(self.newsPrefProject(e_u))))

        user_repr = self._scaled_dp_attention(news_query, his_news_repr, his_news_repr)


        #新添加
        # print(cdd_news_repr.size()) [5,5,400]
        # print(his_news_repr.size()) [5,50,400]
        his_news_reprs=his_news_repr.permute(0,2,1)
        cdd_news_reprs = self._attention_news(his_news_reprs, cdd_news_repr) #[5,5,150]
        cd = cdd_news_repr.unsqueeze(dim=-2)  # [5,5,1,150]
        te = cdd_news_reprs.unsqueeze(dim=-2)  # [5,5,1,150]
        out = torch.cat([cd, te], dim=2)
        att = F.softmax(out, dim=2)
        cd_finall_repr = att[:,:, 0] * cdd_news_repr + att[:,:, 1] * cdd_news_reprs
        score = self._click_predictor(cd_finall_repr, user_repr)



        # score = self._click_predictor(cdd_news_repr, user_repr)
        return score