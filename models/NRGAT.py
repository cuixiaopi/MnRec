import torch
import torch.nn as nn
import math
from models.base_model import BaseModel


class NRGATodel(BaseModel):
    def __init__(self, hparams, vocab):
        super().__init__(hparams)
        self.name = hparams['name']

        self.cdd_size = (hparams['npratio'] + 1) if hparams['npratio'] > 0 else 1

        self.device = torch.device(hparams['device'])
        if hparams['train_embedding']:
            self.embedding = nn.Parameter(vocab.vectors.clone().detach().requires_grad_(True).to(self.device))
        else:
            self.embedding = vocab.vectors.to(self.device)

        self.batch_size = hparams['batch_size']
        self.signal_length = hparams['title_size']
        self.his_size = hparams['his_size']
        self.ReLU = nn.ReLU()
        self.head_num = hparams['head_num']
        self.query_dim = hparams['query_dim']
        self.embedding_dim = hparams['embedding_dim']
        self.value_dim = hparams['value_dim']
        self.LayerNorm = nn.LayerNorm(self.query_dim)
        self.repr_dim = self.head_num * self.value_dim

        self.query_words = nn.Parameter(torch.randn((1, self.query_dim), requires_grad=True))
        self.query_news = nn.Parameter(torch.randn((1, self.query_dim), requires_grad=True))

        # elements in the slice along dim will sum up to 1
        self.softmax = nn.Softmax(dim=-1)
        self.ReLU = nn.ReLU()
        self.DropOut = nn.Dropout(p=hparams['dropout_p'])

        self.queryProject_words = nn.ModuleList([]).extend(
            [nn.Linear(self.embedding_dim, self.embedding_dim, bias=False) for _ in range(self.head_num)])
        self.queryProject_news = nn.ModuleList([]).extend(
            [nn.Linear(self.repr_dim, self.repr_dim, bias=False) for _ in range(self.head_num)])

        self.forqueryProject_news = nn.Linear(self.repr_dim, self.repr_dim, bias=True)

        self.valueProject_words = nn.ModuleList([]).extend(
            [nn.Linear(self.embedding_dim, self.value_dim, bias=False) for _ in range(self.head_num)])
        self.valueProject_news = nn.ModuleList([]).extend(
            [nn.Linear(self.repr_dim, self.value_dim, bias=False) for _ in range(self.head_num)])

        self.forvalueProject_news = nn.Linear(self.repr_dim, self.repr_dim, bias=True)

        self.keyProject_words = nn.Linear(self.value_dim * self.head_num, self.query_dim, bias=True)
        self.keyProject_news = nn.Linear(self.value_dim * self.head_num, self.query_dim, bias=True)

        self.feat = nn.Linear(1, self.his_size)
        self.feat_query = nn.Linear(self.value_dim * self.head_num, self.query_dim, bias=True)
        self.gfeat_query = nn.Linear(self.value_dim * self.head_num, self.query_dim, bias=True)
        self.feat_trans = nn.Linear(self.value_dim * self.head_num, self.query_dim, bias=True)

        self.learningToRank = nn.Linear(self.repr_dim, 1)

    def _scaled_dp_attention(self, query, key, value):
        """ calculate scaled attended output of values

        Args:
            query: tensor of [*, query_num, key_dim]
            key: tensor of [batch_size, *, key_num, key_dim]
            value: tensor of [batch_size, *, key_num, value_dim]

        Returns:
            attn_output: tensor of [batch_size, *, query_num, value_dim]
        """

        # make sure dimension matches
        assert query.shape[-1] == key.shape[-1]
        key = key.transpose(-2, -1)

        attn_weights = torch.matmul(query, key) / math.sqrt(query.shape[-1])
        attn_weights = self.softmax(attn_weights)

        attn_output = torch.matmul(attn_weights, value)

        return attn_output

    def _self_attention(self, input, head_idx, mode):
        """ apply self attention of head#idx over input tensor

        Args:
            input: tensor of [batch_size, *, embedding_dim]
            head_idx: interger of attention head index
            mode: 0/1, 1 for words self attention, 0 for news self attention

        Returns:
            self_attn_output: tensor of [batch_size, *, value_dim]
        """
        if mode:
            query = self.queryProject_words[head_idx](input)  # y_query
            # print(query.size()) [10,5,20,300]  [10,50,20,300]
            attn_output = self._scaled_dp_attention(query, input, input)
            self_attn_output = self.valueProject_words[head_idx](attn_output)

            return self_attn_output

        else:

            query = self.queryProject_news[head_idx](input)  # [10,50,256]
            value_for = self.forvalueProject_news(input)
            attn_output = self._scaled_dp_attention(query, value_for, value_for)
            self_attn_output = self.valueProject_news[head_idx](attn_output)  # [10,50,16]
            return self_attn_output

    def _self_attention_news(self, input, query_for, head_idx, mode):
        """ apply self attention of head#idx over input tensor

        Args:
            input: tensor of [batch_size, *, embedding_dim]
            head_idx: interger of attention head index
            mode: 0/1, 1 for words self attention, 0 for news self attention

        Returns:
            self_attn_output: tensor of [batch_size, *, value_dim]
        """
        if mode:
            query = self.queryProject_words[head_idx](input)  # y_query
            # print(query.size()) [10,5,20,300]  [10,50,20,300]
            attn_output = self._scaled_dp_attention(query, input, input)
            self_attn_output = self.valueProject_words[head_idx](attn_output)

            return self_attn_output

        else:

            query = self.queryProject_news[head_idx](query_for)  # [10,50,256]
            value_for = self.forvalueProject_news(input)
            attn_output = self._scaled_dp_attention(query, value_for, value_for)
            self_attn_output = self.valueProject_news[head_idx](attn_output)  # [10,50,16]
            return self_attn_output

    def _word_attention(self, query, key, value):
        """ apply word-level attention

        Args:
            query: tensor of [1, query_dim]
            key: tensor of [batch_size, *, signal_length, query_dim]
            value: tensor of [batch_size, *, signal_length, repr_dim]

        Returns:
            attn_output: tensor of [batch_size, *, repr_dim]
        """
        # query = query.expand(key.shape[0], key.shape[1], 1, self.query_dim)

        attn_output = self._scaled_dp_attention(query, key, value).squeeze(dim=-2)

        return attn_output

    def _news_attention(self, query, key, value):
        """ apply news-level attention

        Args:
            attn_word_embedding_key: tensor of [batch_size, his_size, query_dim]
            attn_word_embedding_value: tensor of [batch_size, his_size, repr_dim]

        Returns:
            attn_output: tensor of [batch_size, 1, repr_dim]
        """

        attn_output = self._scaled_dp_attention(query, key, value)

        return attn_output

    def _multi_head_self_attention(self, input, mode):
        """ apply multi-head self attention over input tensor

        Args:
            input: tensor of [batch_size, *, signal_length, embedding_dim]
            mode: 0/1, 1 for words self attention, 0 for news self attention

        Returns:
            multi_head_self_attn: tensor of [batch_size, *, 1, repr_dim]
        """
        if mode:
            self_attn_outputs = [self._self_attention(input, i, 1) for i in range(self.head_num)]

            # project the embedding of each words to query subspace
            # keep the original embedding of each words as values
            multi_head_self_attn_value = torch.cat(self_attn_outputs, dim=-1)
            # 多头注意力结束
            # print(multi_head_self_attn_value.size()) #orch.Size([10, 5, 20, 256]) //torch.Size([10, 50, 20, 256])

            multi_head_self_attn_key = torch.tanh(self.keyProject_words(multi_head_self_attn_value))
            additive_attn_repr = self._word_attention(self.query_words, multi_head_self_attn_key,
                                                      multi_head_self_attn_value)

            return additive_attn_repr

        else:

            query_for = self.forqueryProject_news(input)
            self_attn_outputs = [self._self_attention_news(input, query_for, i, 0) for i in range(self.head_num)]
            # project the embedding of each words to query subspace
            multi_head_self_attn_value = torch.cat(self_attn_outputs, dim=-1)

            # multi_head_self_attn_key = torch.tanh(self.keyProject_news(multi_head_self_attn_value))
            # additive_attn_repr = self._news_attention(self.query_news, multi_head_self_attn_key,multi_head_self_attn_value)

            return query_for, multi_head_self_attn_value

    # def _multi_head_self_attention1(self, input, mode):
    #     """ apply multi-head self attention over input tensor
    #
    #     Args:
    #         input: tensor of [batch_size, *, signal_length, embedding_dim]
    #         mode: 0/1, 1 for words self attention, 0 for news self attention
    #
    #     Returns:
    #         multi_head_self_attn: tensor of [batch_size, *, 1, repr_dim]
    #     """
    #     if mode:
    #         self_attn_outputs = [self._self_attention(input, i, 1) for i in range(self.head_num)]
    #
    #         # project the embedding of each words to query subspace
    #         # keep the original embedding of each words as values
    #         multi_head_self_attn_value = torch.cat(self_attn_outputs, dim=-1)
    #         # 多头注意力结束
    #         # print(multi_head_self_attn_value.size()) #orch.Size([10, 5, 20, 256]) //torch.Size([10, 50, 20, 256])
    #
    #         multi_head_self_attn_key = torch.tanh(self.keyProject_words(multi_head_self_attn_value))
    #         additive_attn_repr = self._word_attention(self.query_words, multi_head_self_attn_key,
    #                                                   multi_head_self_attn_value)
    #
    #         return additive_attn_repr
    #
    #     else:
    #         self_attn_outputs = [self._self_attention(input, i, 0) for i in range(self.head_num)]
    #
    #         # project the embedding of each words to query subspace
    #         # keep the original embedding of each words as values
    #         multi_head_self_attn_value = torch.cat(self_attn_outputs, dim=-1)
    #         multi_head_self_attn_key = torch.tanh(self.keyProject_news(multi_head_self_attn_value))
    #         additive_attn_repr = self._news_attention(self.query_news, multi_head_self_attn_key,
    #                                                   multi_head_self_attn_value)
    #
    #         return additive_attn_repr

    def _news_encoder(self, news_batch):
        """ encode set of news to news representations of [batch_size, cdd_size, tranformer_dim]

        Args:
            news_batch: tensor of [batch_size, cdd_size, title_size]

        Returns:
            news_reprs: tensor of [batch_size, cdd_size, repr_dim]
        """
        news_embedding = self.DropOut(self.embedding[news_batch])
        news_reprs = self._multi_head_self_attention(news_embedding, 1)
        return news_reprs

    def _user_encoder(self, his_news_batch):
        """ encode set of news to news representations of [batch_size, cdd_size, tranformer_dim]

        Args:
            his_news_batch: tensor of [batch_size, his_size, title_size]

        Returns:
            user_reprs: tensor of [batch_size, 1, repr_dim]
        """
        his_news_reprs = self._news_encoder(his_news_batch)  # [10,50,256]

        return his_news_reprs

    def _click_predictor(self, sc):
        """ calculate batch of click probability

        Args:
            cdd_news_repr: tensor of [batch_size, cdd_size, repr_dim]
            user_repr: tensor of [batch_size, 1, repr_dim]

        Returns:
            score: tensor of [batch_size, cdd_size]
        """
        score = self.learningToRank(sc).squeeze(dim=-1)
        if self.cdd_size > 1:
            score = nn.functional.log_softmax(score, dim=1)
        else:
            score = torch.sigmoid(score)
        return score

    def forward(self, x):
        if x['candidate_title'].shape[0] != self.batch_size:
            self.batch_size = x['candidate_title'].shape[0]
        if x['candidate_title'].shape[1] != self.cdd_size:
            self.cdd_size = x['candidate_title'].shape[1]

        cdd_news_reprs = self._news_encoder(x['candidate_title'].long().to(self.device))  # x
        # print(cdd_news_reprs.size()) [10,5,256]

        his_news_reprs = self._user_encoder(x['clicked_title'].long().to(self.device))  # y
        # print(his_news_reprs.size()) torch.Size([10, 50, 256])

        # 改进模块

        # x的变换
        x_cdd_news_reprs = cdd_news_reprs.unsqueeze(dim=2).transpose(-1, -2)
        x_cdd_trans = self.feat(x_cdd_news_reprs).transpose(-1, -2)  # torch.Size([10, 5, 50, 256])

        # y_q以及z的生成
        y_q, multi_head_self_attn_value = self._multi_head_self_attention(his_news_reprs, 0)

        # v的生成（无法判断是否仍tanh）
        multi_head_self_attn_key = self.keyProject_news(multi_head_self_attn_value)  # torch.Size([10, 50, 256])
        v_feat = self.feat_query(y_q)  # torch.Size([10, 50, 256])
        v = (multi_head_self_attn_key + v_feat).unsqueeze(dim=1)  # 【10，1,50，256】

        # g的生成
        #
        # # tip 1 乘积形式
        # g_feat = self.gfeat_query(y_q).unsqueeze(dim=1)  # torch.Size([10, 1, 50, 256])
        # g=torch.sigmoid(x_cdd_trans*g_feat) #[10,5,50,256]

        # tip 2 加和形式(论文方法)
        g_feat = self.gfeat_query(y_q).unsqueeze(dim=1)  # torch.Size([10, 1, 50, 256])

        if self.cdd_size > 1:
            g_feat = g_feat.repeat(1, 5, 1, 1)
        else:
            g_feat = g_feat.repeat(1, 1, 1, 1)

        x_cdd_trans = self.feat_trans(x_cdd_trans)
        g = torch.sigmoid(g_feat + x_cdd_trans)  # #[10,5,50,256]

        sc_for = g * v  # print(sc_for.size()) torch.Size([10, 5, 50, 256])

        # print(sc_for.size())

        # 残差连接归一化

        if self.cdd_size > 1:
            his_news_norm = his_news_reprs.unsqueeze(dim=1).repeat(1, 5, 1, 1)
        else:
            his_news_norm = his_news_reprs.unsqueeze(dim=1).repeat(1, 1, 1, 1)

        his = self.LayerNorm(sc_for + his_news_norm)  # [10,5,50,256]

        # add
        # his = sc_for + his_news_norm # [10,5,50,256] 【原始信息+修正信息】

        # user_reprs = self._multi_head_self_attention1(his, 0).squeeze(dim=-2)

        # print(user_reprs.size())

        user_reprs = self._news_attention(self.query_news, his, his).squeeze(dim=-2)  # [10,5,256]

        cdd_news_reprs = self.LayerNorm(cdd_news_reprs)  # [10,5,50,256]

        score_for = user_reprs * cdd_news_reprs

        score = self._click_predictor(score_for)
        # print(score.size())

        return score