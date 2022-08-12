import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .Encoders import NRGMS_Encoder
from .Encoders import NPA_Encoder
from models.base_model import BaseModel
import math
from .multihead_self import MultiHeadSelfAttention


class NRGMSModel(BaseModel):
    def __init__(self, hparams, vocab, user_num):
        super().__init__(hparams)
        self.name = hparams['name']  # 模型的名称
        self.cdd_size = (hparams['npratio'] + 1) if hparams['npratio'] > 0 else 1  # 负采样新闻个数
        self.dropout_p = hparams['dropout_p']  # 减少过拟合 设置丢弃率 为0.2
        self.metrics = hparams['metrics']  # 评价指标
        self.batch_size = hparams['batch_size']  # 传入数据的批量大小
        self.signal_length = hparams['title_size']  # 新闻标题的长度
        self.his_size = hparams['his_size']  # 用于学习用户表示的最大新闻点击数设置为50
        self.W_b = nn.Parameter(torch.randn(128, 128).requires_grad_(True))
        self.head_num = hparams['head_num']
        self.trans_dim = hparams['trans_dim']

        self.Tanh = nn.Tanh()
        self.kernel_size = 3
        self.filter_num = hparams['filter_num']  # 通道个数//150
        self.embedding_dim = hparams['embedding_dim']  # word编码以后的向量维度300
        self.user_dim = hparams['user_dim']  # 编码用户的向量维度，学习用户的表示
        # self.preference_dim =hparams['preference_dim'] #单词和新闻偏好查询向量q设置大小为200
        self.device = torch.device(hparams['device'])  # gpu加载程序和数据
        self.encoder = NRGMS_Encoder(hparams, vocab, user_num)
        self.query_dim = hparams['trans_dim']
        self.hidden_dim = self.encoder.hidden_dim  # 400
        self.preference_dim = self.encoder.query_dim
        self.user_dim = self.encoder.user_dim
        self.ReLU = nn.ReLU()
        self.user_embedding_news = nn.Parameter(torch.randn((self.hidden_dim, self.hidden_dim)).requires_grad_(True))
        self.device = torch.device(hparams['device'])
        self.LayerNorm = nn.LayerNorm(32)
        self.user_embedding = self.encoder.user_embedding
        # project e_u to word news preference vector of preference_dim
        self.newsPrefProject = nn.Linear(self.user_dim, self.preference_dim)
        # project preference query to vector of filter_num
        self.newsQueryProject = nn.Linear(self.preference_dim, 128)
        self.newsQueryProject2 = nn.Linear(self.preference_dim, 256)

        self.trans_gate = nn.Linear(self.his_size, self.his_size)

        self.query_inte = nn.Linear(self.trans_dim, self.trans_dim)
        self.mu_inte = nn.Linear(self.trans_dim, self.trans_dim)
        self.query_news1 = nn.Parameter(torch.randn((1, self.query_dim), requires_grad=True))
        self.query_gate = nn.Linear(self.trans_dim, self.trans_dim)
        self.mu_gate = nn.Linear(self.trans_dim, self.trans_dim)

        self.query_news = nn.Parameter(torch.randn((1, self.query_dim), requires_grad=True))
        self.multihead_self_attention = MultiHeadSelfAttention(self.trans_dim, self.head_num)
        # pretrained embedding 预训练嵌入
        # self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=False)
        self.embedding = nn.Parameter(vocab.vectors.clone().detach().requires_grad_(True).to(self.device))
        # elements in the slice along dim will sum up to 1
        self.softmax = nn.Softmax(dim=-1)
        self.CNN = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.filter_num, kernel_size=1)
        self.RELU = nn.ReLU()
        self.DropOut = nn.Dropout(p=self.dropout_p)
        self.DropOut1 = nn.Dropout(p=0.5)

        self.learningToRank = nn.Linear(384, 128)
        self.learningToRank1 = nn.Linear(128, 1)

        self.attention_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )

        self.attention_layer1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )

        self.CNN_d1 = nn.Conv1d(in_channels=self.filter_num, out_channels=100,
                                kernel_size=self.kernel_size, dilation=1, padding=1)
        self.CNN_d2 = nn.Conv1d(in_channels=self.filter_num, out_channels=100,
                                kernel_size=self.kernel_size, dilation=2, padding=2)
        self.CNN_d3 = nn.Conv1d(in_channels=self.filter_num, out_channels=100,
                                kernel_size=self.kernel_size, dilation=3, padding=3)

        self.CNN_c1 = nn.Conv1d(in_channels=128, out_channels=128,
                                kernel_size=self.kernel_size, dilation=1, padding=1)
        self.CNN_c2 = nn.Conv1d(in_channels=128, out_channels=128,
                                kernel_size=self.kernel_size, dilation=2, padding=2)
        self.CNN_c3 = nn.Conv1d(in_channels=128, out_channels=128,
                                kernel_size=self.kernel_size, dilation=3, padding=3)

        self.lstm_net = nn.LSTM(300, 128, num_layers=2, dropout=0.1, bidirectional=True)
        self.lstm_net1 = nn.LSTM(128, 128, num_layers=2, dropout=0.1, bidirectional=True)

    def _cnn_resnet(self, C):
        # C [100,150,20]

        C1 = self.CNN_d1(C)
        RC1 = C1.transpose(-2, -1)
        RC1 = self.ReLU(RC1)

        C2 = self.CNN_d2(C)
        RC2 = C2.transpose(-2, -1)
        RC2 = self.ReLU(RC2)

        C3 = self.CNN_d3(C)
        RC3 = C3.transpose(-2, -1)
        RC3 = self.ReLU(RC3)  # [100,20,150]

        TRC = torch.cat([RC1, RC2, RC3], dim=-1)
        # print(TRC.size())

        return TRC

    def _cnn_resnetxh(self, C):
        # C [100,150,20]

        C1 = self.ReLU(self.CNN_c1(C))
        C2 = self.ReLU(self.CNN_c2(C1))
        C3 = self.ReLU(self.CNN_c3(C2))

        TRC = torch.cat([C1, C2, C3], dim=-2).transpose(-1, -2)
        TRC = self.learningToRank(TRC)
        # print(TRC.size())

        return TRC

    def _attention_news(self, his_news_repr, cdd_news_repr):
        """ apply original attention mechanism over news in user history

        Args:
            his_news_repr 5,50,128
           cdd_news_repr 5,5,128

        Returns:
            attn_aggr: tensor of [batch_size, filter_num], which is batch of user embedding 【100，400】
        """
        # his_news_repr, cdd_news_repr

        C = torch.matmul(cdd_news_repr, torch.matmul(self.W_b, his_news_repr.transpose(1, 2)))
        beta = F.softmax(C, -1)  # [5,5,50]

        target = torch.matmul(beta, his_news_repr)

        return target

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
        query = query.expand(key.shape[0], 1, self.query_dim)
        assert query.shape[-1] == key.shape[-1]
        key = key.transpose(-2, -1)
        attn_weights = torch.matmul(query, key) / math.sqrt(query.shape[-1])
        # print(attn_weights.shape)
        attn_weights = self.softmax(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output

    def _click_predictor(self, score):
        """ calculate batch of click probability

        Args:
            cdd_news_repr: tensor of [batch_size, cdd_size, hidden_dim]
            user_repr: tensor of [batch_size, 1, hidden_dim]

        Returns:
            score: tensor of [batch_size, cdd_size]
        """

        # score = self.learningToRank1(scores).squeeze(dim=-1)
        if self.cdd_size > 1:
            score = nn.functional.log_softmax(score, dim=1)
        else:
            score = torch.sigmoid(score)
        return score

    def attention_net_with_w(self, lstm_out, lstm_hidden):

        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))

        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

    def Iattention_net_with_w(self, lstm_out, lstm_hidden):

        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer1(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))

        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

    def _fusion(self, news_batch):

        # 【30，50，128】
        cdd_news_embedding = F.normalize(news_batch, dim=-1, p=2)
        fusion_matrices = torch.matmul(cdd_news_embedding, cdd_news_embedding.transpose(-1, -2))
        fusion_matrices = (0.5000 + 0.5000 * fusion_matrices)
        sim = torch.sum(fusion_matrices, dim=-1) / 50
        return sim

    def forward(self, x):
        if x['candidate_title'].shape[0] != self.batch_size:
            self.batch_size = x['candidate_title'].shape[0]
        if x['candidate_title'].shape[1] != self.cdd_size:
            self.cdd_size = x['candidate_title'].shape[1]

        user_index = x['user_index'].long().to(self.device)

        # d=x['his_id']
        # m=np.int64(d > 0)
        # mask = torch.from_numpy(m).to(self.device) #[30,50]
        # e_u = self.DropOut(self.user_embedding(user_index))
        # news_query = self.Tanh(self.newsQueryProject(
        #     self.RELU(self.newsPrefProject(e_u))))

        cdd_news = x['candidate_title'].long().to(self.device)
        cdd_news_repr, cnews_embedding_pretrained = self.encoder(cdd_news,
                                                                 user_index=user_index)  # torch.Size([30, 5, 20, 150]))

        his_news = x['clicked_title'].long().to(self.device)
        his_news_repr, hnews_embedding_pretrained = self.encoder(his_news,
                                                                 user_index=user_index)  # torch.Size([30, 50, 20, 150])

        cdd_news_repr = cdd_news_repr.transpose(-1, -2).view(-1, self.filter_num, self.signal_length)
        his_news_repr = his_news_repr.transpose(-1, -2).view(-1, self.filter_num, self.signal_length)

        cdd = self._cnn_resnet(cdd_news_repr)  # [150,20,300]
        his = self._cnn_resnet(his_news_repr)  # [1500,20,300]

        cnews_embedding_pretrained = cnews_embedding_pretrained.transpose(-1, -2)
        hnews_embedding_pretrained = hnews_embedding_pretrained.transpose(-1, -2)

        cdd = cdd + cnews_embedding_pretrained  # [150,20,300]
        his = his + hnews_embedding_pretrained  # [1500,20,300]

        # print(cdd.size())[100,20,300]
        # torch.Size([20*5, 20, 256])----->[20,5,128]
        output, (final_hidden_state, final_cell_state) = self.lstm_net(cdd.transpose(0, 1))
        # print(output.size())torch.Size([20, 100, 256])
        output = output.permute(1, 0, 2)
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        # print(final_hidden_state.size()) torch.Size([100, 4, 128])
        cdd_news_repr = self.attention_net_with_w(output, final_hidden_state).view(self.batch_size, self.cdd_size,
                                                                                   128)  # [20,5,128]

        # -------------------
        outputh, (final_hidden_stateh, final_cell_stateh) = self.lstm_net(his.transpose(0, 1))
        outputh = outputh.permute(1, 0, 2)
        final_hidden_stateh = final_hidden_stateh.permute(1, 0, 2)
        his_news_repr = self.attention_net_with_w(outputh, final_hidden_stateh).view(self.batch_size, self.his_size,
                                                                                     128)  # [30,50,128]
        # print(his_news_repr.size())#[20,50,128]
        # -------------------

        his_news_repr, h_query = self.multihead_self_attention(his_news_repr)  # torch.Size([20, 50, 128])

        # 实现对候选新闻进行重构转换

        #  -------- 门控调节 -----------------

        # x的变换 input [20,5,128] -->[20,5,1,128]--->[20,5,50,128]
        cdd_trans = cdd_news_repr.unsqueeze(dim=2).repeat(1, 1, self.his_size, 1)

        # -----v的内部信息流生成-----
        qu_intra = self.query_inte(h_query)  # torch.Size([20, 50, 128])
        mu_intra = self.mu_inte(his_news_repr)  # torch.Size([20, 50,128])
        v = (qu_intra + mu_intra).unsqueeze(dim=1)  # 【20，1,50，128】

        # -----g的外部信息流生成-----
        qu_gate = self.query_gate(h_query).unsqueeze(dim=1)  # torch.Size([20,1, 50, 128])

        if self.cdd_size > 1:  # torch.Size([20, 5, 50, 128])
            qu_gate = qu_gate.repeat(1, 5, 1, 1)
        else:
            qu_gate = qu_gate.repeat(1, 1, 1, 1)

        cdd_trans = self.trans_gate(cdd_trans.transpose(-1, -2)).transpose(-1, -2)  # torch.Size([20, 5, 50, 128])
        g = torch.sigmoid(qu_gate + cdd_trans)  # #[20,5,50,128]

        # -----聚合--------
        his_corr = g * v  # torch.Size([20, 5, 50, 128])

        # if self.cdd_size > 1:
        # his_news_repr = his_news_repr.unsqueeze(dim=1).repeat(1, 5, 1, 1)
        # else:
        # his_news_repr = his_news_repr.unsqueeze(dim=1).repeat(1, 1, 1, 1)

        # his = his_corr + his_news_repr  # [10,5,50,256]
        finall_hisnews = his_corr.view(-1, self.his_size, self.trans_dim)

        # part4 残差网络优化
        # his_news_repr【20，50，128】
        # out = self._cnn_resnetxh(his.transpose(-1, -2))
        # finall_hisnews = (his + out)

        user_repr = self._scaled_dp_attention(self.query_news, finall_hisnews, finall_hisnews).view(self.batch_size,
                                                                                                    self.cdd_size,
                                                                                                    self.trans_dim)
        user_repr1 = self._scaled_dp_attention(self.query_news1, his_news_repr, his_news_repr).view(self.batch_size,
                                                                                                    self.trans_dim,
                                                                                                    1)
        cdd1 = user_repr * cdd_news_repr  # torch.Size([20, 5, 128])
        score = torch.bmm(cdd1, user_repr1).squeeze(dim=-1)
        score = self._click_predictor(score)  # 传统
        return score