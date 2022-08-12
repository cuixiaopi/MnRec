import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .Encoders import NPLM_Encoder
from .Encoders import NPA_Encoder
from models.base_model import BaseModel
import math



class NPLMModel(BaseModel):
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



        self.Tanh = nn.Tanh()
        self.kernel_size = 3
        self.filter_num = hparams['filter_num']  # 通道个数//150
        self.embedding_dim = hparams['embedding_dim']  # word编码以后的向量维度300
        self.user_dim = hparams['user_dim']  # 编码用户的向量维度，学习用户的表示
        # self.preference_dim =hparams['preference_dim'] #单词和新闻偏好查询向量q设置大小为200
        self.device = torch.device(hparams['device'])  # gpu加载程序和数据
        self.encoder = NPLM_Encoder(hparams, vocab, user_num)
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
        self.newsQueryProject = nn.Linear(self.preference_dim, 256)




        # pretrained embedding 预训练嵌入
        # self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=False)
        self.embedding = nn.Parameter(vocab.vectors.clone().detach().requires_grad_(True).to(self.device))
        # elements in the slice along dim will sum up to 1
        self.softmax = nn.Softmax(dim=-1)
        self.CNN = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.filter_num, kernel_size=1)
        self.RELU = nn.ReLU()
        self.DropOut = nn.Dropout(p=self.dropout_p)
        self.DropOut1 = nn.Dropout(p=0.5)

        self.learningToRank = nn.Linear(384,256)


        self.attention_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )

        self.CNN_d1 = nn.Conv1d(in_channels=self.filter_num, out_channels=100,
                                kernel_size=self.kernel_size, dilation=1, padding=1)
        self.CNN_d2 = nn.Conv1d(in_channels=self.filter_num, out_channels=100,
                                kernel_size=self.kernel_size, dilation=2, padding=2)
        self.CNN_d3 = nn.Conv1d(in_channels=self.filter_num, out_channels=100,
                                kernel_size=self.kernel_size, dilation=3, padding=3)

        self.CNN_c1 = nn.Conv1d(in_channels=256, out_channels=128,
                                kernel_size=self.kernel_size, dilation=1, padding=1)
        self.CNN_c2 = nn.Conv1d(in_channels=128, out_channels=128,
                                kernel_size=self.kernel_size, dilation=2, padding=2)
        self.CNN_c3 = nn.Conv1d(in_channels=128, out_channels=128,
                                kernel_size=self.kernel_size, dilation=3, padding=3)


        self.lstm_net = nn.LSTM(300, 128, num_layers=2, dropout=0.1, bidirectional=True)



        self.lstm_net1 = nn.LSTM(self.embedding_dim, 128, num_layers=2, dropout=0.1, bidirectional=True)
        self.lstm_net2 = nn.LSTM(256, 128, num_layers=2, dropout=0.1, bidirectional=True)


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

        TRC = torch.cat([C1, C2, C3], dim=-2).transpose(-1,-2)
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
        beta = F.softmax(C, -1)  #[5,5,50]

        target =torch.matmul(beta,his_news_repr)

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
        assert query.shape[-1] == key.shape[-1]
        key = key.transpose(-2, -1)
        attn_weights = torch.matmul(query, key) / math.sqrt(query.shape[-1])
        # print(attn_weights.shape)
        attn_weights = self.softmax(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output

    def _click_predictor(self,scores ):
        """ calculate batch of click probability

        Args:
            cdd_news_repr: tensor of [batch_size, cdd_size, hidden_dim]
            user_repr: tensor of [batch_size, 1, hidden_dim]

        Returns:
            score: tensor of [batch_size, cdd_size]
        """
        # score_t = torch.bmm(cdd_news_repr, user_repr.transpose(-2, -1)).squeeze(dim=-1)
        # scores = torch.bmm(cdd_news_repr, user_repr.transpose(-2, -1)).squeeze(dim=-1)
        if self.cdd_size > 1:
            score = nn.functional.log_softmax(scores, dim=1)
        else:
            score = torch.sigmoid(scores)
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




    def _fusion(self, news_batch):

        #【30，50，128】
        cdd_news_embedding = F.normalize(news_batch, dim=-1, p=2)
        fusion_matrices = torch.matmul(cdd_news_embedding, cdd_news_embedding.transpose(-1,-2))
        fusion_matrices = (0.5000 + 0.5000 * fusion_matrices)
        sim=torch.sum(fusion_matrices,dim=-1)/50
        return sim

    def forward(self, x):
        if x['candidate_title'].shape[0] != self.batch_size:
            self.batch_size = x['candidate_title'].shape[0]
        user_index = x['user_index'].long().to(self.device)
        e_u = self.DropOut(self.user_embedding(user_index))
        news_query = self.Tanh(self.newsQueryProject(self.RELU(self.newsPrefProject(e_u))))

        cdd_news = x['candidate_title'].long().to(self.device)
        cdd_news_repr, cnews_embedding_pretrained = self.encoder(cdd_news, user_index=user_index)  # torch.Size([30, 5, 20, 150]))

        his_news = x['clicked_title'].long().to(self.device)
        his_news_repr, hnews_embedding_pretrained = self.encoder(his_news,user_index=user_index)  # torch.Size([30, 50, 20, 150])

        cdd_news_repr = cdd_news_repr.transpose(-1, -2).view(-1, self.filter_num, self.signal_length)
        his_news_repr = his_news_repr.transpose(-1, -2).view(-1, self.filter_num, self.signal_length)

        #第一种横向粒度表示

        #粒度网络
        cdd = self._cnn_resnet(cdd_news_repr)  # [150,20,300]
        his = self._cnn_resnet(his_news_repr)  # [1500,20,300]
        cnews_embedding_pretrained = cnews_embedding_pretrained.transpose(-1, -2)
        hnews_embedding_pretrained = hnews_embedding_pretrained.transpose(-1, -2)
        cdd = cdd + cnews_embedding_pretrained  # [150,20,300]
        his = his + hnews_embedding_pretrained  # [1500,20,300] {batch*cdd,length,embedding_size]

        # ATT-BILSTM
        output, (final_hidden_state, final_cell_state) = self.lstm_net(cdd.transpose(0, 1))
        output = output.permute(1, 0, 2)
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        cdd_news_repr_1 = self.attention_net_with_w(output, final_hidden_state).view(self.batch_size, self.cdd_size,128)  # [20,5,128]

        outputh, (final_hidden_stateh, final_cell_stateh) = self.lstm_net(his.transpose(0, 1))  # [20,1000,256]
        outputh = outputh.permute(1, 0, 2)
        final_hidden_stateh = final_hidden_stateh.permute(1, 0, 2)
        his_news_repr_1 = self.attention_net_with_w(outputh, final_hidden_stateh).view(self.batch_size, self.his_size, 128)  # [30,50,128]


        #层次粒度
        # input_size [length , batch * cdd , embedding_size]
        #{batch * cdd, length, embedding_size]
        outputl1, _ = self.lstm_net1(cnews_embedding_pretrained.transpose(0,1)) # torch.Size([20, 100, 256])
        outputl2, (cfi_hi_st2, cfi_ce_st2) = self.lstm_net2(outputl1)
        coutputl3 = (outputl1+outputl2).permute(1, 0, 2)
        cfi_hi_st2 = cfi_hi_st2.permute(1, 0, 2)
        c_1 = self.attention_net_with_w(coutputl3, cfi_hi_st2).view(self.batch_size,  self.cdd_size,128)

        hisoutputl1, _ = self.lstm_net1(hnews_embedding_pretrained.transpose(0,1))  # torch.Size([20, 100, 256])
        hisoutputl2, (hfi_hi_st2, hfi_ce_st2) = self.lstm_net2(hisoutputl1)
        hisoutputl3 = (hisoutputl1 + hisoutputl2).permute(1, 0, 2)
        hfi_hi_st2 = hfi_hi_st2.permute(1, 0, 2)
        h_1 = self.attention_net_with_w(hisoutputl3, hfi_hi_st2).view(self.batch_size, self.his_size, 128)
        #层次结束

        #表示聚合

        #动态聚合
        # cdd_news_repr_finall=torch.cat([cdd_news_repr_1,c_1], dim=-1)
        # attcdd=torch.sigmoid(self.liner(cdd_news_repr_finall))
        # ccd = attcdd * cdd_news_repr_1 + (1-attcdd) * c_1
        #
        # his_news_repr_finall = torch.cat([his_news_repr_1, h_1], dim=-1)
        # atthis = torch.sigmoid(self.liner(his_news_repr_finall))
        # hhd = atthis * his_news_repr_1 + (1 - atthis) * h_1

        #
        ccd = torch.cat([cdd_news_repr_1, c_1], dim=-1) #【100，5，256】
        hhd = torch.cat([his_news_repr_1, h_1], dim=-1) #【100，50，256】





        #part4 残差网络优化
        # his_news_repr【20，50，128】
        out = self._cnn_resnetxh(hhd.transpose(-1, -2))
        hhd = hhd + out
        user_repr2 = self._scaled_dp_attention(news_query, hhd, hhd)
        # print(user_repr2.size()) torch.Size([20, 1, 128])

        #预测
        scores5 = torch.bmm(ccd, user_repr2.transpose(-2, -1)).squeeze(dim=-1)


        #part3 结束
        score = self._click_predictor(scores5)  # 传统
        return score