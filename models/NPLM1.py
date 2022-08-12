import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .Encoders import NPLM_Encoder
from .Encoders import NPA_Encoder
from models.base_model import BaseModel
import math
from torch.nn.utils import weight_norm

#底层修改

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
        # trainable lookup layer for user embedding, important to have len(uid2idx) + 1 rows because user indexes start from 1
        self.user_embedding = self.encoder.user_embedding
        # project e_u to word news preference vector of preference_dim
        self.newsPrefProject = nn.Linear(self.user_dim, self.preference_dim)
        # project preference query to vector of filter_num
        self.newsQueryProject = nn.Linear(self.preference_dim, 128)
        self.newsQueryProject2 = nn.Linear(self.preference_dim, 256)


        self.embedding = nn.Parameter(vocab.vectors.clone().detach().requires_grad_(True).to(self.device))
        # elements in the slice along dim will sum up to 1
        self.softmax = nn.Softmax(dim=-1)
        self.CNN = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.filter_num, kernel_size=1)
        self.RELU = nn.ReLU()
        self.DropOut = nn.Dropout(p=self.dropout_p)
        self.DropOut1 = nn.Dropout(p=0.5)

        self.attention_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )

        self.attention_layer1 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True)
        )

        self.conv1 = weight_norm(nn.Conv1d(300, 256, 3, stride=1, padding=1, dilation=1))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.45)

        self.conv2 = weight_norm(nn.Conv1d(256, 256, 3, stride=1, padding=2, dilation=2))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.45)

        self.conv3 = weight_norm(nn.Conv1d(128, 256, 3, stride=1, padding=1, dilation=1))
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.45)

        self.conv4 = weight_norm(nn.Conv1d(256, 128, 3, stride=1, padding=2, dilation=2))
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.45)


        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,self.conv2, self.relu2, self.dropout2)
        self.net1 = nn.Sequential(self.conv3, self.relu3, self.dropout3, self.conv4, self.relu4, self.dropout4)

        self.downsample = nn.Conv1d(300, 256, 1)
        self.downsample1 = nn.Conv1d(128,128, 1)

        self.init_weights()
        self.init_weights1()



        self.net = nn.Sequential(self.conv1,  self.relu1, self.dropout1,
                                 self.conv2,  self.relu2, self.dropout2)

        self.CNN_d1 = nn.Conv1d(in_channels=self.filter_num, out_channels=100,
                                kernel_size=self.kernel_size, dilation=1, padding=1)
        self.CNN_d2 = nn.Conv1d(in_channels=self.filter_num, out_channels=100,
                                kernel_size=self.kernel_size, dilation=2, padding=2)
        self.CNN_d3 = nn.Conv1d(in_channels=self.filter_num, out_channels=100,
                                kernel_size=self.kernel_size, dilation=3, padding=3)

        self.CNN_c1 = nn.Conv1d(in_channels=128, out_channels=32,
                                kernel_size=self.kernel_size, dilation=1, padding=1)
        self.CNN_c2 = nn.Conv1d(in_channels=128, out_channels=32,
                                kernel_size=self.kernel_size, dilation=2, padding=2)
        self.CNN_c3 = nn.Conv1d(in_channels=128, out_channels=32,
                                kernel_size=self.kernel_size, dilation=3, padding=3)
        self.CNN_c4 = nn.Conv1d(in_channels=128, out_channels=32,
                                kernel_size=self.kernel_size, dilation=4, padding=4)

        self.lstm_net = nn.LSTM(256, 128, num_layers=2, dropout=0.1, bidirectional=True)
        self.lstm_net1 = nn.LSTM(128, 128, num_layers=2, dropout=0.1, bidirectional=True)

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    def init_weights1(self):
        self.conv3.weight.data.normal_(0, 0.01)
        self.conv4.weight.data.normal_(0, 0.01)
        if self.downsample1 is not None:
            self.downsample1.weight.data.normal_(0, 0.01)


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

    def _cnn_resneth(self, C):
        # C [100,150,20]

        C1 = self.CNN_c1(C)
        #RC1 = C1.transpose(-2, -1)
        RC1 = self.LayerNorm(C1.transpose(-2, -1))
        RC1 = self.DropOut(self.ReLU(RC1))


        C2 = self.CNN_c2(C)
        # RC2 = C2.transpose(-2, -1)
        RC2 = self.LayerNorm(C2.transpose(-2, -1))
        RC2 = self.DropOut(self.ReLU(RC2))

        C3 = self.CNN_c3(C)
        # RC3 = C3.transpose(-2, -1)
        RC3 = self.LayerNorm(C3.transpose(-2, -1))
        RC3 = self.DropOut(self.ReLU(RC3))  # [100,20,150]

        C4 = self.CNN_c4(C)
        # RC4 = C4.transpose(-2, -1)
        RC4 = self.LayerNorm(C4.transpose(-2, -1))
        RC4 = self.DropOut(self.ReLU(RC4))  # [100,20,150]

        TRC = torch.cat([RC1, RC2, RC3, RC4], dim=-1)
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

        # print(cnews_embedding_pretrained.size()) #torch.Size([100, 300, 20])

        cout = self.net(cnews_embedding_pretrained)
        cres = cnews_embedding_pretrained if self.downsample is None else self.downsample(cnews_embedding_pretrained)
        cdd_news=self.RELU(cout + cres).transpose(-1,-2) #[torch.Size([100, 256, 20])

        hout = self.net(hnews_embedding_pretrained)
        hres = hnews_embedding_pretrained if self.downsample is None else self.downsample(hnews_embedding_pretrained)
        his_news = self.RELU(hout + hres).transpose(-1,-2) #torch.Size([1000, 256, 20])

        # print(cdd.size())[100,20,300]
        # torch.Size([20*5, 20, 256])----->[20,5,128]
        output, (final_hidden_state, final_cell_state) = self.lstm_net(cdd_news.transpose(0, 1))
        # print(output.size())torch.Size([20, 100, 256])
        output = output.permute(1, 0, 2)
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        # print(final_hidden_state.size()) torch.Size([100, 4, 128])
        cddrepr = self.attention_net_with_w(output, final_hidden_state).view(self.batch_size, self.cdd_size,
                                                                                   128)  # [20,5,128]

        # -------------------
        outputh, (final_hidden_stateh, final_cell_stateh) = self.lstm_net(his_news.transpose(0, 1))
        outputh = outputh.permute(1, 0, 2)
        final_hidden_stateh = final_hidden_stateh.permute(1, 0, 2)
        hisrepr = self.attention_net_with_w(outputh, final_hidden_stateh).view(self.batch_size, self.his_size, 128).transpose(-1,-2)  # [30,50,128]  # [30,50,128]
        #print(hisrepr.size()) [20,128,50]




        ##part5 利用残差网络分层建模用户兴趣
        hn = self.net1(hisrepr)
        hs = hisrepr if self.downsample1 is None else self.downsample1(hisrepr)
        hoo = self.RELU(hn + hs).transpose(-1,-2)


        # part5.1 注意力方案聚合兴趣表示
        user_repr2 = self._scaled_dp_attention(news_query, hoo, hoo)
        # print(user_repr2.size()) torch.Size([20, 1, 128])
        scores2 = torch.bmm(cddrepr, user_repr2.transpose(-2, -1)).squeeze(dim=-1)
        # part1 结束

       #------------------------------------------------
        # # # part5.2 LSTM聚合兴趣表示
        # # #【20，50，128】--->[20,1,64]
        # Ioutput, (Ifinal_hidden_state, Ifinal_cell_state) = self.lstm_net1(hoo.transpose(0, 1))
        # # print(Ioutput.size()) torch.Size([50, 20, 256])
        # Ioutput = Ioutput.permute(1, 0, 2)
        # Ifinal_hidden_state = Ifinal_hidden_state.permute(1, 0, 2)
        # # print(Ifinal_hidden_state.size()) torch.Size([20, 4, 128])
        # user_repr3 = self.Iattention_net_with_w(Ioutput, Ifinal_hidden_state).unsqueeze(dim=1) #torch.Size([20, 128])
        # scores3 = torch.bmm(cddrepr, user_repr3.transpose(-2, -1)).squeeze(dim=-1)
        # # part2 结束


        score = self._click_predictor(scores3)  # 传统
        return score
