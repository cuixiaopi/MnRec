import torch
import math
import torch.nn as nn
from .Encoders import FIM_Encoder
from models.base_model import BaseModel
from .Encoders import NPA_Encoder

class FIMNModel(BaseModel):
    def __init__(self, hparams, vocab, user_num):
        super().__init__(hparams)
        self.name = 'fim'

        self.cdd_size = (hparams['npratio'] + 1) if hparams['npratio'] > 0 else 1
        self.his_size = hparams['his_size']
        self.batch_size = hparams['batch_size']

        self.signal_length = hparams['title_size']

        self.kernel_size = 3

        self.encoder = FIM_Encoder(hparams, vocab)
        self.encoder_dis = NPA_Encoder(hparams, vocab, user_num)

        self.user_embedding = self.encoder_dis.user_embedding
        self.hidden_dim = self.encoder.hidden_dim
        self.level = self.encoder.level


        self.preference_dim = self.encoder_dis.query_dim
        self.user_dim = self.encoder_dis.user_dim
        self.RELU = nn.ReLU()

        self.newsPrefProject = nn.Linear(self.user_dim, self.preference_dim)
        # project preference query to vector of filter_num
        self.newsQueryProject = nn.Linear(self.preference_dim, self.hidden_dim)

        self.device = hparams['device']
        self.Tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.ReLU = nn.ReLU()
        self.DropOut = self.encoder.DropOut
        self.SeqCNN3D = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=32, kernel_size=[3, 3, 3], padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3]),
            nn.Conv3d(in_channels=32, out_channels=16, kernel_size=[3, 3, 3], padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3])
        )

        self.learningToRank = nn.Linear(int((int((self.his_size - 3) / 3 + 1) - 3) / 3 + 1) * 2 * 2 * 16, 1)

    def _fusion(self, cdd_news_embedding, his_news_embedding):
        """ construct fusion tensor between candidate news repr and history news repr at each dilation level

        Args:
            cdd_news_embedding: tensor of [batch_size, cdd_size, signal_length, level, filter_num]
            his_news_embedding: tensor of [batch_size, his_size, signal_length, level, filter_num]

        Returns:
            fusion_tensor: tensor of [batch_size, 320], where 320 is derived from MaxPooling with no padding
        """

        cdd_news_embedding = cdd_news_embedding.transpose(-2, -3)
        his_news_embedding = his_news_embedding.transpose(-2, -3)

        # [batch_size, cdd_size, his_size, level, signal_length, signal_length]
        fusion_tensor = torch.matmul(cdd_news_embedding.unsqueeze(dim=2),
                                     his_news_embedding.unsqueeze(dim=1).transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        # reshape the tensor in order to feed into 3D CNN pipeline
        fusion_tensor = fusion_tensor.view(-1, self.his_size, self.level, self.signal_length,
                                           self.signal_length).transpose(1, 2)

        fusion_tensor = self.SeqCNN3D(fusion_tensor).view(self.batch_size, self.cdd_size, -1)

        return fusion_tensor

    def _click_predictor_local(self, fusion_tensors):
        """ calculate batch of click probabolity

        Args:
            fusion_tensors: tensor of [batch_size, cdd_size, 320]

        Returns:
            score: tensor of [batch_size, npratio+1], which is normalized click probabilty
        """
        score = self.learningToRank(fusion_tensors).squeeze(dim=-1)
        if self.cdd_size > 1:
            score = nn.functional.log_softmax(score, dim=1)
        else:
            score = torch.sigmoid(score)
        return score

    def _click_predictor_dis(self, cdd_news_repr, user_repr):
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

    def _click_predictor(self,scores):
        """ calculate batch of click probability

        Args:
            cdd_news_repr: tensor of [batch_size, cdd_size, hidden_dim]
            user_repr: tensor of [batch_size, 1, hidden_dim]

        Returns:
            score: tensor of [batch_size, cdd_size]
        """
        # score_t = torch.bmm(cdd_news_repr, user_repr.transpose(-2, -1)).squeeze(dim=-1)

        if self.cdd_size > 1:
            score = nn.functional.log_softmax(scores, dim=1)
        else:
            score = torch.sigmoid(scores)
        return score



    def _scaled_dp_attention(self, query, key, value):

        assert query.shape[-1] == key.shape[-1]
        key = key.transpose(-2, -1)

        attn_weights = torch.matmul(query, key) / math.sqrt(query.shape[-1])

        attn_weights = self.softmax(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output

    def forward(self, x):
        if x['candidate_title'].shape[0] != self.batch_size:
            self.batch_size = x['candidate_title'].shape[0]

        cdd_news = x['candidate_title'].long().to(self.device)
        cdd_news_embedding, _ = self.encoder(cdd_news)

        his_news = x['clicked_title'].long().to(self.device)
        his_news_embedding, _ = self.encoder(his_news)

        fusion_tensors = self._fusion(cdd_news_embedding, his_news_embedding)

        #分布式表示
        user_index = x['user_index'].long().to(self.device)

        cdd_news_dis = x['candidate_title'].long().to(self.device)
        _, cdd_news_repr = self.encoder_dis(cdd_news_dis, user_index=user_index)

        his_news_dis = x['clicked_title'].long().to(self.device)
        _, his_news_repr = self.encoder_dis(his_news_dis, user_index=user_index)

        e_u = self.DropOut(self.user_embedding(user_index))
        news_query = self.Tanh(self.newsQueryProject(
            self.RELU(self.newsPrefProject(e_u))))

        user_repr = self._scaled_dp_attention(news_query, his_news_repr, his_news_repr)



     #改动



        score1 = self._click_predictor_local(fusion_tensors)
        score0 = self._click_predictor_dis(cdd_news_repr, user_repr)

        score=0.7*score0+0.3*score1
        score=self._click_predictor(score)

        return score