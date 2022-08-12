import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .Encoders import FIMG_Encoder
from models.base_model import BaseModel


class FIMGModel(BaseModel):
    def __init__(self, hparams, vocab, user_num):
        super().__init__(hparams)
        self.name = hparams['name']
        self.cdd_size = (hparams['npratio'] + 1) if hparams['npratio'] > 0 else 1
        self.his_size = hparams['his_size']
        self.batch_size = hparams['batch_size']
        self.signal_length = hparams['title_size']
        self.kernel_size = 3
        self.encoder = FIMG_Encoder(hparams, vocab)
        self.hidden_dim = self.encoder.hidden_dim
        self.level = self.encoder.level
        self.device = hparams['device']
        self.softmax = nn.Softmax(dim=-1)
        self.ReLU = nn.ReLU()
        self.DropOut = self.encoder.DropOut

        self.conv_3D_a = nn.Conv3d(in_channels=10, out_channels=32, kernel_size=3, padding=1)
        self.conv_3D_b = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.maxpool_3D = torch.nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3])
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
        fusion_tensor = fusion_tensor.view(-1, self.his_size, 1, self.signal_length,
                                           self.signal_length).transpose(1, 2)

        return fusion_tensor

    def _click_predictor(self, fusion_tensors):
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

    def forward(self, x):
        if x['candidate_title'].shape[0] != self.batch_size:
            self.batch_size = x['candidate_title'].shape[0]

        cdd_news = x['candidate_title'].long().to(self.device)
        c0, c1, c2, c3 = self.encoder(cdd_news)  # torch.Size([75, 20, 256])

        his_news = x['clicked_title'].long().to(self.device)
        b0, b1, b2, b3= self.encoder(his_news)
        # torch.Size([15, 5, 20, 1, 256])


        # 九张特征图
        fusion_tensors0 = self._fusion(c0, b0)
        fusion_tensors1 = self._fusion(c1, b1)
        fusion_tensors2 = self._fusion(c1, b2)
        fusion_tensors3 = self._fusion(c1, b3)
        fusion_tensors4 = self._fusion(c2, b1)
        fusion_tensors5 = self._fusion(c2, b2)
        fusion_tensors6 = self._fusion(c2, b3)
        fusion_tensors7 = self._fusion(c3, b1)
        fusion_tensors8 = self._fusion(c3, b2)
        fusion_tensors9 = self._fusion(c3, b3)

        fusion_tensors = torch.cat([fusion_tensors0,fusion_tensors1, fusion_tensors2, fusion_tensors3,
                                    fusion_tensors4, fusion_tensors5, fusion_tensors6,
                                    fusion_tensors7, fusion_tensors8, fusion_tensors9], dim=1)

        # 3D-CNN
        Q1 = F.elu(self.conv_3D_a(fusion_tensors), inplace=True)  # [batch_size * news_num, conv3D_filter_num_first, max_history_num, HDC_sequence_length, HDC_sequence_length]
        Q1 = self.maxpool_3D(Q1)  # [batch_size * news_num, conv3D_filter_num_first, max_history_num_conv1_size, HDC_sequence_length_conv1_size, HDC_sequence_length_conv1_size]
        Q2 = F.elu(self.conv_3D_b(Q1),inplace=True)  # [batch_size * news_num, conv3D_filter_num_second, max_history_num_pool1_size, HDC_sequence_length_pool1_size, HDC_sequence_length_pool1_size]
        fusion_tensors = self.maxpool_3D(Q2).view(self.batch_size, self.cdd_size, -1)

        #点击预测
        score = self._click_predictor(fusion_tensors)

        return score