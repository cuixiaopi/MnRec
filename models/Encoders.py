import os
import logging
import math
import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F


class NRMS_Encoder(nn.Module):
    def __init__(self, hparams, vocab):
        super().__init__()
        self.name = 'nrms-encoder'

        self.level = 1

        self.embedding_dim = hparams['embedding_dim']
        self.value_dim = hparams['value_dim']
        self.query_dim = hparams['query_dim']
        self.head_num = hparams['head_num']

        # dimension for the final output embedding/representation
        self.hidden_dim = self.value_dim * self.head_num

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors,freeze=False)

        self.softmax = nn.Softmax(dim=-1)
        self.DropOut = nn.Dropout(p=hparams['dropout_p'])

        self.query = nn.Parameter(torch.randn(
            (1, self.query_dim), requires_grad=True))
        self.queryProjects = nn.ModuleList(
            [nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(self.head_num)])
        self.valueProjects = nn.ModuleList(
            [nn.Linear(self.embedding_dim, self.value_dim) for _ in range(self.head_num)])
        self.keyProject = nn.Linear(self.hidden_dim, self.query_dim)
        self.valueProject = nn.Linear(self.hidden_dim, self.hidden_dim)

    def _scaled_dp_attention(self, query, key, value):
        """ calculate scaled attended output of values

        Args:
            query: tensor of [batch_size, batch_size, *, query_num, key_dim]
            key: tensor of [batch_size, *, key_num, key_dim]
            value: tensor of [batch_size, *, key_num, value_dim]

        Returns:
            attn_output: tensor of [batch_size, *, query_num, value_dim]
        """

        # make sure dimension matches
        assert query.shape[-1] == key.shape[-1]
        key = key.transpose(-2, -1)

        attn_weights = torch.matmul(query, key)/math.sqrt(query.shape[-1])
        attn_weights = self.softmax(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output

    def _self_attention(self, input, head_idx):
        """ apply self attention of head#idx over input tensor

        Args:
            input: tensor of [batch_size, *, embedding_dim]
            head_idx: interger of attention head index

        Returns:
            self_attn_output: tensor of [batch_size, *, value_dim]
        """
        query = self.queryProjects[head_idx](input)
        attn_output = self._scaled_dp_attention(query, input, input)
        self_attn_output = self.valueProjects[head_idx](attn_output)
        return self_attn_output

    def _multi_head_self_attention(self, input):
        """ apply multi-head self attention over input tensor

        Args:
            input: tensor of [batch_size, *, signal_length, repr_dim]

        Returns:
            additive_attn_repr: tensor of [batch_size, *, repr_dim]
            multi_head_self_attn_value: tensor of [batch_size, *, signal_length, repr_dim]

        """
        self_attn_outputs = [self._self_attention(input, i) for i in range(self.head_num)]
        mha_embedding_native = torch.cat(self_attn_outputs, dim=-1)
        mha_embedding = self.valueProject(mha_embedding_native)

        # project the embedding of each words to query subspace
        # keep the original embedding of each words as values
        multi_head_self_attn_key = torch.tanh(
            self.keyProject(mha_embedding_native))
        mha_repr = self._scaled_dp_attention(
            self.query, multi_head_self_attn_key, mha_embedding_native).squeeze(dim=-2)
        return mha_embedding, mha_repr

    def forward(self, news_batch, **kwargs):
        """ encode news

        Args:
            news_batch: batch of news tokens, of size [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, level, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        news_embedding_pretrained = self.DropOut(self.embedding(news_batch))

        news_embedding, news_repr = self._multi_head_self_attention(
            news_embedding_pretrained)
        return news_embedding.view(news_batch.shape + (self.level, self.hidden_dim)), news_repr


# class FIM_Encoder(nn.Module):
#     def __init__(self, hparams, vocab):
#         super().__init__()
#         self.name = 'fim-encoder'
#
#         self.kernel_size = 3
#         self.level = 3
#
#         # concatenate category embedding and subcategory embedding
#         self.hidden_dim = hparams['filter_num']
#         self.embedding_dim = hparams['embedding_dim']
#
#         # pretrained embedding
#         self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=False)
#
#         # elements in the slice along dim will sum up to 1
#         self.softmax = nn.Softmax(dim=-1)
#
#         self.ReLU = nn.ReLU()
#         self.LayerNorm = nn.LayerNorm(self.hidden_dim)
#         self.DropOut = nn.Dropout(p=hparams['dropout_p'])
#
#         self.query_words = nn.Parameter(torch.randn(
#             (1, self.hidden_dim), requires_grad=True))
#         self.query_levels = nn.Parameter(torch.randn(
#             (1, self.hidden_dim), requires_grad=True))
#
#         self.CNN_d1 = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.hidden_dim,
#                                 kernel_size=self.kernel_size, dilation=1, padding=1)
#         self.CNN_d2 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim,
#                                 kernel_size=self.kernel_size, dilation=2, padding=2)
#         self.CNN_d3 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim,
#                                 kernel_size=self.kernel_size, dilation=3, padding=3)
#
#         self.SeqCNN3D = nn.Sequential(
#             nn.Conv3d(in_channels=self.level, out_channels=32,
#                       kernel_size=[3, 3, 3], padding=1),
#             nn.ReLU(),
#             nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3]),
#             nn.Conv3d(in_channels=32, out_channels=16,
#                       kernel_size=[3, 3, 3], padding=1),
#             nn.ReLU(),
#             nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3])
#         )
#
#     def _scaled_dp_attention(self, query, key, value):
#         """ calculate scaled attended output of values
#
#         Args:
#             query: tensor of [batch_size, batch_size, *, query_num, key_dim]
#             key: tensor of [batch_size, *, key_num, key_dim]
#             value: tensor of [batch_size, *, key_num, value_dim]
#
#         Returns:
#             attn_output: tensor of [batch_size, *, query_num, value_dim]
#         """
#
#         # make sure dimension matches
#         assert query.shape[-1] == key.shape[-1]
#         key = key.transpose(-2, -1)
#
#         attn_weights = torch.matmul(query, key)/math.sqrt(query.shape[-1])
#         attn_weights = self.softmax(attn_weights)
#
#         attn_output = torch.matmul(attn_weights, value)
#         return attn_output
#
#     def _HDC(self, news_embedding_set):
#         """ stack 1d CNN with dilation rate expanding from 1 to 3
#
#         Args:
#             news_embedding_set: tensor of [set_size, signal_length, embedding_dim]
#
#         Returns:
#             news_embedding_dilations: tensor of [set_size, levels(3), signal_length, filter_num]
#         """
#
#         # don't know what d_0 meant in the original paper
#         news_embedding_dilations = torch.zeros(
#             (news_embedding_set.shape[0], news_embedding_set.shape[1], self.level, self.hidden_dim), device=news_embedding_set.device)
#
#         news_embedding_set = news_embedding_set.transpose(-2, -1)
#
#         news_embedding_d1 = self.CNN_d1(news_embedding_set)
#         news_embedding_d1 = self.LayerNorm(news_embedding_d1.transpose(-2, -1))
#         news_embedding_dilations[:, :, 0, :] = self.ReLU(news_embedding_d1)
#
#         news_embedding_d2 = self.CNN_d2(news_embedding_d1.transpose(-2, -1))
#         news_embedding_d2 = self.LayerNorm(news_embedding_d2.transpose(-2, -1))
#         news_embedding_dilations[:, :, 1, :] = self.ReLU(news_embedding_d2)
#
#         news_embedding_d3 = self.CNN_d3(news_embedding_d2.transpose(-2, -1))
#         news_embedding_d3 = self.LayerNorm(news_embedding_d3.transpose(-2, -1))
#         news_embedding_dilations[:, :, 2, :] = self.ReLU(news_embedding_d3)
#
#         return news_embedding_dilations
#
#     def forward(self, news_batch, **kwargs):
#         """ encode set of news to news representation
#
#         Args:
#             news_batch: tensor of [batch_size, *, signal_length]
#
#         Returns:
#             news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, level, hidden_dim]
#             news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
#         """
#         news_embedding = self.DropOut(
#             self.embedding(news_batch)).view(-1, news_batch.shape[2], self.embedding_dim)
#         news_embedding_dilations = self._HDC(news_embedding).view(
#             news_batch.shape + (self.level, self.hidden_dim))
#         news_embedding_attn = self._scaled_dp_attention(
#             self.query_levels, news_embedding_dilations, news_embedding_dilations).squeeze(dim=-2)
#         news_reprs = self._scaled_dp_attention(self.query_words, news_embedding_attn, news_embedding_attn).squeeze(
#             dim=-2).view(news_batch.shape[0], news_batch.shape[1], self.hidden_dim)
#         return news_embedding_dilations, news_reprs

class FIMG1_Encoder(nn.Module):
    def __init__(self, hparams, vocab):
        super().__init__()
        self.name =  hparams['name']
        self.kernel_size = 3
        self.level = 3

        # concatenate category embedding and subcategory embedding
        self.hidden_dim = hparams['filter_num']
        self.embedding_dim = hparams['embedding_dim']

        # pretrained embedding
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=False)

        # elements in the slice along dim will sum up to 1
        self.softmax = nn.Softmax(dim=-1)

        self.ReLU = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(self.hidden_dim)
        self.DropOut = nn.Dropout(p=hparams['dropout_p'])

        self.query_words = nn.Parameter(torch.randn(
            (1, self.hidden_dim), requires_grad=True))
        self.query_levels = nn.Parameter(torch.randn(
            (1, self.hidden_dim), requires_grad=True))

        self.CNN_d0 = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.hidden_dim,
                                kernel_size=1)
        self.CNN_d1 = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size, dilation=1, padding=1)
        self.CNN_d2 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size, dilation=2, padding=2)
        self.CNN_d3 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size, dilation=3, padding=3)

        self.SeqCNN3D = nn.Sequential(
            nn.Conv3d(in_channels=self.level, out_channels=32,
                      kernel_size=[3, 3, 3], padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3]),
            nn.Conv3d(in_channels=32, out_channels=16,
                      kernel_size=[3, 3, 3], padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3, 3, 3], stride=[3, 3, 3])
        )

    def _scaled_dp_attention(self, query, key, value):
        """ calculate scaled attended output of values

        Args:
            query: tensor of [batch_size, batch_size, *, query_num, key_dim]
            key: tensor of [batch_size, *, key_num, key_dim]
            value: tensor of [batch_size, *, key_num, value_dim]

        Returns:
            attn_output: tensor of [batch_size, *, query_num, value_dim]
        """

        # make sure dimension matches
        assert query.shape[-1] == key.shape[-1]
        key = key.transpose(-2, -1)

        attn_weights = torch.matmul(query, key)/math.sqrt(query.shape[-1])
        attn_weights = self.softmax(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output

    def _HDC(self, news_embedding_set):
        """ stack 1d CNN with dilation rate expanding from 1 to 3

        Args:
            news_embedding_set: tensor of [set_size, signal_length, embedding_dim]

        Returns:
            news_embedding_dilations: tensor of [set_size, levels(3), signal_length, filter_num]
        """

        # don't know what d_0 meant in the original paper
        news_embedding_dilations = torch.zeros(
            (news_embedding_set.shape[0], news_embedding_set.shape[1], self.level, self.hidden_dim), device=news_embedding_set.device)

        news_embedding_set = news_embedding_set.transpose(-2,-1)
        # print(news_embedding_set.size()) torch.Size([75, 300, 20])

        news_embedding_d1 = self.CNN_d1(news_embedding_set)
        news_embedding_d1 = self.LayerNorm(news_embedding_d1.transpose(-2,-1))
        news_embedding_dilations[:,:,0,:] = self.ReLU(news_embedding_d1)


        news_embedding_d2 = self.CNN_d2(news_embedding_dilations[:,:,0,:].transpose(-2,-1))
        news_embedding_d2 = self.LayerNorm(news_embedding_d2.transpose(-2,-1))
        news_embedding_dilations[:,:,1,:] = self.ReLU(news_embedding_d2)

        news_embedding_d3 = self.CNN_d3(news_embedding_dilations[:,:,1,:] .transpose(-2,-1))
        news_embedding_d3 = self.LayerNorm(news_embedding_d3.transpose(-2,-1))
        news_embedding_dilations[:, :, 2, :]= self.ReLU(news_embedding_d3)
        # print(news_embedding_dilations[:,:,2,:].size()) torch.Size([75, 20, 256])

        #新改动
        news_embedding_d0 = self.CNN_d0(news_embedding_set).transpose(-2,-1)
        news_embedding_dilations[:, :, 2, :] = self.ReLU(news_embedding_d0 + news_embedding_dilations[:, :, 2, :])



        return news_embedding_dilations

    def forward(self, news_batch, **kwargs):
        """ encode set of news to news representation

        Args:
            news_batch: tensor of [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, level, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        news_embedding = self.DropOut(
            self.embedding(news_batch)).view(-1, news_batch.shape[2], self.embedding_dim)
        news_embedding_dilations = self._HDC(news_embedding).view(
            news_batch.shape + (self.level, self.hidden_dim))

       #  # print(news_embedding_dilations.size()) #[5,5,20,3,150]
       #  news_embedding_attn = self._scaled_dp_attention(
       #      self.query_levels, news_embedding_dilations, news_embedding_dilations).squeeze(dim=-2)
       # # print(news_embedding_attn.size()) #[5,5,20,150]
       #
       #  news_reprs = self._scaled_dp_attention(self.query_words, news_embedding_attn, news_embedding_attn).squeeze(
       #      dim=-2).view(news_batch.shape[0], news_batch.shape[1], self.hidden_dim)
        return news_embedding_dilations



class FIMG_Encoder(nn.Module):
    def __init__(self, hparams, vocab):
        super().__init__()
        self.name = hparams['name']

        self.kernel_size = 3
        self.level = 3

        # concatenate category embedding and subcategory embedding
        self.hidden_dim = hparams['filter_num']
        self.embedding_dim = hparams['embedding_dim']

        # pretrained embedding
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=False)

        # elements in the slice along dim will sum up to 1
        self.softmax = nn.Softmax(dim=-1)

        self.ReLU = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(self.hidden_dim)
        self.DropOut = nn.Dropout(p=hparams['dropout_p'])

        self.query_words = nn.Parameter(torch.randn(
            (1, self.hidden_dim), requires_grad=True))
        self.query_levels = nn.Parameter(torch.randn(
            (1, self.hidden_dim), requires_grad=True))

        self.CNN_d0 = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.hidden_dim,kernel_size=1)

        self.CNN_d1 = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size, dilation=1, padding=1)
        self.CNN_d2 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size, dilation=2, padding=2)
        self.CNN_d3 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size, dilation=3, padding=3)


    def _scaled_dp_attention(self, query, key, value):
        """ calculate scaled attended output of values

        Args:
            query: tensor of [batch_size, batch_size, *, query_num, key_dim]
            key: tensor of [batch_size, *, key_num, key_dim]
            value: tensor of [batch_size, *, key_num, value_dim]

        Returns:
            attn_output: tensor of [batch_size, *, query_num, value_dim]
        """

        # make sure dimension matches
        assert query.shape[-1] == key.shape[-1]
        key = key.transpose(-2, -1)

        attn_weights = torch.matmul(query, key)/math.sqrt(query.shape[-1])
        attn_weights = self.softmax(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output

    def _HDC(self, news_embedding_set):
        """ stack 1d CNN with dilation rate expanding from 1 to 3

        Args:
            news_embedding_set: tensor of [set_size, signal_length, embedding_dim]

        Returns:
            news_embedding_dilations: tensor of [set_size, levels(3), signal_length, filter_num]
        """

        news_embedding_set = news_embedding_set.transpose(-2,-1)
        news_embedding_d0 = self.LayerNorm(F.relu(self.CNN_d0(news_embedding_set)).transpose(-1, -2))


        news_embedding_d1 = self.LayerNorm(F.relu(self.CNN_d1(news_embedding_set)).transpose(-1, -2)) ### batch, wordsize,150,
        news_embedding_d2 = self.LayerNorm(F.relu(self.CNN_d2(news_embedding_d1.transpose(-1, -2))).transpose(-1, -2))  ### batch, wordsize,150,
        news_embedding_d3 = self.LayerNorm(F.relu(self.CNN_d3(news_embedding_d2.transpose(-1, -2))).transpose(-1, -2))  ### batch, wordsize,150,






        return news_embedding_d0, news_embedding_d1, news_embedding_d2, news_embedding_d3

    def forward(self, news_batch, **kwargs):
        """ encode set of news to news representation

        Args:
            news_batch: tensor of [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, level, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        news_embedding = self.DropOut(
            self.embedding(news_batch)).view(-1, news_batch.shape[2], self.embedding_dim)
        news0, news1,news2,news3= self._HDC(news_embedding)
         #torch.Size([15, 5, 20, 1, 256])
        news0 = news0.view(news_batch.shape + (1, self.hidden_dim))
        news1 = news1.view(news_batch.shape + (1, self.hidden_dim))
        news2 = news2.view(news_batch.shape + (1, self.hidden_dim))
        news3 = news3.view(news_batch.shape + (1, self.hidden_dim))

        return news0, news1,news2, news3


class NRGMS_Encoder(nn.Module):
    def __init__(self, hparams, vocab, user_num):
        super().__init__()  # 原始
        self.name = 'npa-encoder'

        self.dropout_p = hparams['dropout_p']

        self.level = 1
        self.hidden_dim = hparams['filter_num']
        self.embedding_dim = hparams['embedding_dim']
        self.user_dim = hparams['user_dim']
        self.query_dim = hparams['preference_dim']

        # pretrained embedding
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=False)
        # elements in the slice along dim will sum up to 1
        self.softmax = nn.Softmax(dim=-1)

        # trainable lookup layer for user embedding, important to have len(uid2idx) + 1 rows because user indexes start from 1
        self.user_embedding = nn.Embedding(user_num + 1, self.user_dim)
        self.user_embedding.weight.requires_grad = True

        # project e_u to word query preference vector of query_dim
        self.wordPrefProject = nn.Linear(self.user_dim, self.query_dim)
        # project preference query to vector of hidden_dim
        self.wordQueryProject = nn.Linear(self.query_dim, self.hidden_dim)

        # input tensor shape is [batch_size,in_channels,signal_length]
        # in_channels is the length of embedding, out_channels indicates the number of filters, signal_length is the length of title
        # set paddings=1 to get the same length of title, referring M in the paper
        self.CNN = nn.Conv1d(in_channels=self.embedding_dim,
                             out_channels=self.hidden_dim, kernel_size=1)

        self.RELU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.DropOut = nn.Dropout(p=hparams['dropout_p'])

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

    def forward(self, news_batch, **kwargs):
        """ encode news through 1-d CNN and combine embeddings with personalized attention

        Args:
            news_batch: tensor of [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        e_u = self.DropOut(self.user_embedding(kwargs['user_index']))
        word_query = self.Tanh(self.wordQueryProject(
            self.RELU(self.wordPrefProject(e_u))))

        news_embedding_pretrained = self.DropOut(self.embedding(
            news_batch)).view(-1, news_batch.shape[-1], self.embedding_dim).transpose(-2, -1)

        news_embedding = self.RELU(self.CNN(
            news_embedding_pretrained)).transpose(-2, -1).view(news_batch.shape + (self.hidden_dim,))

        return news_embedding, news_embedding_pretrained


class NPA_Encoder(nn.Module):
    def __init__(self, hparams, vocab, user_num):
        super().__init__()
        self.name = 'npa-encoder'
        self.dropout_p = hparams['dropout_p']
        self.level = 1
        self.hidden_dim = hparams['filter_num']
        self.embedding_dim = hparams['embedding_dim']
        self.user_dim = hparams['user_dim']
        self.query_dim = hparams['preference_dim']

        # pretrained embedding
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors,freeze=False)

        # elements in the slice along dim will sum up to 1
        self.softmax = nn.Softmax(dim=-1)

        # trainable lookup layer for user embedding, important to have len(uid2idx) + 1 rows because user indexes start from 1
        self.user_embedding = nn.Embedding(user_num + 1, self.user_dim)
        self.user_embedding.weight.requires_grad = True

        # project e_u to word query preference vector of query_dim
        self.wordPrefProject = nn.Linear(self.user_dim, self.query_dim)
        # nn.init.xavier_normal_(self.wordPrefProject.weight)
        # project preference query to vector of hidden_dim

        self.wordQueryProject = nn.Linear(self.query_dim, self.hidden_dim)
        # nn.init.xavier_normal_(self.wordQueryProject.weight)

        # input tensor shape is [batch_size,in_channels,signal_length]
        # in_channels is the length of embedding, out_channels indicates the number of filters, signal_length is the length of title
        # set paddings=1 to get the same length of title, referring M in the paper

        self.CNN = nn.Conv1d(
            in_channels=self.embedding_dim,
            out_channels=self.hidden_dim,
            kernel_size=3,
            padding=1
        )
        # nn.init.xavier_normal_(self.CNN.weight)

        self.RELU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.DropOut = nn.Dropout(p=hparams['dropout_p'])

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

    def forward(self, news_batch, **kwargs):
        """ encode news through 1-d CNN and combine embeddings with personalized attention

        Args:
            news_batch: tensor of [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        e_u = self.DropOut(self.user_embedding(kwargs['user_index']))
        word_query = self.Tanh(self.wordQueryProject(
            self.RELU(self.wordPrefProject(e_u))))

        news_embedding_pretrained = self.DropOut(self.embedding(
            news_batch)).view(-1, news_batch.shape[-1], self.embedding_dim).transpose(-2, -1)

        news_embedding = self.RELU(self.CNN(
            news_embedding_pretrained)).transpose(-2, -1).view(news_batch.shape + (self.hidden_dim,))

        news_repr = self._scaled_dp_attention(word_query.view(
            word_query.shape[0], 1, 1, word_query.shape[-1]), news_embedding, news_embedding).squeeze(dim=-2)
        return news_embedding.view(news_batch.shape + (self.level, self.hidden_dim)), news_repr


class NPAU_Encoder(nn.Module):
    def __init__(self, hparams, vocab, user_num):
        super().__init__()
        self.name = 'npa-encoder'
        self.dropout_p = hparams['dropout_p']
        self.level = 1
        self.hidden_dim = hparams['filter_num']
        self.embedding_dim = hparams['embedding_dim']
        self.user_dim = hparams['user_dim']
        self.query_dim = hparams['preference_dim']

        # pretrained embedding
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=False)

        # elements in the slice along dim will sum up to 1
        self.softmax = nn.Softmax(dim=-1)

        # trainable lookup layer for user embedding, important to have len(uid2idx) + 1 rows because user indexes start from 1
        self.user_embedding = nn.Embedding(user_num + 1, self.user_dim)
        self.user_embedding.weight.requires_grad = True


        self.wordPrefProject = nn.Linear(self.user_dim, self.query_dim)
        self.wordQueryProject = nn.Linear(self.query_dim, self.hidden_dim)

        self.CNN = nn.Conv1d(in_channels=self.embedding_dim,out_channels=self.hidden_dim, kernel_size=3,padding=1)


        self.RELU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.DropOut = nn.Dropout(p=hparams['dropout_p'])

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

    def forward(self, news_batch, **kwargs):
        """ encode news through 1-d CNN and combine embeddings with personalized attention

        Args:
            news_batch: tensor of [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        e_u = self.DropOut(self.user_embedding(kwargs['user_index']))
        word_query = self.Tanh(self.wordQueryProject(self.RELU(self.wordPrefProject(e_u))))

        news_embedding_pretrained = self.DropOut(self.embedding(
            news_batch)).view(-1, news_batch.shape[-1], self.embedding_dim).transpose(-2, -1)
        # print(news_embedding_pretrained.size())[150,300,20]
        news_embedding = self.RELU(self.CNN(
            news_embedding_pretrained)).transpose(-2, -1).view(news_batch.shape + (self.hidden_dim,))

        news_repr = self._scaled_dp_attention(word_query.view(
            word_query.shape[0], 1, 1, word_query.shape[-1]), news_embedding, news_embedding).squeeze(dim=-2)
        return news_embedding.view(news_batch.shape + (self.level, self.hidden_dim)), news_repr, e_u, news_embedding_pretrained


class NPLM_Encoder(nn.Module):
    def __init__(self, hparams, vocab, user_num):
        super().__init__()
        self.name = 'npa-encoder'
        self.kernel_size = 3
        self.level = 3
        self.dropout_p = hparams['dropout_p']

        self.ReLU = nn.ReLU()
        self.hidden_dim = hparams['filter_num']
        self.embedding_dim = hparams['embedding_dim']
        self.user_dim = hparams['user_dim']
        self.query_dim = hparams['preference_dim']

        # pretrained embedding
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=False)
        # elements in the slice along dim will sum up to 1
        self.softmax = nn.Softmax(dim=-1)
        self.LayerNorm = nn.LayerNorm(self.hidden_dim)
        # trainable lookup layer for user embedding, important to have len(uid2idx) + 1 rows because user indexes start from 1
        self.user_embedding = nn.Embedding(user_num + 1, self.user_dim)
        self.user_embedding.weight.requires_grad = True

        # project e_u to word query preference vector of query_dim
        self.wordPrefProject = nn.Linear(self.user_dim, self.query_dim)
        # project preference query to vector of hidden_dim
        self.wordQueryProject = nn.Linear(self.query_dim, self.hidden_dim)

        # input tensor shape is [batch_size,in_channels,signal_length]
        # in_channels is the length of embedding, out_channels indicates the number of filters, signal_length is the length of title
        # set paddings=1 to get the same length of title, referring M in the paper
        self.CNN = nn.Conv1d(in_channels=self.embedding_dim,
                             out_channels=self.hidden_dim, kernel_size=1)

        self.CNN_b1 = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size, dilation=1, padding=1)
        self.CNN_b2 = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size, dilation=2, padding=2)
        self.CNN_b3 = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size, dilation=3, padding=3)

        self.RELU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.DropOut = nn.Dropout(p=hparams['dropout_p'])

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

    def _HDC(self, news_embedding_set):
        """ stack 1d CNN with dilation rate expanding from 1 to 3

        Args:
            news_embedding_set: tensor of [set_size, signal_length, embedding_dim]

        Returns:
            news_embedding_dilations: tensor of [set_size, levels(3), signal_length, filter_num]
        """

        # don't know what d_0 meant in the original paper
        news_embedding_dilations = torch.zeros(
            (news_embedding_set.shape[0], news_embedding_set.shape[1], self.level, self.hidden_dim), device=news_embedding_set.device)

        news_embedding_set = news_embedding_set.transpose(-2,-1)

        news_embedding_d1 = self.CNN_b1(news_embedding_set)
        news_embedding_d1 = self.LayerNorm(news_embedding_d1.transpose(-2,-1))
        news_embedding_dilations[:,:,0,:] = self.ReLU(news_embedding_d1)

        news_embedding_d2 = self.CNN_b2(news_embedding_set)
        news_embedding_d2 = self.LayerNorm(news_embedding_d2.transpose(-2,-1))
        news_embedding_dilations[:,:,1,:] = self.ReLU(news_embedding_d2)

        news_embedding_d3 = self.CNN_b3(news_embedding_set)
        news_embedding_d3 = self.LayerNorm(news_embedding_d3.transpose(-2,-1))
        news_embedding_dilations[:,:,2,:] = self.ReLU(news_embedding_d3)

        return news_embedding_dilations

    def forward(self, news_batch, **kwargs):
        """ encode news through 1-d CNN and combine embeddings with personalized attention

        Args:
            news_batch: tensor of [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        e_u = self.DropOut(self.user_embedding(kwargs['user_index']))
        word_query = self.Tanh(self.wordQueryProject(
            self.RELU(self.wordPrefProject(e_u))))

        news_embedding_pre = self.DropOut(self.embedding(news_batch))
        news_embedding_pretrained = news_embedding_pre.view(-1, news_batch.shape[-1], self.embedding_dim).transpose(-2, -1)

        news_embedding = self.RELU(self.CNN(
            news_embedding_pretrained)).transpose(-2, -1).view(news_batch.shape + (self.hidden_dim,))

        # #新添加
        # news_embedding_FIM = news_embedding_pre.view(-1, news_batch.shape[2], self.embedding_dim)
        # news_embedding_dilations = self._HDC(news_embedding_FIM).view(news_batch.shape + (self.level, self.hidden_dim))
        # #新添加结束




        return news_embedding, news_embedding_pretrained

class NPLM_Encoder1(nn.Module):
    def __init__(self, hparams, vocab, user_num):
        super().__init__()
        self.name = 'npa-encoder'

        self.dropout_p = hparams['dropout_p']

        self.level = 1
        self.hidden_dim = hparams['filter_num']
        self.embedding_dim = hparams['embedding_dim']
        self.user_dim = hparams['user_dim']
        self.query_dim = hparams['preference_dim']

        # pretrained embedding
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=False)
        # elements in the slice along dim will sum up to 1
        self.softmax = nn.Softmax(dim=-1)

        # trainable lookup layer for user embedding, important to have len(uid2idx) + 1 rows because user indexes start from 1
        self.user_embedding = nn.Embedding(user_num + 1, self.user_dim)
        self.user_embedding.weight.requires_grad = True

        # project e_u to word query preference vector of query_dim
        self.wordPrefProject = nn.Linear(self.user_dim, self.query_dim)
        # project preference query to vector of hidden_dim
        self.wordQueryProject = nn.Linear(self.query_dim, self.hidden_dim)

        # input tensor shape is [batch_size,in_channels,signal_length]
        # in_channels is the length of embedding, out_channels indicates the number of filters, signal_length is the length of title
        # set paddings=1 to get the same length of title, referring M in the paper
        self.CNN = nn.Conv1d(in_channels=self.embedding_dim,
                             out_channels=self.hidden_dim, kernel_size=1)

        self.RELU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.DropOut = nn.Dropout(p=hparams['dropout_p'])

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

    def forward(self, news_batch, **kwargs):
        """ encode news through 1-d CNN and combine embeddings with personalized attention

        Args:
            news_batch: tensor of [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        e_u = self.DropOut(self.user_embedding(kwargs['user_index']))
        word_query = self.Tanh(self.wordQueryProject(
            self.RELU(self.wordPrefProject(e_u))))

        news_embedding_pretrained = self.DropOut(self.embedding(
            news_batch)).view(-1, news_batch.shape[-1], self.embedding_dim).transpose(-2, -1)

        news_embedding = self.RELU(self.CNN(
            news_embedding_pretrained)).transpose(-2, -1).view(news_batch.shape + (self.hidden_dim,))

        # news_embedding_fin = self.RELU(self.CNN_fin(
        #     news_embedding_pretrained)).transpose(-2, -1).view(news_batch.shape + (self.embedding_dim,))
        return news_embedding,news_embedding_pretrained


class Pipeline_Encoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.name = 'pipeline-encoder'

        news_repr_path = 'data/tensors/news_repr_{}_{}-[{}].tensor'.format(hparams['scale'],hparams['mode'],hparams['pipeline'])
        news_embedding_path = 'data/tensors/news_embedding_{}_{}-[{}].tensor'.format(hparams['scale'],hparams['mode'],hparams['pipeline'])

        if os.path.exists(news_repr_path) and os.path.exists(news_embedding_path):
            self.news_repr = nn.Embedding.from_pretrained(torch.load(news_repr_path), freeze=True)
            news_embedding = torch.load(news_embedding_path)
            self.news_embedding = nn.Embedding.from_pretrained(news_embedding.view(news_embedding.shape[0],-1), freeze=True)
        else:
            logger = logging.getLogger(__name__)
            logger.warning("No encoded news at '{}', please encode news first!".format(news_embedding_path))
            raise ValueError

        # print(self.news_repr.weight.shape, news_embedding.shape, self.news_embedding.weight.shape)

        self.level = news_embedding.shape[-2]
        self.hidden_dim = news_embedding.shape[-1]
        # print(self.level, self.hidden_dim)

    def forward(self,news_batch,**kwargs):
        """ encode news by lookup table

        Args:
            news_batch: tensor of [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        news_repr = self.news_repr(kwargs['news_id'])
        news_embedding = self.news_embedding(kwargs['news_id']).view(news_batch.shape + (self.level, self.hidden_dim))
        return news_embedding, news_repr


class MHA_Encoder(nn.Module):
    def __init__(self, hparams, vocab):
        super().__init__()
        self.name = 'mha-encoder'

        self.level = 1

        self.embedding_dim = hparams['embedding_dim']
        self.value_dim = hparams['value_dim']
        self.query_dim = hparams['query_dim']
        self.head_num = hparams['head_num']

        # dimension for the final output embedding/representation
        self.hidden_dim = self.value_dim * self.head_num

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=False)

        self.softmax = nn.Softmax(dim=-1)

        self.query = nn.Parameter(torch.randn(
            (1, self.query_dim), requires_grad=True))

        self.MHA = nn.MultiheadAttention(self.hidden_dim, self.head_num, dropout=hparams['dropout_p'], kdim=self.embedding_dim, vdim=self.embedding_dim)
        self.queryProject = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.keyProject = nn.Linear(self.hidden_dim, self.query_dim)
        self.valueProject = nn.Linear(self.hidden_dim, self.hidden_dim)

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
        attn_weights = self.softmax(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output

    def forward(self, news_batch, **kwargs):
        """ encode news with built-in multihead attention

        Args:
            news_batch: batch of news tokens, of size [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, level, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        news_embedding_pretrained = self.embedding(news_batch)

        # news_embedding, news_repr = self._multi_head_self_attention(
        #     news_embedding_pretrained)
        query = self.queryProject(news_embedding_pretrained).view(-1,news_batch.shape[2],self.hidden_dim).transpose(0,1)
        key = news_embedding_pretrained.view(-1,news_batch.shape[2],self.embedding_dim).transpose(0,1)
        value = key

        news_embedding,_ = self.MHA(query, key, value)
        news_embedding = news_embedding.transpose(0,1).view(news_batch.shape + (self.hidden_dim,))

        multi_head_self_attn_key = torch.tanh(self.keyProject(news_embedding))
        news_repr = self._scaled_dp_attention(
            self.query, multi_head_self_attn_key, news_embedding).squeeze(dim=-2)

        return news_embedding.view(news_batch.shape + (self.level, self.hidden_dim)), news_repr

class Bert_Encoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.name = 'bert-encoder'

        self.level = hparams['level']

        # dimension for the final output embedding/representation
        self.hidden_dim = 768

        self.bert = AutoModel.from_pretrained(
            hparams['bert'],
            # output hidden embedding of each transformer layer
            output_hidden_states=True
        )

    def forward(self, news_batch, **kwargs):
        """ encode news with bert

        Args:
            news_batch: batch of news tokens, of size [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, level, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        output = self.bert(news_batch.view(-1, news_batch.shape[2]), attention_mask=kwargs['attn_mask'].view(-1, news_batch.shape[2]))
        # stack the last level dimensions to form multi-level embedding
        news_embedding = torch.stack(output['hidden_states'][-self.level:],dim=-2)
        # use the hidden state of the last layer of [CLS] as news representation
        news_repr = news_embedding[:,0,-1,:].view(news_batch.shape[0], news_batch.shape[1], self.hidden_dim)

        return news_embedding.view(news_batch.shape + (self.level, self.hidden_dim)), news_repr

class Encoder_Wrapper(nn.Module):
    def __init__(self, hparams, encoder):
        super().__init__()
        self.encoder = encoder
        self.name = 'pipeline-'+encoder.name

        self.device = hparams['device']
    
    def forward(self,x):
        if x['candidate_title'].shape[0] != self.batch_size:
            self.batch_size = x['candidate_title'].shape[0]

        news = x['candidate_title'].long().to(self.device)
        news_embedding, news_repr = self.encoder(
            news,
            news_id=x['news_id'].long().to(self.device),
            attn_mask=x['candidate_title_pad'].to(self.device))

        return news_embedding, news_repr

class RNN_Encoder(nn.Module):
    def __init__(self, hparams, vocab):
        super().__init__()
        self.name = 'bert-encoder'

        self.level = 2

        self.embedding_dim = hparams['embedding_dim']
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors,sparse=True,freeze=False)

        # dimension for the final output embedding/representation
        self.hidden_dim = 200

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True,bidirectional=True)

    def forward(self, news_batch, **kwargs):
        """ encode news with bert
        Args:
            news_batch: batch of news tokens, of size [batch_size, *, signal_length]
        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, level, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """

        # conpress news_num into batch_size
        news_embedding_pretrained = self.embedding(news_batch).view(-1, news_batch.shape[-1], self.embedding_dim)
        news_embedding,output = self.lstm(news_embedding_pretrained)
        news_repr = torch.mean(output[0],dim=0).view(news_batch.shape[0],news_batch.shape[1],self.hidden_dim)

        return news_embedding.view(news_batch.shape + (self.level, self.hidden_dim)), news_repr

#NPA+位置编码信息
class NPA_Encoder1(nn.Module):
    def __init__(self, hparams, vocab, user_num):
        super().__init__()
        self.name = 'npa-encoder'

        self.dropout_p = hparams['dropout_p']

        self.level = 1
        self.hidden_dim = hparams['filter_num']
        self.embedding_dim = hparams['embedding_dim']
        self.user_dim = hparams['user_dim']
        self.query_dim = hparams['preference_dim']
        self.LayerNorm = nn.LayerNorm(self.embedding_dim)

        # pretrained embedding
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=False)
        # elements in the slice along dim will sum up to 1
        self.softmax = nn.Softmax(dim=-1)

        # trainable lookup layer for user embedding, important to have len(uid2idx) + 1 rows because user indexes start from 1
        self.user_embedding = nn.Embedding(user_num + 1, self.user_dim)
        self.user_embedding.weight.requires_grad = True

        # project e_u to word query preference vector of query_dim
        self.wordPrefProject = nn.Linear(self.user_dim, self.query_dim)
        # project preference query to vector of hidden_dim
        self.wordQueryProject = nn.Linear(self.query_dim, self.hidden_dim)

        # input tensor shape is [batch_size,in_channels,signal_length]
        # in_channels is the length of embedding, out_channels indicates the number of filters, signal_length is the length of title
        # set paddings=1 to get the same length of title, referring M in the paper
        self.CNN = nn.Conv1d(in_channels=self.embedding_dim,
                             out_channels=self.hidden_dim, kernel_size=3, padding=1)
        self.RELU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.DropOut = nn.Dropout(p=hparams['dropout_p'])

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

    def PosEncoder(self,x, min_timescale=1.0, max_timescale=1.0e4):  # 位置编码
        x = x.transpose(1, 2)
        length = x.size()[1]
        channels = x.size()[2]
        signal = self.get_timing_signal(length, channels, min_timescale, max_timescale)
        return (x + signal.cuda()).transpose(1, 2)

    def get_timing_signal(self,length, channels, min_timescale=1.0, max_timescale=1.0e4):
        position = torch.arange(length).type(torch.float32)
        num_timescales = channels // 2
        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)  # 用三角函数计算位置编码
        m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
        signal = m(signal)
        signal = signal.view(1, length, channels)
        return signal

    def forward(self, news_batch, **kwargs):
        """ encode news through 1-d CNN and combine embeddings with personalized attention

        Args:bre
            news_batch: tensor of [batch_size, *, signal_length]

        Returns:
            news_embedding: hidden vector of each token in news, of size [batch_size, *, signal_length, hidden_dim]
            news_repr: hidden vector of each news, of size [batch_size, *, hidden_dim]
        """
        e_u = self.DropOut(self.user_embedding(kwargs['user_index']))
        word_query = self.Tanh(self.wordQueryProject(
            self.RELU(self.wordPrefProject(e_u))))

        news_embedding_pretrained = self.DropOut(self.embedding(
            news_batch)).view(-1, news_batch.shape[-1], self.embedding_dim).transpose(-2, -1)

        #print(news_embedding_pretrained.size()) [25,300,20]

        news_embedding_pretrained = F.normalize(self.PosEncoder(news_embedding_pretrained),dim=-2)


        news_embedding = self.RELU(self.CNN(
            news_embedding_pretrained)).transpose(-2, -1).view(news_batch.shape + (self.hidden_dim,))

        news_repr = self._scaled_dp_attention(word_query.view(
            word_query.shape[0], 1, 1, word_query.shape[-1]), news_embedding, news_embedding).squeeze(dim=-2)
        return news_embedding.view(news_batch.shape + (self.level, self.hidden_dim)), news_repr
