import torch
import torch.nn as nn
import numpy as np


class TransformerModel(nn.Module):

    def __init__(self, seq_len, d_model, n_head, n_hidden, n_layers, dropout=0.1):
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(seq_len, d_model, dropout)

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self):
        pass


class PositionalEncoding:
    def __init__(self, seq_len, d_model, dropout):
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def cal_angle(self, pos, i):
        return pos / np.power(10000, 2 * (i // 2) / self.d_model)

    def pos_encoding(self):
        # seq_len의 각 위치(pos)에서 d_model의 각 i에 대해 계산한 angle의 sin, cos 값
        # 1. Calculate angles
        i = np.arange(self.d_model)  # np.arange를 사용해야 다른 dim인 벡터에 대해 연산 가능
        sinusoid_tab = torch.FloatTensor([self.cal_angle(pos, i) for pos in range(self.seq_len)])
        # 2. Calculate sine value of given angle at "even" index
        sinusoid_tab[:, 0::2] = np.sin(sinusoid_tab[:, 0::2])
        # 3. Calculate cosine value of given angle at "odd" index
        sinusoid_tab[:, 1::2] = np.cos(sinusoid_tab[:, 1::2])
        return sinusoid_tab  # (seq_len, d_model)

    def forward(self, x):
        # Input = Input Embedding(x) + Positional Encoding
        # 패딩한 곳에 pe 값 더해도 되는지?
        x += self.pos_encoding()
        return self.dropout(x)
