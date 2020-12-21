import torch
import torch.nn as nn
import numpy as np

import tokenizer


class TransformerModel(nn.Module):
    def __init__(self, seq_len, d_model, n_head, n_hidden, n_layers, vocab_size, dropout=0.1):
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.vocab_size = vocab_size

        self.pos_encoder = PositionalEncoding(seq_len, d_model, dropout)
        self.init_weights()

    def init_weights(self):
        pass

    def encode(self, inputs):
        word_embedding = tokenizer.embed_input(self.vocab_size, self.d_model, inputs)
        input_encoded = self.pos_encoder.forward(word_embedding)

        # TODO use encoder

        return input_encoded

    def forward(self):
        pass


class PositionalEncoding:
    def __init__(self, seq_len, d_model, dropout=0.1):
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


class Encoder(nn.Module):
    """
    Composed of stack N = 6 identical layers.
    * multi-head attention
    * position-wise FC FNN
        => residual connection is applied between two sub-layers.
    * output dim : d_model (512)
    """

    def __init__(self, vocab_size, d_model, n_head, dropout):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def encode_layer(self, inputs, mask):
        """
        :param inputs: Tensor
        :return:
        """
        # Multi Head Attn
        attn_output = MultiHeadAttention(self.d_model, self.n_head).forward(inputs, inputs, inputs, mask)
        # Dropout
        attn_output = self.dropout(attn_output)
        # Residual Connection & Layer Normalization
        attn_output = self.layer_norm(inputs + attn_output)
        # Position-wise FFN
        ff_output = FeedFoward(self.d_model).forward(attn_output)
        # Dropout
        attn_output = self.dropout(ff_output)
        # Residual Connection & Layer Normalization
        attn_output = self.layer_norm(inputs + attn_output)

        return attn_output

    def forward(self):
        """
        :return: TODO layer not tensor
        """
        pass


class Decoder(nn.Module):
    pass


class ScaledDotAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        """
        :param q: query (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        :param k: key (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
        :param v: value (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
        :param mask: (batch_size, 1, 1, key의 문장 길이)
        :return:
        """
        # Q * K^T about each batch
        # transpose(2, 1)로 하면 3차원 이상 텐서에서만 적용 가능함
        mul_qk = torch.matmul(q, k.transpose(-1, -2))
        # Scaling by root(d_k)
        scale_factor = 1/(k.size(-1) ** 0.5)
        mul_qk = mul_qk.mul(scale_factor)

        # Apply mask: padding이 0인 부분을 매우 작은 값으로 변경
        if mask is not None:
            mul_qk.masked_fill(mask, -1e9)
        # Calculate weight prob distribution
        # Size: (batch_size, num_heads, query의 문장 길이, key의 문장 길이)
        attn_weights = nn.Softmax(dim=-1)(mul_qk)  # dim -1 ?, normalized along axis -1

        # Multiply V
        # Size: (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        result = torch.matmul(attn_weights, v)

        return result, attn_weights


class MultiHeadAttention(nn.Module):
    # 여러 번의 어텐션을 병렬로 사용
    # multi-head attn mat = concatenated matrix * w_0
    # d_model = d_v * n_head
    # concatenated matrix : (seq_len, d_model)
    # w_0 = (d_v * n_head, d_model)
    # multi-head attn mat : (seq_len, d_model) => 인풋 행렬의 크기가 유지됨!! (다음 인코더에서 입력으로 사용)
    def __init__(self, d_model, n_head, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.depth = d_model // n_head  # 논문의 512/8 = 64
        self.dropout = nn.Dropout(dropout)
        self.w_0 = nn.Linear(d_model, d_model)  # Dense Layer
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.output_layer = nn.Linear(self.d_model, self.d_model)

    def split_head(self, inputs, batch_size):
        # n_head 수 만큼 q, k, v를 split
        inputs = inputs.view(batch_size, -1, self.n_head, self.depth)
        return torch.transpose(inputs, 1, 2)  # ??

    def forward(self, q, k, v, mask):
        """
        :param q: query (batch_size, query의 문장 길이, d_model)
        :param k: key   (batch_size, key의 문장 길이, d_model)
        :param v: value (batch_size, value의 문장 길이, d_model)
        :param mask: (batch_size, 1, 1, key의 문장 길이)
        :return:
        """
        batch_size = q.size(0)  # Q에서 가져옴
        # Split into Multi Head
        q_dense = self.split_head(self.w_q(q), batch_size)
        k_dense = self.split_head(self.w_k(k), batch_size)
        v_dense = self.split_head(self.w_v(v), batch_size)
        # Scaled Dot Product Attention
        attn_output, _ = ScaledDotAttention().forward(q_dense, k_dense, v_dense, mask)
        # Concat Heads
        concat_output = attn_output.view(batch_size, -1, self.d_model)
        # Pass Output Dense Layer
        output = self.output_layer(concat_output)
        print(output.size())

        return output


class FeedFoward(nn.Module):
    def __init__(self, d_model, d_hidden=2048, dropout=0.1):
        super(FeedFoward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        x = nn.ReLU()(self.linear_1(x))
        x = self.linear_2(x)
        return x
