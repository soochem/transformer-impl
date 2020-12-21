import sentencepiece as spm
# import logging
# import numpy as np
# import torch
import torch.nn as nn

import tokenizer
import model

"""
Reference
- GitHub google/sentencepiece: https://github.com/google/sentencepiece
    - Colab: https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb
- Pytorch Tutorial: https://tutorials.pytorch.kr/beginner/transformer_tutorial.html
# error (nbest 라는 파라미터는 사용하지 않음)
    result = s.encode('New York', out_type=str, enable_sampling=True, alpha=0.1, nbest=-1)
    TypeError: Encode() got an unexpected keyword argument 'nbest'
"""


# Load SP model after training
sp = spm.SentencePieceProcessor(model_file='../model/test_spm.model')
# Get vocab_size (두 가지 방법)
# print(sp.vocab_size())
# print(len(sp))

# Test SentencePiece model
# 여러 번 시행하는 이유? deterministic, stochastic 차이
# <unk>, <s>, </s> are defined by default. Their ids are (0, 1, 2)
# <s> and </s> are defined as 'control' symbol.
for n in range(5):
    result = sp.encode('New York', out_type=str, enable_sampling=True, alpha=0.1)
    print(result)
    result_ids = sp.encode_as_ids('New York')
    print(result_ids)


# Input Embedding
inputs = tokenizer.encode_sequences()
vocab_size = sp.vocab_size()  # vocab count
d_model = 128                 # hidden size
embedding = nn.Embedding(vocab_size, d_model)
inputs_emb = embedding(inputs)
# 100, 69, 128 (sequence length, 문장 당 최대 token 수, hidden_size)
print(inputs_emb.size())


# Positional Encoding
dropout = 0.1
seq_len = inputs_emb.size(1)  # 최대 문장 길이

# pos_encoding => inputs_emb와 같은 차원
# pos_encoding = PositionalEncoding(seq_len, d_model).pos_encoding()
# print(pos_encoding.size())  # (69, 128)
model.PositionalEncoding(seq_len, d_model, dropout).forward(inputs_emb)
