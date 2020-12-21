from model import *


# Hyperparameter
seq_len = 8
vocab_size = 2000
dropout = 0.1
d_model = 128
n_head = 2
# d_head = 64 = d_k

# INPUT
# inputs = tokenizer.encode_sequences('../data/test/test.txt', '../model/test_spm.model')

# Create Random Tensor
# rand_sample = np.random.rand(2, 8, 8)
rand_sample = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0],
                            [1, 2, 3, 4, 5, 6, 7, 8]])
# rand_sample = np.array([[1, 2, 3, 4, 5, 6, 0, 0],
#                         [1, 2, 3, 4, 5, 6, 7, 8]])

# Input Embedding
rand_sample = torch.LongTensor(rand_sample)  # data type for embedding?
input_embedding = tokenizer.embed_input(rand_sample, vocab_size=vocab_size, d_model=d_model)

# Positional Embedding
# x.shape = (seq_len, seq_len) = (2, 8)
# pos_encoding 결과 = (2, 8, 128)
input_sums = PositionalEncoding(seq_len, d_model, dropout).forward(input_embedding)
print(input_sums.size())  # (seq_len, seq_len, d_model) = (2, 8, 128)

# Generate Mask
# unsqueeze => torch.Size([2, 1, 8])
# expand => (Q.size(0), Q.size(1), K.size(1)) => 2, 1, 8 중에 1*8세트 => 2, 8, 8
mask = rand_sample.eq(0).unsqueeze(1).expand(input_sums.size(0), input_sums.size(1), input_sums.size(1))

# Scaled Dot Attention
s = ScaledDotAttention(dropout)
output, attn_w = s.forward(input_sums, input_sums, input_sums, mask)
print(output.size(), attn_w.size())

# 테스트
# 실험 1. query는 key의 2행과 일치한다.
# attn_weight의 2번째 값이 1일 것
# temp_weight: tensor([[0, 1, 0, 0]], dtype=torch.int32)
# temp_output: tensor([[10,  0]], dtype=torch.int32)
temp_q = torch.tensor([[0, 10, 0]], dtype=torch.float32)  # (1, 3)

# 실험 2. query는 key의 3, 4행과 일치한다.
# temp_weight: tensor([[0, 0, 0.5, 0.5]], dtype=torch.int32)
# temp_output: tensor([[550.0000,  5.5000]], dtype=torch.int32)
# 550 <= (100 + 1000) 0.5,  5.5 <= (5 + 6) * 0.5
temp_q = torch.tensor([[0, 0, 10]], dtype=torch.float32)  # (1, 3)

# 실험 3. 3개의 query
temp_q = torch.tensor([[0, 0, 10],
                       [0, 10, 0],
                       [10, 10, 0]], dtype=torch.float32)  # (3, 3)
# 공통의 k, v를 활용하여 실험
temp_k = torch.tensor([[10, 0, 0],
                        [0, 10, 0],
                        [0, 0, 10],
                        [0, 0, 10]], dtype=torch.float32)  # (4, 3)
temp_v = torch.tensor([[1, 0],
                        [10, 0],
                        [100, 5],
                        [1000, 6]], dtype=torch.float32)  # (4, 2)

temp_output, temp_weight = s.forward(temp_q, temp_k, temp_v, None)

# 어텐션 분포
print(temp_weight)
# 어텐션 값
print(temp_output)


## Multi Head Attention
output = MultiHeadAttention(d_model, n_head, dropout).forward(input_sums, input_sums, input_sums, mask)
