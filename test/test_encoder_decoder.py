from model import *

# Hyperparameter
seq_len = 8
vocab_size = 2000
dropout = 0.1
d_model = 128
n_head = 2

# INPUT
# inputs = tokenizer.encode_sequences('../data/test/test.txt', '../model/test_spm.model')

# Create Random Tensor
rand_sample = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0],
                            [1, 2, 3, 4, 5, 6, 7, 8]])
rand_sample = torch.LongTensor(rand_sample)  # data type for embedding?


# Generate Padding Mask
# 패딩 마스크를 구현하는 방법은 입력된 정수 시퀀스에서 패딩 토큰의 인덱스인지, 아닌지를 판별하는 함수를 구현하는 것입니다.
# 아래의 함수는 정수 시퀀스에서 0인 경우에는 1로 변환하고, 그렇지 않은 경우에는 0으로 변환하는 함수입니다.
# 둘다 true/false로 하거나 1/0으로 통일해야함
def create_padding_mask(x): #, size):
    # mask = x.eq(0).unsqueeze(1).expand(input_sums.size(0), input_sums.size(1), input_sums.size(1))
    mask = x.eq(0).to(torch.float32)  # bool -> int
    mask = mask.unsqueeze(1).expand(x.size(0), x.size(1), x.size(1))
    return mask


def create_look_ahead_mask(x, mask):
    # 자기 자신 보다 미래에 있는 단어들은 참고하지 못하도록 마스크를 씌운다.
    # seq_len = x.size(1)
    # 마스킹 하려는 위치에 1
    # [[1,0,0], [1,1,0], [1,1,1]]
    # [[0,1,1], [0,0,1], [0,0,0]]
    look_ahead_mask = 1 - torch.tril(torch.ones((x.size(0), x.size(1), x.size(1))))
    # 1, 0 중에 max 값 리턴
    return torch.max(look_ahead_mask, mask)


# TEST
input_embedding = tokenizer.embed_input(rand_sample, vocab_size, d_model)
# Positional Encoding
input_sums = PositionalEncoding(seq_len, d_model).forward(input_embedding)
# Padding mask
pad_mask = create_padding_mask(rand_sample)
print('pad_mask size %s' % str(pad_mask.size()))  # (batch_size, seq_len, seq_len)
# Look ahead mask
look_ahead_mask = create_look_ahead_mask(rand_sample, pad_mask)
print('look_ahead_mask size %s' % str(look_ahead_mask.size()))

enc_output = Encoder(vocab_size, d_model, n_head, dropout).encode_layer(input_sums, pad_mask)
dec_output = Decoder(vocab_size, d_model, n_head, dropout).decode_layer(input_sums, enc_output, pad_mask, look_ahead_mask)
