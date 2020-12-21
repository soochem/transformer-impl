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

# TEST
input_embedding = tokenizer.embed_input(rand_sample, vocab_size, d_model)
# Positional Encoding
input_sums = PositionalEncoding(seq_len, d_model).forward(input_embedding)
# Generate Padding Mask
mask = rand_sample.eq(0).unsqueeze(1).expand(input_sums.size(0), input_sums.size(1), input_sums.size(1))

enc_output = Encoder(vocab_size, d_model, n_head, dropout).encode_layer(input_sums, mask)
