from tokenizer import *
from model import PositionalEncoding

d_model = 128
dropout = 0.1
vocab_size = 2000
INPUT_FILE_PATHS = '../data/fr-en/fr-en.en.txt,../data/fr-en/fr-en.fr.txt'

# Train Sentencepiece Model
train_sentencepiece(INPUT_FILE_PATHS)

# Load and Tokenize Datasets from File
test_inputs = encode_sequences('test_spm.model', input_file=INPUT_FILE_PATHS)
# Input Embedding
test_inputs = embed_input(vocab_size, d_model, test_inputs)

# Add Positional Encoding
seq_len = test_inputs.size(1)  # 최대 문장 길이
test_pe = PositionalEncoding(seq_len, d_model, dropout).forward(test_inputs)
