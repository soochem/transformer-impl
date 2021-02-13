from model import *

import os
import sentencepiece as spm


# Hyperparameter
seq_len = 8
vocab_size = 2000
dropout = 0.1
d_model = 128
n_head = 2
n_layers = 1


# INPUT
#  -------1. fr-en -------
INPUT_FILE_PATHS = '../data/fr-en/fr-en.en.txt,../data/fr-en/fr-en.fr.txt'
# fr_en_inputs = tokenizer.encode_sequences('../data/test/test.txt', '../model/test_spm.model')
fr_en_inputs = tokenizer.encode_sequences(INPUT_FILE_PATHS, '../model/test_spm.model')

# ------- 2. NSMC -------
# vocab loading
data_dir = '../data/kowiki'
model_name = 'kowiki'
vocab_file = os.path.join(data_dir, model_name + '.model')
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)
print('vocab size: %d' % vocab.vocab_size())

vocab_size = vocab.vocab_size()

# 간단한 문장으로 테스트
lines = [
  "와 이 영화 꿀잼 ㅋㅋㅋ",
  "정말 재미 없는 영화로군요.",
  "세기의 명작!",
  "겨울 날씨가 너무 춥죠?",
  "이 원소는 원자량 16를 가집니다. 탄소 보단 무겁습니다.",
  "원자량은 특정 환경 하에서 해당 원소의 동위 원소의 존재 비율에 따라 가중평균한 질량을 말한다. "
]

nsmc_inputs = []
for line in lines:
    pieces = vocab.encode_as_pieces(line)
    ids = vocab.encode_as_ids(line)
    nsmc_inputs.append(torch.tensor(ids))
    print(line)
    print(pieces)
    print(ids)
    print()

nsmc_inputs = nn.utils.rnn.pad_sequence(nsmc_inputs, batch_first=True, padding_value=0)


# ------- 3. Create Random Tensor -------
rand_sample = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0],
                            [1, 2, 3, 4, 5, 6, 7, 8]])
rand_sample = torch.LongTensor(rand_sample)  # data type for embedding?

# 테스트 인풋
# test_input = fr_en_inputs
test_input = nsmc_inputs
# test_input = rand_sample
seq_len = test_input.size(1)

# MODEL
transformer = TransformerModel(seq_len, d_model, n_head, n_layers, vocab_size, dropout=0.1)
output = transformer.forward(test_input)
print('output size: %s' % str(output.size()))

