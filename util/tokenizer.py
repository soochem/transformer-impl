import sentencepiece as spm
import logging
import torch
import torch.nn as nn
import model

INPUT_FILE_PATHS = '../data/output/fr-en.en.txt,../data/output/fr-en.fr.txt'


def train_sentencepiece(train_input_file=INPUT_FILE_PATHS,
                        prefix='test_spm',
                        model_type='unigram'):
    # Train sentencepiece model
    templates = '--input={} \
                --pad_id={} \
                --bos_id={} \
                --eos_id={} \
                --unk_id={} \
                --model_prefix={} \
                --vocab_size={} \
                --character_coverage={} \
                --model_type={}'

    # Assign hyper-parameter
    # fr, en을 같이 학습하는지, 따로 하는지?
    # input: 리스트 가능 (comma separated string)
    # ex. trainer_interface.cc(376) LOG(INFO) Loaded all 3083 sentences
    # train_input_file = "../data/output/fr-en.en.txt,../data/output/fr-en.fr.txt"
    pad_id = 0  # <pad> token
    bos_id = 1  # <start> token
    eos_id = 2  # <end> token
    unk_id = 3  # <unknown> token
    # prefix = 'test_spm'  # 저장할 tokenizer 모델 이름
    vocab_size = 2000  # Vocab 사이즈
    character_coverage = 1.0  # To reduce character set
    # model_type = 'unigram'  # Choose from unigram (default), bpe, char, or word => 차이점??

    # Create cmd
    cmd = templates.format(train_input_file,
                           pad_id,
                           bos_id,
                           eos_id,
                           unk_id,
                           prefix,
                           vocab_size,
                           character_coverage,
                           model_type)
    logging.info(cmd)

    # Train => True
    return spm.SentencePieceTrainer.Train(cmd)


def encode_sequences(d_model,
                     model_file,
                     input_file=INPUT_FILE_PATHS):
    # Load model after training
    if model_file is not None:
        sp = spm.SentencePieceProcessor(model_file='test_spm.model')
    else:
        try:
            sp = spm.SentencePieceProcessor()
        except Exception:
            raise Exception('No trained SentencePieceProcessor!')

    # txt file로 저장된 fr-en.en.txt, fe-en.fr.txt를 토큰 id로 임베딩
    input_file_list = input_file.split(',')
    print(input_file_list)
    i = 0  # 테스트용

    input_data = []
    for path in input_file_list:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if i == 100:  # 테스트용
                    break
                # Get token pieces
                # pieces = sp.encode_as_pieces(line)
                # Get token ids
                ids = sp.encode_as_ids(line)
                # print(pieces)
                # print(ids)
                # sent_n 에 대한 token id 리스트를 tensor로 변환하여 추가
                input_data.append(torch.tensor(ids))
                i += 1

    # Get encoded dataset
    # TODO [sent1, sent2, ...] => [[id1, id2, ...], [id10, id11, ...]]
    # 길이가 다 다르면 어떻게 Input Embedding?
    # => 가장 token 수가 많은 sequence 길이에 맞춰서 연산에 영향을 주지 않는 0으로 padding
    # print(len(input_data), end=', ')  # inputs의 행 길이 (문장 개수)
    # print(max([len(i) for i in input_data]))  # inputs의 열 길이 (최대 개수의 token을 가진 문장의 token 수)
    inputs = nn.utils.rnn.pad_sequence(input_data, batch_first=True, padding_value=0)
    print('Input Size : {}'.format(inputs.size()))
    # logging.info('Input Size : {}'.format(inputs.size()))

    vocab_size = sp.vocab_size()
    embedding = nn.Embedding(vocab_size, d_model)
    inputs_emb = embedding(inputs)
    # 100, 69, 128 (sequence length, 문장 당 최대 token 수, hidden_size)
    print(inputs_emb.size())
    return inputs_emb


# def embed_input(vocab_size, d_model, inputs):
#     embedding = nn.Embedding(vocab_size, d_model)
#     inputs_emb = embedding(inputs)
#     # 100, 69, 128 (sequence length, 문장 당 최대 token 수, hidden_size)
#     print(inputs_emb.size())
#     return inputs_emb

# ---- TEST ---- #
d_model = 128
dropout = 0.1

# # Train Sentencepiece Model
# train_sentencepiece(INPUT_FILE_PATHS)
#
# # Load and Tokenize Datasets from File & Input Embedding
# test_inputs = encode_sequences(d_model, 'test_spm.model', input_file=INPUT_FILE_PATHS)
#
# # Add Positional Encoding
# seq_len = test_inputs.size(1)  # 최대 문장 길이
# test_pe = model.PositionalEncoding(seq_len, d_model, dropout).forward(test_inputs)
