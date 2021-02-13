import os
import sentencepiece as spm
import pandas as pd
import json


# vocab loading
data_dir = '../data/kowiki'
model_name = 'kowiki'
vocab_file = os.path.join(data_dir, model_name + '.model')
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)
print('vocab size: %d' % vocab.vocab_size())


def prepare_train(vocab, infile, outfile):
    """
    nsmc 데이터를 전처리하는 과정
    """
    df = pd.read_csv(infile, sep="\t", engine="python")
    with open(outfile, "w", encoding='utf-8') as f:
        for index, row in df.iterrows():
            document = row["document"]
            if type(document) != str:
                continue
            instance = {"id": row["id"], "doc": vocab.encode_as_pieces(document), "label": row["label"]}
            f.write(json.dumps(instance))
            f.write("\n")
        print(outfile + ' is written')

data_dir = '../data/nsmc'
train_name = os.path.join(data_dir, 'ratings_train')
test_name = os.path.join(data_dir, 'ratings_test')
# prepare_train(vocab, train_name + '.txt', train_name+'.json')
prepare_train(vocab, test_name + '.txt', test_name+'.json')
