import sentencepiece as spm
import logging

"""
https://github.com/google/sentencepiece
# pip install "msgpack-numpy<0.4.4.0"
# error
    result = s.encode('New York', out_type=str, enable_sampling=True, alpha=0.1, nbest=-1)
    TypeError: Encode() got an unexpected keyword argument 'nbest'
"""

# Train sentencepiece
templates = '--input={} \
            --pad_id={} \
            --bos_id={} \
            --eos_id={} \
            --unk_id={} \
            --model_prefix={} \
            --vocab_size={} \
            --character_coverage={} \
            --model_type={}'

# hyperparameter 지정
# fr, en을 같이 학습하는지, 따로 하는지?
# input: 리스트 가능 (comma separated string)
# ex. trainer_interface.cc(376) LOG(INFO) Loaded all 3083 sentences
train_input_file = "../data/output/fr-en.en.txt,../data/output/fr-en.fr.txt"
pad_id = 0                 # <pad> token
bos_id = 1                 # <start> token
eos_id = 2                 # <end> token
unk_id = 3                 # <unknown> token
prefix = 'test_spm'        # 저장할 tokenizer 모델 이름
vocab_size = 2000          # Vocab 사이즈
character_coverage = 1.0   # To reduce character set
model_type = 'unigram'     # Choose from unigram (default), bpe, char, or word => 차이점??

# cmd 생성
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

spm.SentencePieceTrainer.Train(cmd)

# Load Model
s = spm.SentencePieceProcessor(model_file='test_spm.model')

# 여러 번 시행하는 이유? deterministic, stochastic 차이
for n in range(5):
    result = s.encode('New York', out_type=str, enable_sampling=True, alpha=0.1)
    print(result)
