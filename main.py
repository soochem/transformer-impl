# import numpy
import torch
import logging
import data_loader


# torch version 1.6.0+cuda 10.1
print(torch.__version__)

# logging config
logging.basicConfig(filename='./logs/test.log', level=logging.INFO)

# test path
TAR_PATH = './data/fr-en.tgz'
TAR_URL = "https://wit3.fbk.eu/archive/2017-01-trnted/texts/fr/en/fr-en.tgz"
OUTPUT_PATH = './data/output'
DATA_PATH = './data/fr-en'


if __name__ == '__main__':
    data_loader.load_data_from_file(DATA_PATH, OUTPUT_PATH)
