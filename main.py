# import numpy
import torch
import logging
import argparse

import data_utils
import dataset
from dataset import FrEnDataset


# logging config
logging.basicConfig(filename='./logs/test.log', level=logging.INFO)

# torch version 1.6.0+cuda 10.1
logging.info('Torch Version {}'.format(torch.__version__))

# test path => TODO argparse
DATA_PATH = './data/fr-en'


if __name__ == '__main__':
    # Load Dataset
    data_utils.load_data_from_file(DATA_PATH, DATA_PATH)
    en_dataset = FrEnDataset(txt_files='./data/fr-en/fr-en.en.txt',
                             root_dir='./data/fr-en/')
    fr_dataset = FrEnDataset(txt_files='./data/fr-en/fr-en.fr.txt',
                             root_dir='./data/fr-en/')

    en_loader = dataset.DataLoader(en_dataset, batch_size=4,
                                   shuffle=True, num_workers=1)



    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch, sample_batched.size())
    #
    #     # observe 4th batch and stop.
    #     if i_batch == 3:
    #         show_text_batch(sample_batched)
    #         break

    # Train

    # Test
