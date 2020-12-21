from dataset import *
from torch.utils.data import DataLoader


def show_text_batch(sample_batched):
    """Show texts for a batch of samples."""
    batch_size = len(sample_batched)

    for i in range(batch_size):
        print(sample_batched[:5])


if __name__ == '__main__':

    fr_en_dataset = FrEnDataset(txt_files='../data/fr-en/fr-en.en.txt',
                                root_dir='../data/fr-en/')

    for i in range(len(fr_en_dataset)):
        sample = fr_en_dataset[i]

        print(i, sample.shape)
        print(sample[:5])

        if i == 3:
            break

    dataloader = DataLoader(fr_en_dataset, batch_size=4,
                            shuffle=True, num_workers=1)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched.size())

        # observe 4th batch and stop.
        if i_batch == 3:
            show_text_batch(sample_batched)
            break
