import torch
from torch.utils.data import Dataset

from tokenizer import encode_sequences

"""
Reference
* https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html
"""


class FrEnDataset(Dataset):
    """IWSLT17 dataset for FR <=> EN translation."""

    def __init__(self, txt_files, root_dir, transform=None):
        """
        Args:
            txt_files (string): txt 파일의 경로
            root_dir (string): 모든 xml 파일이 존재하는 디렉토리 경로
            transform (callable, optional): 샘플에 적용될 Optional transform
        """
        # input_file, output_file = txt_files.split(",")
        # self.input_seq = encode_sequences(input_file)
        # self.output_seq = encode_sequences(output_file)
        self.datasets = encode_sequences(txt_files, '../model/test_spm.model')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """
        :return: 데이터셋의 크기
        """
        return len(self.datasets)

    def __getitem__(self, idx):
        """
        i번째 샘플을 찾는데 사용
        :param idx:
        :return: i번째 샘플
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = (self.input_seq[idx], self.output_seq[idx])
        sample = self.datasets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
