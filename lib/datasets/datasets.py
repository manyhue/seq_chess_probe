from typing import Dict, List
import torch.utils.data as td
import torch
import torchvision
from lib.utils import Base
from datasets import IterableDataset
import math
from itertools import chain


class SeqDataset(td.Dataset, Base):
    # gap and samples Not Implemented yet
    def __init__(self, data, seq_len, y_len=1, gap=1) -> None:
        self.save_attr()

    def __len__(self) -> int:
        return len(self.data[0]) - self.seq_len

    def __getitem__(self, i):
        end = i + self.seq_len
        return self.data[0][i:end], self.data[1][end + 1 - self.y_len : end + 1]


class DataframeDataset(td.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, index):
        row = self.dataframe.iloc[index].to_numpy()
        features = row[1:]
        label = row[0]
        return features, label

    def __len__(self):
        return len(self.dataframe)


class SubDataset(td.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


class UnionDataset(td.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = (len(ds) for ds in datasets)

    def __getitem__(self, index):
        for i, l in enumerate(self.lengths):
            if index < l:
                return self.datasets[i][index]
            index -= l
        raise IndexError

    def __len__(self):
        return sum(self.lengths)
