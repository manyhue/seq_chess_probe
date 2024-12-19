import math
import os
import random
from torch.utils.data import IterableDataset
import numpy as np
import torch
import tqdm
import zstandard
from dataclasses import dataclass
from lib.data import DataConfig, ClassifierData
from lib.chess import parse_pgn_files_to_move_strings
from lib.utils import Base
from lib.chess import *


class StreamingPGNDataset(IterableDataset, Base):
    def __init__(self, files, seq_len, le):
        self.save_attr()
        self.start = 0
        self.end = len(self.files)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(
                math.ceil((self.end - self.start) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        return self.pgns_to_tensor(iter_start, iter_end)

    def pgns_to_tensor(self, iter_start, iter_end):
        """
        Wrap the parse_pgn_files_to_move_strings generator to transform its results using self.le.transform.

        Args:
            iter_start (int): Starting index for self.files.
            iter_end (int): Ending index for self.files.

        Yields:
            Transformed result using self.le.transform.
        """
        for result in parse_pgn_files_to_move_strings(
            self.files[iter_start:iter_end], self.seq_len, pad_token="<PAD>"
        ):
            yield (torch.tensor(self.le.transform(result)),)


@dataclass(kw_only=True)
class PGNDataConfig(DataConfig):
    seq_len: int = 128
    files_per_epoch: int = 1000


class PGNData(ClassifierData):
    def __init__(self, c: PGNDataConfig):
        super().__init__()
        self.save_config(c)
        self.le = chess_move_labels

        self._files = [
            os.path.join(root, file)
            for root, _, files in os.walk("resources/chess")
            for file in files
            if file.endswith(".pgn")
        ]

        self.files = self._files[: self.files_per_epoch]

        self.dataset = StreamingPGNDataset(
            self.files, c.seq_len + 1, self.le
        )  # will skip first/last when used

    def shuffle_files(self):
        self.files = random.sample(self.files, self.files_per_epoch)
