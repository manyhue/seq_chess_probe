import math
import os
import random
import chess
from typing import Any, Generator, Iterable, List, Optional, Tuple, Dict
from dataclasses import dataclass
import torch
from torch.utils.data import IterableDataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
from lib.chess import *
from tnibs.data import ClassifierData, DataConfig
from datasets import load_dataset

from tnibs.utils import Base


class StreamingPGNSkillDataset(IterableDataset, Base):
    def __init__(self, files, seq_len, le, chess_move_labels, max_games_per_file):
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

    def parse_pgn_files_to_move_strings(self, pgn_files, pad_token="<PAD>"):
        for pgn_file in pgn_files:
            folder_name = os.path.basename(os.path.dirname(pgn_file))
            with open(pgn_file, "r") as pgn:
                if vb(5):
                    print("\nopening:", pgn_file)

                for _ in range(self.max_games_per_file):
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break
                    if "FEN" in game.headers:
                        continue
                    moves = iter_to_moves(
                        game.mainline_moves(), self.seq_len, pad_token
                    )

                    yield folder_name, moves

    def pgns_to_tensor(self, iter_start, iter_end):
        for folder_name, result in self.parse_pgn_files_to_move_strings(
            self.files[iter_start:iter_end],
        ):
            try:
                yield (
                    torch.tensor(self.chess_move_labels.transform(result)),
                    torch.tensor(self.le.transform([folder_name])),
                )
            except:  # not recommended but don't know error type for 0000 move  # noqa: E722
                if vb(10):
                    print("Err:", result)


class PGNSkillDataConfig(DataConfig):
    seq_len: int = 128
    directory: str = "resources/binned"
    max_games_per_file: Optional[int] = 999
    chess_move_labels: LabelEncoder = chess_move_labels


class PGNSkillData(ClassifierData):
    def __init__(self, c: PGNSkillDataConfig):
        super().__init__()
        self.save_config(c)
        self.le = LabelEncoder()
        skills = ["low", "medium", "high", "top"]
        self.le.fit(skills)

        # setup files
        self.files = [
            os.path.join(root, file)
            for root, _, files in os.walk(c.directory)
            for file in files
            if file.endswith(".pgn")
        ]
        self.files.sort(key=lambda f: os.path.basename(f))

        assert self.files

        # Split files into training and validation sets (1:6 split)
        self.train_files = []
        self.val_files = []

        # Iterate through files and split them into train and validation sets based on a 1:6 ratio
        for i, file in enumerate(self.files):
            if i % 6 == 0:  # Every 6th file goes to validation set
                self.val_files.append(file)
            else:
                self.train_files.append(file)

        random.shuffle(self.train_files)

        # Create the dataset with the 1:6 split
        self.dataset = StreamingPGNSkillDataset(
            self.train_files,
            c.seq_len,
            self.le,
            chess_move_labels,
            c.max_games_per_file,
        )

        self.val_set = StreamingPGNSkillDataset(
            self.val_files,
            c.seq_len,
            self.le,
            chess_move_labels,
            c.max_games_per_file,
        )

    def shuffle_files(self):
        random.shuffle(self.train_files)