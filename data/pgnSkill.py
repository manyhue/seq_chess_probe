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
from lib.data import ClassifierData, DataConfig
from datasets import load_dataset

from lib.utils import Base

skills = ["low", "medium", "high", "top"]


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
                    moves = iter_to_move_strings(
                        game.mainline_moves(), self.seq_len, pad_token
                    )
                    yield folder_name, moves

    def pgns_to_tensor(self, iter_start, iter_end):
        for folder_name, result in self.parse_pgn_files_to_move_strings(
            self.files[iter_start:iter_end],
            pad_token="<PAD>",  # todo: how to configure pad_token
        ):
            yield torch.tensor(self.chess_move_labels.transform(result)), torch.tensor(self.le.transform(folder_name))


@dataclass(kw_only=True)
class PGNSquareDataConfig(DataConfig):
    seq_len: int = 128
    files_per_epoch: int = 50
    directory: str = "resources/lichess_elite"
    val_directory: str = "resources/lichess_elite_val"
    max_games_per_file: Optional[int] = 99999
    chess_move_labels: LabelEncoder = chess_move_labels


class PGNSquareData(ClassifierData):
    def __init__(self, c: PGNSquareDataConfig):
        super().__init__()
        self.save_config(c)
        self.le = LabelEncoder()
        self.le.fit(skills)

        # setup files
        self._files = [
            os.path.join(root, file)
            for root, _, files in os.walk(c.directory)
            for file in files
            if file.endswith(".pgn")
        ]
        assert self._files
        assert len(self._files) >= self.files_per_epoch

        self.files = (
            random.sample(self._files, self.files_per_epoch)
            if c.files_per_epoch is not None
            else self._files
        )

        self.dataset = StreamingPGNSkillDataset(
            self.files,
            c.seq_len,
            self.square,
            self.le,
            chess_move_labels,
            c.max_games_per_file,
        )

        # setup files for val
        if c.val_directory:
            self.val_files = [
                os.path.join(root, file)
                for root, _, files in os.walk(c.val_directory)
                for file in files
                if file.endswith(".pgn")
            ]
            assert self.val_files
            self.val_set = StreamingPGNSkillDataset(
                self.val_files,
                c.seq_len,
                self.square,
                self.le,
                chess_move_labels,
                c.max_games_per_file,
            )

    def shuffle_files(self):
        self.files = random.sample(self._files, self.files_per_epoch)
