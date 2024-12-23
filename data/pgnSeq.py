import math
import os
import random
from typing import Optional
from torch.utils.data import IterableDataset
import numpy as np
import torch
import tqdm
import zstandard
from dataclasses import dataclass
from lib.data import DataConfig, ClassifierData
from lib.utils import Base
from lib.chess import *


class StreamingPGNDataset(IterableDataset, Base):
    def __init__(self, files, seq_len, le, max_games_per_file):
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
        """
        Parse a sequence of PGN files and generate move strings for each game.

        Args:
            pgn_files (list): List of file paths to PGN files.
            seq_len (int): Maximum sequence length of moves.
            pad_token (str): Token used to pad the sequences.

        Yields:
            list: A list of move strings for each game, padded to seq_len.
        """

        for pgn_file in pgn_files:
            with open(pgn_file, "r") as pgn:
                if vb(5):
                    print("\nopening:", pgn_file)
                for _ in range(self.max_games_per_file):
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break
                    if "FEN" in game.headers:
                        continue
                    yield iter_to_moves(
                        game.mainline_moves(), self.seq_len, pad_token
                    )  # todo: what sort of exceptions to handle?

    def pgns_to_tensor(self, iter_start, iter_end):
        """
        Wrap the parse_pgn_files_to_move_strings generator to transform its results using self.le.transform.

        Args:
            iter_start (int): Starting index for self.files.
            iter_end (int): Ending index for self.files.

        Yields:
            Transformed result using self.le.transform.
        """
        for result in self.parse_pgn_files_to_move_strings(
            self.files[iter_start:iter_end],
            pad_token="<PAD>",  # todo: how to configure pad_token
        ):
            yield (moves_to_torch(result),)


@dataclass(kw_only=True)
class PGNDataConfig(DataConfig):
    seq_len: int = 128
    files_per_epoch: int = 1000
    directory: str = "resources/chess"
    val_directory: Optional[str] = None
    max_games_per_file: Optional[int] = 99999


class PGNData(ClassifierData):
    def __init__(self, c: PGNDataConfig):
        super().__init__()
        self.save_config(c)
        self.le = chess_move_labels

        self._files = [
            os.path.join(root, file)
            for root, _, files in os.walk(c.directory)
            for file in files
            if file.endswith(".pgn")
        ]
        assert self._files

        self.files = (
            random.sample(self._files, self.files_per_epoch)
            if c.files_per_epoch is not None
            else self._files
        )

        self.dataset = StreamingPGNDataset(
            self.files, c.seq_len + 1, self.le, c.max_games_per_file
        )  # will skip first/last move at prepare_batch so increment seq_len

        # setup files for val
        if c.val_directory:
            self.val_files = [
                os.path.join(root, file)
                for root, _, files in os.walk(c.val_directory)
                for file in files
                if file.endswith(".pgn")
            ]
            assert self.val_files
            self.val_set = StreamingPGNDataset(
                self.val_files,
                c.seq_len + 1,
                self.le,
                c.max_games_per_file,
            )

    def shuffle_files(self):
        self.files = random.sample(self._files, self.files_per_epoch)
