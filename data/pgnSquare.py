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


class StreamingPGNSquareDataset(IterableDataset, Base):
    def __init__(
        self, files, seq_len, square, le, chess_move_labels, max_games_per_file
    ):
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

    def process_chess_game(self, moves: Iterable):
        """
        Process a movetext to yield multiple training examples.

        Args:
            movetext: String containing chess moves

        Returns:
            Dictionary containing 'game' and 'piece' lists
        """
        move_strings, board = iter_to_moves(
            moves,
            seq_len=self.seq_len,
            return_board=True,
        )

        total_moves = len(board.move_stack)

        if total_moves < 15:
            return [], []

        num_examples = min(3 + (total_moves - 10) // 16, 5)

        # Select random turns between move 10 and end
        selected_turns = random.sample(range(10, total_moves + 1), num_examples)

        for turn in selected_turns:
            # Get moves up to this turn
            game_sequence = move_strings[:turn] + (self.seq_len - turn) * ["<PAD>"]

            # Reset board and play moves up to this turn
            temp_board = chess.Board()
            for move in board.move_stack[:turn]:
                temp_board.push(move)

            # Get piece
            piece = temp_board.piece_at(self.square)
            piece_symbol = piece.symbol() if piece else " "

            yield game_sequence, piece_symbol

    def parse_pgn_files_to_move_strings(self, pgn_files):
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
                    yield from self.process_chess_game(
                        game.mainline_moves()
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
        for game, piece in self.parse_pgn_files_to_move_strings(
            self.files[iter_start:iter_end],
        ):
            yield (
                moves_to_torch(game),
                torch.tensor(self.le.transform([piece])),
            )


@dataclass(kw_only=True)
class PGNSquareDataConfig(DataConfig):
    seq_len: int = 128
    files_per_epoch: int = 50
    directory: str = "resources/lichess_elite"
    val_directory: str = "resources/lichess_elite_val"
    max_games_per_file: Optional[int] = 99999
    square: chess.Square = chess.E4
    chess_move_labels: LabelEncoder = chess_move_labels


class PGNSquareData(ClassifierData):
    def __init__(self, c: PGNSquareDataConfig):
        super().__init__()
        self.save_config(c)
        self.le = piece_labels

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

        self.dataset = StreamingPGNSquareDataset(
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
            self.val_set = StreamingPGNSquareDataset(
                self.val_files,
                c.seq_len,
                self.square,
                self.le,
                chess_move_labels,
                c.max_games_per_file,
            )

    def shuffle_files(self):
        self.files = random.sample(self._files, self.files_per_epoch)
