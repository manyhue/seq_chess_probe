import random
import chess
from typing import List, Tuple, Dict
from dataclasses import dataclass
import torch
from torch.utils.data import IterableDataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
from lib.chess import *
from tnibs.data import ClassifierData, DataConfig
from datasets import load_dataset

pieces = ["K", "Q", "N", "R", "B", "P"]


class SquareDataConfig(DataConfig):
    seq_len: int = 129
    square: chess.Square = chess.E4
    chess_move_labels: LabelEncoder = chess_move_labels


class SquareData(ClassifierData):
    def __init__(self, c: SquareDataConfig):
        super().__init__()
        self.save_config(c)
        self.le = LabelEncoder()
        self.le.fit([" "] + pieces + [x.lower() for x in pieces])

        train_set = load_dataset(
            "Lichess/standard-chess-games",
            split="train",
            streaming=True,
            columns=["TimeControl", "movetext", "WhiteElo", "BlackElo"],
        )

        def min_time(x):
            time_string = x["TimeControl"]
            for i, char in enumerate(time_string):
                if not char.isdigit():
                    initial_time_string = time_string[:i]
            if initial_time_string:
                return int(initial_time_string) >= 300
            return False

        def min_elo(x):
            if not (x["WhiteElo"] and x["BlackElo"]):
                return False
            return (int(x["WhiteElo"]) + int(x["BlackElo"])) > 4000

        train_set = train_set.filter(min_time)
        train_set = train_set.filter(min_elo)

        def transform_batch(row):
            all_games = []
            all_pieces = []

            for movetext in row["movetext"]:
                game, pieces = self.process_chess_game(movetext)
                all_games.extend(game)
                all_pieces.extend(pieces)

            return {"game": all_games, "piece": all_pieces}

        info_columns = train_set.column_names
        info_columns.remove("movetext")
        games_only = train_set.remove_columns(info_columns)

        self.dataset = games_only.map(
            transform_batch, remove_columns="movetext", batched=True
        )

    def process_chess_game(self, movetext: str) -> Tuple[List, List]:
        """
        Process a movetext to create multiple training examples.

        Args:
            movetext: String containing chess moves

        Returns:
            Dictionary containing 'game' and 'piece' lists
        """
        move_strings, board = iter_to_moves(
            movetext_to_moves(movetext),
            seq_len=self.seq_len,
            return_board=True,
        )

        total_moves = len(board.move_stack)

        if total_moves < 10:
            return [], []

        num_examples = min(3 + (total_moves - 10) // 16, 5)

        # Select random turns between move 10 and end
        selected_turns = random.sample(range(10, total_moves + 1), num_examples)

        games = []
        pieces = []

        for turn in selected_turns:
            # Get moves up to this turn
            game_sequence = move_strings[:turn] + (self.seq_len - turn) * ["<PAD>"]
            encoded_seq = torch.tensor(self.chess_move_labels.transform(game_sequence))
            games.append(encoded_seq)

            # Reset board and play moves up to this turn
            temp_board = chess.Board()
            for move in board.move_stack[:turn]:
                temp_board.push(move)

            # Get piece
            piece = temp_board.piece_at(self.square)
            piece_symbol = piece.symbol() if piece else " "
            encoded_piece = torch.tensor(self.le.transform(piece_symbol))
            pieces.append(encoded_piece)

        return games, pieces


# Usage w
