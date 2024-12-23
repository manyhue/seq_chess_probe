from dataclasses import dataclass
import torch
from torch.utils.data import IterableDataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
from lib.chess import *
from lib.data import ClassifierData, DataConfig
from datasets import load_dataset


@dataclass(kw_only=True)
class PGNDataConfig(DataConfig):
    seq_len: int = 129


class PGNData(ClassifierData):
    def __init__(self, c: PGNDataConfig):
        super().__init__()
        self.save_config(c)
        self.le = LabelEncoder()
        self.le.fit(["<PAD>"] + generate_chess_moves())

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

        def transform(row):
            pgn_string = row["movetext"]
            return {
                "game": torch.tensor(
                    self.le.transform(
                        iter_to_moves(
                            chess.pgn.read_game(io.StringIO(pgn_string)),
                            seq_len=self.seq_len + 1,
                            pad_token="<PAD>",
                        )
                    )
                ),
            }

        info_columns = train_set.column_names
        info_columns.remove("movetext")
        games_only = train_set.remove_columns(info_columns)

        self.dataset = games_only.map(transform, remove_columns="movetext")
