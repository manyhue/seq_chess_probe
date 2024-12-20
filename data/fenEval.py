from lib.utils import *
from lib.data import ClassifierData, DataConfig
from sklearn.model_selection import KFold
from lib.chess import fen_labels
import torch.utils.data as td


class FENEval(ClassifierData):
    def __init__(self, c: DataConfig):
        super().__init__()
        self.save_config(c)

        target = "Evaluation"

        df = pl.read_parquet("resources/eval/eval.parquet")
        X = df.drop(target)
        y = df[target]

        self.set_data(X, y)
        self.set_folds(KFold(n_splits=800))
        self.vocab = len(fen_labels.classes_)
