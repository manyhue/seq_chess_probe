import glob
from torch import nn
from torch.nn import functional as F

from .utils import *


# Do not use properties beginning with _
class Module(nn.Module, Base):
    """The base class of models."""

    def __init__(self, **kwargs):
        super().__init__()
        self.save_attr()  # saves kwargs

    ## Training section
    # Defaults for classification are used and should likely be overridden

    def loss(self, outputs, y):
        loss = F.mse_loss(outputs, y.reshape(outputs.shape), reduction="mean")
        return loss

    def forward(self, X):
        assert hasattr(self, "net"), "Neural network is defined"
        return self.net(X)

    # @property doesn't work, overridden by nn somehow
    def filename(self):
        # join key-value pairs from self.c, each subclass should accept a c: Config and call save_config(c)
        param_str = "__".join([f"{k}={v}" for k, v in self.c.dict().items()])
        return f"{self.__class__.__name__}__{param_str}"

    def pred(self, output):
        return output.squeeze(-1)

    def layer_summary(self, X_shape):
        """Displays model output dimensions for each layer given input X-shape.

        Args:
            X_shape (tuple)
        """
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, "output shape:\t", X.shape)

    def show(self):
        for p in self.named_parameters():
            print(p)

    def save_to_pth(self, filename=None, params={}):
        torch.save(
            {
                "params": self.c,
                "model": self.model.state_dict(),
            },
            filename or self.filename(),
        )

    def load_from_pth(self, pth_path, epoch_glob=None):
        # if epoch_glob:
        #     with change_dir(self.save_path):
        #         files = glob.glob(self.filename() + f"*__epoch={epoch_glob}.pth")
        #         files.sort(key=os.path.getmtime)
        #         if len(files) > 0:
        #             print("Found older file:", files[-1])
        #             checkpoint = torch.load(files[-1])
        #         else:
        #             raise Exception("No file found")
        # else:
        checkpoint = torch.load(pth_path)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):  # unwanted prefixes?
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        self.load_state_dict(state_dict)
        return checkpoint


# Just a reminder of basic field names, better to declare all fields when subclassing and subclass Config instead
@dataclass(kw_only=True)
class ClassifierConfig(Config):
    n_blks: int  # num primary blocks
    n_classes: int
    dropout: float = 0.1
    hidden_size: int  # first MLP


class ClassifierModule(Module):
    """
    The base class of classification models.
    """

    def loss(self, outputs, Y, averaged=True):
        outputs = outputs.reshape(-1, outputs.shape[-1])
        Y = Y.reshape(
            -1,
        )
        loss = F.cross_entropy(outputs, Y, reduction="mean" if averaged else "none")
        return loss

    # Used in evaluation/metrics steps, which compare pred(output) and label
    def pred(self, output):
        return output.argmax(dim=-1).to(torch.int64)


class SmoothClassifierModule(ClassifierModule):
    """
    The base class of classification models.
    """

    def loss(self, outputs, Y, averaged=True):
        outputs = outputs.reshape(-1, outputs.shape[-1])
        Y = Y.reshape(
            -1,
        )
        loss = F.cross_entropy(
            outputs, Y, reduction="mean" if averaged else "none", ignore_index=0
        )
        return loss


class MultiClassifier(Module):
    """
    The base class of classification models.
    """

    def loss(self, outputs, Y, averaged=True):
        outputs = outputs.reshape(-1, outputs.shape[-1])
        Y = Y.reshape(
            -1,
        )
        return F.cross_entropy(outputs, Y, reduction="mean" if averaged else "none")

    def pred(self, output):
        return output.argmax(dim=1)
