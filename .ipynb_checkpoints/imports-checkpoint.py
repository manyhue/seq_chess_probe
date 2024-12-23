import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from lib import *
from lib.utils import *
from lib.data import *
from lib.modules import *
from lib.metrics import *
from lib.train import *
import logging
import importlib
from pprint import pprint
from dataclasses import dataclass

# import torcheval.metrics as ms
# import torch.utils.data as td

sns.set_theme(style="darkgrid")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#     print(f"Using: {device}. Device: {torch.cuda.get_device_name()}")


if is_notebook():
    from IPython.display import display, clear_output
