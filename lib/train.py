import os
import glob
from pathlib import Path
import sys
import torch
import torcheval
from .metrics import _default_pred
from .utils import *
import torch.nn as nn
import logging
import torch.nn.functional as F
from . import infer
import torcheval.metrics as ms
from .metrics import *
from .config import *
from torch.optim.lr_scheduler import OneCycleLR
import torch.utils.data as td

if is_notebook():
    from tqdm.notebook import tqdm as tqbar
else:
    from tqdm import tqdm as tqbar


@dataclass(kw_only=True)
class OptimizerConfig(Config):
    lr: float = 0.1
    weight_decay: float = 0.01
    betas: Optional[tuple] = None
    eps: Optional[float] = None


@dataclass(kw_only=True)
class TrainerSchedulerOpts(Config):
    steps_per_epoch: float = 0.1
    epochs: float = 0.01


@dataclass(kw_only=True)
class TrainerConfig(OptimizerConfig):
    max_epochs: int = 200
    gpus: Optional[List[int]] = None  # Optional list of GPUs to use
    gradient_clip_val: float = 0.0
    save_model_every: int = 0
    load_previous: bool = False
    save_loss_threshold: float = 1.0
    logger: Optional[Any] = None
    verbosity: int = 0
    train_mfs: MetricsFrame | List[MetricsFrame] = field(default_factory=list)
    val_mfs: MetricsFrame | List[MetricsFrame] = field(default_factory=list)
    batch_end_callback: Callable | List[Callable] = field(default_factory=list)
    epoch_end_callback: Callable | List[Callable] = field(default_factory=list)
    loss_every: float = 0.2  # record loss every n epochs
    flush_mfs: bool = True
    flush_epoch_units: bool = True  # treat
    set_pred: bool = True  # use model.pred as the predictor function for metric frames
    save_path: str | Path = "./out"
    use_dataparallel: bool = False
    scheduler: Optional[
        Callable[
            [torch.optim.Optimizer, TrainerSchedulerOpts],
            torch.optim.lr_scheduler.LRScheduler,
        ]
    ] = None
    optimizer: Optional[torch.optim.Optimizer] = None


class Trainer(Base):
    def __init__(self, c: TrainerConfig):
        self.save_config(c, ignore=["loaders"])
        if not c.gpus:
            self.gpus = get_gpus(-1)  # get all gpus by default

        self.optimizer_config = OptimizerConfig.create(c)
        self.tunable = [
            "lr",
            "weight_decay",
            "betas",
            "eps",
        ]  # affects saved model name

        self.board = None  # ProgressBoard(xlabel="epoch")
        self._best_loss = 9999  # save model loss threshold
        if not self.flush_epoch_units:
            assert isinstance(self.loss_every, int)  # set batch length to log mean loss

    def prepare_optimizers(self, **kwargs):
        self.optim = self.c.optimizer or torch.optim.AdamW(
            self.model.parameters(),
            **self.optimizer_config.dict(),
            **kwargs,
        )

        if self.scheduler:
            opts = TrainerSchedulerOpts(
                steps_per_epoch=self.num_train_batches, epochs=self.max_epochs
            )
            self.sched = self.scheduler(self.optim, opts)
        else:
            self.sched = None

    def prepare_batch(self, batch):
        if self.use_dataparallel:
            return batch  # Handled by DataParallel
        else:
            return [a.to(self.device) for a in batch]  # Move batch to the first device

    @property
    def num_train_batches(self):
        return (
            len(self.train_loader)
            if isinstance(self.train_loader, td.Dataset)
            and not isinstance(self.train_loader, td.IterableDataset)
            else None
        )

    @property
    def num_val_batches(self):
        return (
            len(self.val_loader)
            if isinstance(self.val_loader, td.Dataset)
            and not isinstance(self.val_loader, td.IterableDataset)
            else None
        )

    def prepare_model(self, model):
        self._loaded = False
        # Easy way to run on gpu, better to use the following
        # https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
        if self.use_dataparallel:
            self.model = nn.DataParallel(model, self.gpus)
        else:
            self.model = model.to(self.device)
        self._model = model

        if self.load_previous:
            params = self._load_previous(model)
            if params is not None:
                self._loaded = True
                # if not self.gpus and params["gpus"]:
                #     self.model = self.model.module.to(
                #         "cpu"
                #     )  # unwrap from DataParallel if wrapped

        if not self._loaded:
            self.prepare_optimizers()
            self.first_epoch = 0

        self.loss = getattr(self, "loss", model.loss)
        model.trainer = self

        # if vb(10):
        #     print(self.model.named_parameters())

    def prepare_metrics(self, loss_board=None):
        """Prepare metrics

        Args:
            loss_board (ProgressBoard | False | None, optional): Defaults to None, creating a progress board. If False, none will be created.
        """
        # Called after loaders are set
        if loss_board is None:
            loss_board = make_loss_board()

        self.train_loss_mf = MetricsFrame(
            [
                from_te(
                    torcheval.metrics.Mean,
                    "loss",
                )
            ],
            flush_every=self.loss_every,
            train=True,
        )

        if self.flush_epoch_units:
            self.train_loss_mf.set_flush_unit(self.num_train_batches)

        self.train_loss_mf.to(self.device)
        if loss_board is not False:
            loss_board.add_mf(self.train_loss_mf)

        self.train_mfs = k_level_list(self.train_mfs, k=1)
        self.batch_end_callback = k_level_list(self.batch_end_callback, k=1)
        self.epoch_end_callback = k_level_list(self.epoch_end_callback, k=1)

        for mf in self.train_mfs:
            mf.train = True
            mf.to(self.device)
            if self.set_pred:
                mf.pred_fun = lambda x, *ys: (
                    self._model.pred(x),
                    *(a.squeeze(-1) for a in ys),
                )
            if self.flush_epoch_units:
                mf.set_flush_unit(self.num_train_batches)

            if self.logger:
                if mf.logger is None:
                    mf.logger = self.logger

        # repeat with val
        if self.val_loader:
            self.val_loss_mf = MetricsFrame(
                [
                    from_te(
                        torcheval.metrics.Mean,
                        "loss",
                    ),
                ],
                flush_every=self.loss_every,
            )
            if self.flush_epoch_units:
                self.val_loss_mf.set_flush_unit(self.num_val_batches)
            self.val_loss_mf.to(self.device)
            if loss_board is not False:
                loss_board.add_mf(self.val_loss_mf)

            self.val_mfs = k_level_list(self.val_mfs, k=1)

            for mf in self.val_mfs:
                mf.train = False
                mf.to(self.device)
                if self.set_pred:
                    mf.pred_fun = lambda x, *ys: (
                        self._model.pred(x),
                        *(a.squeeze(-1) for a in ys),
                    )
                if self.flush_epoch_units:
                    mf.set_flush_unit(self.num_val_batches)

                if self.logger:
                    if mf.logger is None:
                        mf.logger = self.logger

            mfs = self.train_mfs + self.val_mfs
        else:
            mfs = self.train_mfs

        # Configure graphical parameters

        # get all the unique boards
        self.boards = ([loss_board] if loss_board else []) + list(
            set(mf.board for mf in mfs if mf.board is not None)
        )

    def init(self, model=None, loaders=None, loss_board=None):
        """(Re)initialize model, loaders, metric frames. Will error if any are not already initialized.
        Useful if you want to further customize initialized objects such as trainer.val_loss_mf before calling trainer.fit(init=False).
        Set loss_board to False to disable loss plotting.

        Args:
            model (_type_, optional): _description_. Defaults to None.
            loaders (_type_, optional): _description_. Defaults to None.
        """
        if loaders:
            self.train_loader = loaders[0]
            self.val_loader = loaders[1] if len(loaders) > 1 else None
        if model:
            self.prepare_model(model)
        self.prepare_metrics(loss_board=loss_board)

    def fit(
        self,
        model=None,
        loaders=None,
    ):
        """Calls trainer.init(model, data), and begins training.
        Plots metrics every epoch.
        Skips initialization if neither are supplied.

        Args:
            model (Module)
            loaders (Data)

        Returns:
            float: best loss
        """

        if model is not None or loaders is not None:
            assert model is not None and loaders is not None
            self.init(model, loaders)

        self.train_batch_idx = 0
        self.val_batch_idx = 0

        epoch_bar = tqbar(
            range(self.first_epoch, self.first_epoch + self.max_epochs),
            desc="Epochs progress",
            unit="Epoch",
        )

        _save_model_counter = 0

        for self.epoch in epoch_bar:
            epoch_loss = self._fit_epoch()
            epoch_bar.set_description(
                "Epochs progress [Loss: {:.3e}]".format(epoch_loss)
            )
            for c in self.epoch_end_callback:
                c()
            _save_model_counter += 1
            if epoch_loss <= self._best_loss:
                self._best_loss = epoch_loss
                if (
                    self.save_model_every != 0
                    and epoch_loss <= self.save_loss_threshold
                    and _save_model_counter >= self.save_model_every
                ):
                    self._save_model()
                    _save_model_counter = 0

        for b in self.boards:
            b.close()  # Close as many plots as are associated, a bit wonky but works ok
        if self.flush_mfs:
            for mf in self.val_mfs + self.train_mfs:
                try:
                    mf.flush()
                except AssertionError:
                    pass
        return self._best_loss

    def _fit_epoch(self, train_loader=None, val_loader=None, y_len=1):
        train_loader = train_loader or self.train_loader
        val_loader = val_loader or self.val_loader
        _init_batch_idx = self.train_batch_idx

        losses = 0
        self.model.train()
        for batch in train_loader:
            batch = self.prepare_batch(batch)
            with torch.set_grad_enabled(True):
                outputs = self.model(*batch[:-y_len])
                Y = batch[-y_len:]
                loss = self.loss(
                    outputs, Y[-1].to(self.device)
                )  # Sending to the main gpu generalizes to a distributed process
                self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if val_loader is None:
                    losses += loss
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
                if self.sched is not None:
                    self.sched.step()

                for m in self.train_mfs:
                    m.update(
                        outputs,
                        *Y,
                        batch_num=self.train_batch_idx,
                    )
                if self.train_loss_mf:
                    self.train_loss_mf.update(
                        loss,
                        batch_num=self.train_batch_idx,
                    )
                # if self.model_mf:
                #     self.model_mf.update(self.model)
            self.train_batch_idx += 1
            for c in self.batch_end_callback:
                c()

            # debug
            # for param in self.model.named_parameters():
            #     if param[1].grad is None:
            #         print("No gradient for parameter:", param)
            #     elif torch.all(param[1].grad == 0):
            #         print("Zero gradient for parameter:", param)

        if val_loader is not None:
            self.model.eval()  # Set the model to evaluation mode, this disables training specific operations such as dropout and batch normalization
            for batch in map(self.prepare_batch, val_loader):
                with torch.no_grad():
                    outputs = self.model(*batch[:-y_len])
                    Y = batch[-y_len:]
                    loss = self.loss(outputs, Y[-1].to(self.device))
                    losses += loss
                    for mf in self.val_mfs:
                        mf.update(
                            outputs,
                            *Y,
                            batch_num=self.train_batch_idx,
                        )
                    if self.val_loss_mf:
                        self.val_loss_mf.update(
                            loss,
                            batch_num=self.train_batch_idx,
                        )
                self.val_batch_idx += 1
                if vb(6):
                    if self.epoch == self.first_epoch + self.max_epochs - 1:
                        print("validation outputs", outputs)

        epoch_loss = losses / (
            self.num_train_batches or self.train_batch_idx - _init_batch_idx
        )
        for b in self.boards:
            b.draw_mfs()

        return epoch_loss

    def clip_gradients(self, grad_clip_val, model):
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm

    def plot(self, label, y, train):
        """Plot a point wrt epoch"""
        if train:
            if self.train_points_per_epoch == 0:  # use to disable plotting/storage
                return
            x = self.train_batch_idx / self.num_train_batches
            n = self.num_train_batches // self.train_points_per_epoch
        else:
            x = self.epoch + 1
            if self.num_val_batches == 0:
                return
            n = self.valid_points_per_epoch // self.num_val_batches

        # move to cpu
        if getattr(y, "device") not in ["cpu", None]:
            y = y.detach().cpu()
        else:
            y = y.detach()

        label = f"{'train/' if train else ''}{label}"
        self.board.draw_points(x, y, label, every_n=n)

    def training_step(self, batch):
        """Compute (and plot loss of a batch) during training step"""
        Y_hat = self.model(*batch[:-1])
        l = self.loss(Y_hat, batch[-1].to(self.device))
        self.plot("loss", l, train=True)
        return l

    # returns a dict
    def validation_step(self, batch):
        """Compute (and plot loss of a batch) during validation step"""
        Y_hat = self.model(*batch[:-1])
        l = self.loss(Y_hat, batch[-1].to(self.device))
        self.plot("loss", l, train=False)
        return {"val_loss", l}

    # def eval(
    #     self,
    #     torchevals=[],
    #     batch_fun=None,
    #     pred_funs=None,
    #     loader=None,
    #     loss=False,
    # ):
    #     """
    #     Evaluates the model on a given loader and computes the metrics and/or loss.

    #     Args:
    #         torchevals (list, optional): List of evaluation metrics or metric groups to be updated during evaluation.
    #                                     Example: [ms.torcheval.metrics.Cat()] or [[torcheval.metrics.BinaryAccuracy()], [torcheval.metrics.Cat()]]
    #                                     For a metric in group i, it is updated with m.update(pred_funs[i](outputs),  *Y)
    #         pred_funs (list, optional): List of prediction functions to be applied to the model outputs.
    #                                     By default, will apply model.pred to group 1 if defined. If torch_evals is longer, the groups use output directly: i.e. m.update(outputs, *Y)
    #         batch_fun (function, optional): Custom definition of function that is applied with batch_fun(outputs, *Y) to each batch. If not supplied, will update supplied torchevals using pred_funs, then output [predictions] or [predictions, loss].
    #         loader (DataLoader, optional): The DataLoader to iterate through during evaluation. If None,
    #                                             defaults to `self.val_loader`.
    #         loss (bool, optional): Whether to output batch_loss in batch_fun. Defaults to False. A custom loss function can also be supplied.

    #     Returns:
    #         tuple: List of metrics, as many as are output by batch_fun. Tensor type metrics are concatenated, while others are arrays of len(loader).
    #             updated metrics, and the second element is the computed loss if requested.
    #     """
    #     assert getattr(self, "_model", None) is not None

    #     if loss is True:
    #         loss = self.loss

    #     if pred_funs is None:
    #         pred_funs = [getattr(self._model, "pred", lambda x: x.squeeze(-1))]

    #     if batch_fun is None:
    #         batch_fun = infer.make_batch_fun(torchevals, pred_funs, loss)

    #     return infer.infer(
    #         self.model,
    #         loader or self.val_loader,
    #         batch_fun,
    #         device=self.device,
    #     )

    def eval(
        self,
        mfs: Union[MetricsFrame, List[MetricsFrame]] = [],
        pred: Union[Callable, bool] = False,
        loss: Union[Callable, bool] = False,
        loader: torch.utils.data.DataLoader = None,
        batch_fun=None,
        flush_zero_flush_every_mfs=True,
    ):
        """Evaluates the model on a given loader and updates the given MetricFrames on the output. Also computes loss/pred if specified.

        Args:
            mfs (Union[MetricsFrame, List[MetricsFrame]], optional): _description_. Defaults to [].
            pred (Union[Callable, bool], optional): Whether to track model predictions. Defaults to False. A custom pred function can also be supplied.
            loss (Union[Callable, bool], optional): Whether to track loss. Defaults to False. A custom loss function can also be supplied. Defaults to False.
            loader (DataLoader, optional): The DataLoader to iterate through during evaluation. If None,
                defaults to `self.val_loader`.
            batch_fun (Callable, optional): _description_. Defaults to None.

        Returns:
            A dictionary with loss and/or pred columns. None if loss and pred are both unspecified.
        """
        assert getattr(self, "_model", None) is not None
        mfs = k_level_list(mfs, k=1)

        output_cols = []
        pred_fn = pred if callable(pred) else self._model.pred

        if pred is not False:
            output_cols.append(
                CatMetric(
                    "pred",
                    update=lambda x, *ys: (pred_fn(x),),
                    num_outs=1,
                    device=self.device,
                )
            )

        if loss is not False:
            loss_fn = loss if callable(loss) else self.loss
            output_cols.append(
                CatMetric(
                    "loss",
                    update=lambda x, *ys: (loss_fn(x, *ys),),
                    num_outs=1,
                    device=self.device,
                )
            )

        for mf in mfs:
            mf.set_flush_unit(1)
            mf.to(self.device)
            if self.set_pred:
                mf.pred_fun = lambda x, *ys: (
                    self._model.pred(x),
                    *_default_pred(ys),
                )

        output_mf = None

        if loss is not False or pred is not False:
            output_mf = MetricsFrame(
                output_cols, flush_every=0, xlabel=None, index_fn=lambda x, y: x
            )  # output concatenation at end
            mfs.append(output_mf)

        # for mf in mfs:
        #     mf.index_fn = lambda *args: self._eval_batch_num
        #     mf.xlabel = mf.xlabel if mf.xlabel is None else "batch_num"

        if batch_fun is None:

            def batch_fun(*args, batch_num):
                for mf in mfs:
                    mf.update(*args, batch_num=batch_num)

        loader = loader or self.val_loader

        infer.infer(
            self.model,
            loader,
            batch_fun,
            device=self.device,
        )

        if flush_zero_flush_every_mfs:
            for mf in mfs:
                if mf.flush_every == 0:
                    mf.flush()

        if output_mf is not None:
            return mfs.pop().dict

    @property
    def filename(self):
        # Filter and create the string for the tunable parameters that exist in self.p
        param_str = "__".join(
            [f"{k}={v}" for k, v in self.c.dict().items() if k in self.tunable]
        )
        return f"{self._model.filename()}__{param_str}"

    def _save_model(self, params={}):
        with change_dir(self.save_path):
            filename = (
                self.filename + f"__epoch={self.first_epoch}-{self.epoch}" + ".pth"
            )
            torch.save(
                {
                    "params": params,
                    "epoch": self.epoch,  # save the epoch of the model
                    "model": self.model.state_dict(),
                    "optimizer": self.optim.state_dict(),
                },
                filename,
            )

    # load a previous model to train further
    def _load_previous(self, epoch="*"):  # glob string, i.e. 100-200
        with change_dir(self.save_path):
            files = glob.glob(self.filename + f"__epoch={epoch}.pth")
            # look for the most recent file
            files.sort(key=os.path.getmtime)
            if len(files) > 0:
                print("Found older file:", files[-1])
                print("Loading.....")
                checkpoint = torch.load(
                    files[-1]
                )  # todo: how to assign devices with dp/ddp

                state_dict = checkpoint["model"]
                unwanted_prefix = "_orig_mod."
                for k, v in list(state_dict.items()):  # unwanted prefixes?
                    if k.startswith(unwanted_prefix):
                        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
                self.model.load_state_dict(state_dict)

                self.optim.load_state_dict(checkpoint["optimizer"])
                # continue on next epoch
                self.first_epoch = checkpoint["epoch"] + 1
                return checkpoint["params"]
            return None

    # sweep_configuration = {
    #     "method": "random",
    #     "name": "sweep",
    #     "metric": {"goal": "minimize", "name": "loss"},
    #     "parameters": {
    #         "batch_size": {"values": [16, 32, 64]},
    #         "max_epochs": {"values": [5, 10, 15]},
    #         "lr": {"max": 0.1, "min": 0.0001},
    #     },
    # }


# # Create the sweep in WandB
# sweep_id = wandb.sweep(sweep_configuration, project="my_project")

# # Define different datasets (Replace these with actual datasets or dataset paths)
# datasets = [
#     "dataset_1",
#     "dataset_2",
#     "dataset_3",
#     "dataset_4",
#     "dataset_5",
#     "dataset_6",
# ]


# # Define the function that will train and log results for each sweep
# def train_rnn(config):
#     # Initialize the run for the sweep
#     wandb.init(config=config)

#     # Select the dataset for the current run
#     dataset = config.dataset_name
#     print(f"Training on {dataset} dataset...")

#     # Extract hyperparameters from the config
#     learning_rate = config.learning_rate
#     batch_size = config.batch_size
#     hidden_size = config.hidden_size
#     dropout = config.dropout

#     # Simulate training and log metrics (replace with actual model training)
#     for epoch in range(10):
#         accuracy = random.uniform(0.7, 1.0)  # Simulated accuracy
#         loss = random.uniform(0.1, 1.0)  # Simulated loss

#         # Log the metrics
#         wandb.log({"epoch": epoch + 1, "accuracy": accuracy, "loss": loss})

#         # Simulate epoch duration
#         time.sleep(0.5)

#     wandb.finish()


# # Run sweeps for each dataset, passing the dataset name for each experiment
# for dataset in datasets:
#     # Update the config for each dataset
#     config = {
#         "dataset_name": dataset,
#     }

#     # Start the sweep agent to run the experiment for the current dataset
#     wandb.agent(sweep_id, function=train_rnn, count=1)  # Run 1 experiment per dataset
