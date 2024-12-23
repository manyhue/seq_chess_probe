import os
import glob
from pathlib import Path
import sys
from typing import Tuple
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
    logger: Optional[Any] = None
    verbosity: int = 0
    train_mfs: MetricsFrame | List[MetricsFrame] = field(default_factory=list)
    val_mfs: MetricsFrame | List[MetricsFrame] = field(default_factory=list)
    batch_end_callback: Callable | List[Callable] = field(
        default_factory=list
    )  # provides trainer and loss
    epoch_end_callback: Callable | List[Callable] = field(default_factory=list)
    save_dir: str | Path = "./out"
    use_dataparallel: bool = False
    scheduler: Optional[
        Callable[
            [torch.optim.Optimizer, TrainerSchedulerOpts],
            torch.optim.lr_scheduler.LRScheduler,
        ]
    ] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    epoch_every: Optional[int] = (
        None  # artificial epoch every n batches. Max_epochs still controls how many times to run the dataloader. So tqbar progress is not updated on these miniepochs. A function of using training time and update time to configure epoch_every and max_epochs would be friendly.
    )
    # Set epoch units for each mf with unit=None. This also controls whether to use batches or epochs for loss units. If you specifically want to set loss mf units seperately, use set_unit(1, "batch") after trainer.init
    flush_loss_every: Optional[float] = (
        0.2  # 1/plot points when epoch units. When batches_per_epoch is None, this should be an integer representing batches.
    )
    # the following are qol options that should be fine to not change
    save_loss_threshold: float = 10
    mf_epoch_units: bool = True  # 400 is around 4-8 min before first plot @ 1-3 sec per batch, ~ 30 hrs for 200 epochs.
    flush_mfs: bool = True
    set_pred: bool = True  # use model.pred as the predictor function for metric frames
    plot_start: Tuple[int, int] = (20, 20)


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
        if not self.mf_epoch_units:
            assert isinstance(
                self.flush_loss_every, int
            )  # set batch length to log mean loss
        else:
            self.flush_loss_every = self.flush_loss_every or c.max_epochs / 600

    def prepare_optimizers(self, **kwargs):
        self.optim = self.c.optimizer or torch.optim.AdamW(
            self.model.parameters(),
            **self.optimizer_config.dict(),
            **kwargs,
        )

        if self.scheduler:
            opts = TrainerSchedulerOpts(
                steps_per_epoch=self.batches_per_epoch, epochs=self.max_epochs
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
    def batches_per_epoch(self):
        return (
            self.epoch_every or self.num_train_batches
        )  # perhaps should not be a property and rather be set in prepare_data

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

        self.prepare_optimizers()

        if self.load_previous:
            params = self._load_previous(
                relaxed=(self.load_previous == "relaxed")
            )  # if u want to change lr or something
            if params is not None:
                self._loaded = True
                # if not self.gpus and params["gpus"]:
                #     self.model = self.model.module.to(
                #         "cpu"
                #     )  # unwrap from DataParallel if wrapped
        if not self._loaded:
            self.first_epoch = 1

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
            flush_every=self.flush_loss_every,
            train=True,
        )

        if self.mf_epoch_units:
            if self.batches_per_epoch is None:
                print("Warning: Cannot set epoch units since batches_per_epoch is None")
            else:
                self.train_loss_mf.set_unit(self.batches_per_epoch, "epoch")
                for mf in self.train_mfs:
                    mf.set_unit(self.batches_per_epoch, "epoch")

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

            if self.logger:
                if mf.logger is None:
                    mf.logger = self.logger

        # repeat with val
        self.val_loss_mf = None
        if self.val_loader is not None:
            self.val_loss_mf = MetricsFrame(
                [
                    from_te(
                        torcheval.metrics.Mean,
                        "loss",
                    ),
                ],
                flush_every=min(self.flush_loss_every, 1),
            )  # val_loader is only called once per epoch

            if self.mf_epoch_units:
                if self.num_val_batches is None:
                    print(
                        "Warning: Cannot set val epoch units since batches_per_epoch is None."  # todo: provide synchronization for shared metric frames with boards.
                    )
                else:
                    self.val_loss_mf.set_unit(
                        self.batches_per_epoch, "epoch", scale_flush_interval=False
                    )  # note that we use the train index to keep in step. This works better in the situation when num_val_batches is not defined: the loss plot with train and val batches has a longer train line. This does not plot a graph of the validation progress however. num_val_batches is undefined occurs iff both train and val uses batch units. In this case, to keep synchronization, one could consider supplying a custom diminishing scaling function from INIT_TRAIN_BATCH to train_batch_idx.
                    for mf in self.val_mfs:
                        mf.set_unit(self.num_val_batches, "epoch")

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

    def init(self, model=None, loaders=None):
        """(Re)initialize model, loaders, metric frames. Will error if any are not already initialized.
        Useful if you want to further customize initialized objects such as trainer.val_loss_mf before calling trainer.fit(init=False).
        Set loss_board to False to disable loss plotting.

        Args:
            model (_type_, optional): _description_. Defaults to None.
            loaders (_type_, optional): _description_. Defaults to None.
        """
        self.train_loader = loaders[0] if loaders else None
        self.val_loader = loaders[1] if loaders and len(loaders) > 1 else None
        if model:
            self.prepare_model(model)

    def fit(self, model=None, loaders=None, loss_board=None):
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
        self.prepare_metrics(loss_board=loss_board)

        # this is actually guaranteed by the default process
        if self.train_loss_mf is not None and self.val_loss_mf is not None:
            assert (
                (self.train_loss_mf.unit in [None, 1])
                == (self.val_loss_mf.unit in [None, 1])
            ), "loss and train must be plotted on same units of epoch or batch. Batch units are used when mf_use_epoch units is False or when the training dataset is Iterable and epoch_every is unset."

        self.train_batch_idx = (self.first_epoch - 1) * (self.batches_per_epoch or 0)
        self.val_batch_idx = (self.first_epoch - 1) * (
            self.num_val_batches or 0
        )  # continuing from batch not supported

        self.batch_loss = 0
        self.batch_val_loss = 0

        self.epoch_bar = tqbar(
            range(self.first_epoch, self.first_epoch + self.max_epochs),
            desc="Epochs progress",
            unit="Epoch",
        )

        save_model_counter = 0

        def post_epoch(epoch_loss):
            nonlocal save_model_counter
            for b in self.boards:
                b.draw_mfs()

            self.epoch_bar.set_description(
                "Epochs progress [Loss: {:.3e}]".format(epoch_loss)
            )

            for c in self.epoch_end_callback:
                self.__call_with_optional_self(c)

            save_model_counter += 1
            if epoch_loss <= self._best_loss:
                self._best_loss = epoch_loss
                if (
                    self.save_model_every != 0
                    and epoch_loss <= self.save_loss_threshold
                    and save_model_counter >= self.save_model_every
                ):
                    self.save_model()
                    save_model_counter = 0

        for self.epoch in self.epoch_bar:
            self._fit_epoch(post_epoch=post_epoch)

        for b in self.boards:
            b.close()  # Close as many plots as are associated, a bit wonky but works ok
        if self.flush_mfs:  # flush
            for mf in self.val_mfs:
                mf.flush(self.val_batch_idx)
            for mf in self.train_mfs:
                mf.flush(self.train_batch_idx)

        return self._best_loss

    def __call_with_optional_self(self, func):
        num_args = len(inspect.signature(func).parameters)
        if num_args == 1:
            return func(self)
        else:
            return func()

    def _fit_epoch(
        self,
        train_loader=None,
        val_loader=None,
        y_len=1,
        post_epoch=lambda epoch_loss: None,
    ):
        train_loader = train_loader or self.train_loader
        val_loader = val_loader or self.val_loader

        _INIT_BATCH_IDX = self.train_batch_idx

        losses = 0

        def validate_and_get_epoch_loss(losses):
            if val_loader is not None:
                return self._val_epoch(val_loader, y_len=y_len)
            else:
                mean_losses = losses / (self.train_batch_idx - _INIT_BATCH_IDX)
                return mean_losses

        self.model.train()
        for batch in map(
            self.prepare_batch,
            train_loader,
        ):
            with torch.set_grad_enabled(True):
                outputs = self.model(*batch[:-y_len])
                Y = batch[-y_len:]
                self.train_batch_loss = self.loss(outputs, Y[-1].to(self.device))
                self.optim.zero_grad()
            with torch.no_grad():
                self.train_batch_loss.backward()
                losses += self.train_batch_loss
                self.train_batch_idx += 1

                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)

                self.optim.step()
                if self.sched is not None:
                    self.sched.step()

                for m in self.train_mfs:
                    m.update(
                        outputs,
                        *Y,
                        idx=self.train_batch_idx,
                    )
                if self.train_loss_mf:
                    self.train_loss_mf.update(
                        self.train_batch_loss,
                        idx=self.train_batch_idx,
                    )

                if self.plot_start is not False:
                    q, r = divmod(self.train_batch_idx, self.plot_start[1])
                    if r == 0 and q < self.plot_start[0]:
                        for b in self.boards:
                            b.draw_mfs()

            for c in self.batch_end_callback:
                self.__call_with_optional_self(c)

            if self.epoch_every and self.train_batch_idx % self.epoch_every == 0:
                post_epoch(validate_and_get_epoch_loss(losses))
                losses = 0

        post_epoch(validate_and_get_epoch_loss(losses))

    def _val_epoch(self, val_loader, y_len=1):
        _INIT_VAL_BATCH_IDX = self.val_batch_idx
        losses = 0

        for batch in map(
            self.prepare_batch,
            val_loader,
        ):
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(*batch[:-y_len])
                Y = batch[-y_len:]
                self.val_batch_loss = self.loss(outputs, Y[-1].to(self.device))
                losses += self.val_batch_loss

                for mf in self.val_mfs:
                    mf.update(
                        outputs,
                        *Y,
                        idx=self.val_batch_idx,
                    )
                self.val_batch_idx += 1

        # we are only interested in mean loss across full validation set during training
        mean_losses = losses / (self.val_batch_idx - _INIT_VAL_BATCH_IDX)

        if self.val_loss_mf:
            self.val_loss_mf.update(
                mean_losses,
                idx=self.train_batch_idx,
            )

            if vb(6):
                if self.epoch == self.first_epoch + self.max_epochs - 1:
                    print("validation outputs", outputs)
        return mean_losses

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
            x = self.train_batch_idx / self.batches_per_epoch
            n = self.batches_per_epoch // self.train_points_per_epoch
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

    def eval(
        self,
        mfs: Union[MetricsFrame, List[MetricsFrame]] = [],
        pred: Union[Callable, bool] = False,
        loss: Union[Callable, bool] = False,
        loader: torch.utils.data.DataLoader = None,
        batch_fun=None,
        flush_mfs=True,
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
            A dictionary with losses and/or preds columns. None if loss and pred are both unspecified.
        """

        assert getattr(self, "_model", None) is not None
        flush_mfs = self.flush_mfs if flush_mfs is None else flush_mfs
        mfs = k_level_list(mfs, k=1)

        output_cols = []
        pred_fn = pred if callable(pred) else self._model.pred

        if pred is not False:
            output_cols.append(
                CatMetric(
                    "preds",
                    update=lambda x, *ys: (pred_fn(x),),
                    num_outs=1,
                    device=self.device,
                )
            )

        if loss is not False:
            loss_fn = loss if callable(loss) else self.loss
            output_cols.append(
                CatMetric(
                    "losses",
                    update=lambda x, *ys: (loss_fn(x, *ys),),
                    num_outs=1,
                    device=self.device,
                )
            )

        def _pred_fn(x, *ys):
            return (
                self._model.pred(x),
                *_default_pred(*ys),
            )

        for mf in mfs:
            mf.to(self.device)
            if self.set_pred:
                mf.pred_fun = _pred_fn

        output_mf = None

        if loss is not False or pred is not False:
            output_mf = MetricsFrame(
                output_cols,
                flush_every=0,
                xlabel=None,
            )  # output concatenation at end
            mfs.append(output_mf)

        # for mf in mfs:
        #     mf.index_fn = lambda *args: self._eval_batch_num
        #     mf.xlabel = mf.xlabel if mf.xlabel is None else "batch_num"

        if batch_fun is None:

            def batch_fun(outputs, batch, batch_num):
                for mf in mfs:
                    mf.update(outputs, *batch[1:])

        loader = loader if loader is not None else self.val_loader

        infer.infer(
            self.model,
            loader,
            batch_fun,
            device=self.device,
            prepare_batch=self.prepare_batch,
        )

        if flush_mfs:
            for mf in mfs:
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

    def save_model(self, params={}, prefix="", filename=None):
        with change_dir(self.save_dir):
            if filename is None:
                filename = prefix + (
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
            if vb(3):
                print(self.save_dir, filename)

    # def load_previous(self, epoch="*", relaxed=False):  # glob string, i.e. 100-200
    #     assert self._model
    #     with change_dir(self.save_dir):
    #         print(self.save_dir)
    #         print(self.filename + f"__epoch={epoch}.pth")
    #         files = (
    #             glob.glob(self._model.filename() + f"*__epoch={epoch}.pth")
    #             if relaxed
    #             else glob.glob(self.filename + f"__epoch={epoch}.pth")
    #         )
    #         print(files)

    # load a previous model to train further
    def _load_previous(self, epoch="*", relaxed=False):  # glob string, i.e. 100-200
        with change_dir(self.save_dir):
            if vb(3):
                print(self._model.filename() + f"*__epoch={epoch}.pth")
            files = (
                glob.glob(self._model.filename() + f"*__epoch={epoch}.pth")
                if relaxed
                else glob.glob(self.filename + f"__epoch={epoch}.pth")
            )
            # look for the most recent file
            files.sort(key=os.path.getmtime)
            if len(files) > 0:
                print("Found older file:", files[-1])
                print("Loading.....")
                # todo: how to assign devices with dp/ddp
                checkpoint = torch.load(files[-1])

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
            print("Skipping load. No older file found.")
            return None
