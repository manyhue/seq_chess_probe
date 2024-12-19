from abc import abstractmethod
import logging
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from .utils import *
import IPython.display


from torcheval.metrics.metric import Metric
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
)

# pyre-fixme[24]: Generic type `Metric` expects 1 type parameter.
TSelf = TypeVar("TSelf", bound="Metric")
TComputeReturn = TypeVar("TComputeReturn")
# pyre-ignore[33]: Flexible key data type for dictionary
TState = Union[torch.Tensor, List[torch.Tensor], Dict[Any, torch.Tensor], int, float]


def make_loss_board():
    return ProgressBoard(title="Loss", xlabel="epoch", ylabel="loss", yscale="log")


# operator.add, operator.truediv?
class MeanMetric(Metric):
    def __init__(
        self,
        label,
        statistic: Callable[[object, object], object],
        transform: Callable[[object, object], object] = lambda total, count: total
        / count,
        reduce: Callable[[object, object], object] = lambda total, stat: total + stat,
        device: Optional[torch.device] = None,
    ):
        self.save_attr()
        self.reset()

    @torch.inference_mode()
    def compute(self):
        return self.transform(self._total, self._count)

    @torch.inference_mode()
    def update(self, *args):
        self._total = self.reduce(self._total, self.statistic(*args))
        self._count += 1

    @torch.inference_mode()
    def reset(self):
        self._total = 0
        self._count = 0

    @torch.inference_mode()
    def merge_state(self: TSelf, metrics: Iterable[TSelf]) -> TSelf:
        for metric in metrics:
            self._total = self.reduce(self._total, metric._total)
            self._count += metric._count


class CatMetric(Metric):
    def __init__(
        self,
        label,
        compute=lambda *args: args,
        update=None,
        num_outs=None,
        device=torch.device("cpu"),
        **kwargs,
    ):
        self.label = label
        self._compute = compute
        self._kwargs = kwargs
        self._update = None
        if callable(update):
            assert isinstance(num_outs, int)
            self._update = update
            self.num_outs = num_outs
        else:
            self.num_outs = 2
        self._device = device

        self.reset()

    @torch.inference_mode()
    def compute(self):
        return self._compute(
            *(a.cpu().numpy() for a in self._store),
            **self._kwargs,
        )

    @torch.inference_mode()
    def update(self, *args):
        args = tuple(a.to(self._device) for a in args)
        args = self._update(*args) if callable(self._update) else args

        for idx, tensor in enumerate(self._store):
            new = args[idx]
            self._store[idx] = torch.cat(
                [tensor, new.flatten() if new.dim() == 0 else new], dim=0
            )

    @torch.inference_mode()
    def reset(self):
        self._store = [
            torch.empty(0, dtype=torch.float32, device=self._device)
            for _ in range(self.num_outs)
        ]

    def to(self, device):
        self._store = [u.to(device) for u in self._store]
        self._device = device

    @torch.inference_mode()
    def merge_state(self: TSelf, metrics: Iterable[TSelf]) -> TSelf:
        for metric in metrics:
            for idx, tensor in enumerate(self._store):
                self._store[idx] = torch.cat([tensor, metric._store[idx]], dim=0)


def from_te(torcheval_metric, label, **kwargs):
    class _subclass(torcheval_metric):
        def update(self, *args, **kwargs):
            super().update(*args, **kwargs)

    c = _subclass(**kwargs)
    c.label = label

    return c


# don't squeeze batch index. also tried squeeze(-1) but this may be better
def _default_pred(*args):
    return tuple(a.squeeze(tuple(range(1, len(a.shape)))) for a in args)


# For simplicity, only plotted metrics are kept
class MetricsFrame(Base):
    def __init__(
        self,
        columns: List[Metric],
        flush_every=1,
        index_fn: Callable = lambda batch_num, batches_per_epoch: np.float32(
            batch_num / batches_per_epoch
        ),
        name=None,
        xlabel="epoch",  # not related to board
        train=False,  # used to prefix when plotting
        out_device=torch.device("cpu"),
        pred_fun=_default_pred,
        logger=False,
        plot_on_record=False,
        logging_prefix="",
    ):
        """_summary_

        Args:
            columns (List[Metric]): Instance of torcheval.metrics.metric.Metric with a label property set
            flush_every (int, optional): _description_. Defaults to 1. Flush every n update calls. 0 to never flush.
            index_fn (Callable, optional): _description_. Defaults to lambda batch_num, batches_per_epoch: np.float32(
            batch_num / batches_per_epoch
            plot_on_record (bool, optional): _description_. Defaults to False.
            name (str, optional): Metric frame name, used for plot title. Can be computed from columns by default.
            xlabel (str, optional): Name of column of xlabel, used for plotting. Defaults to "epoch".
            train
            out_device
            pred_fun: defaults to lambda *args: (
                a.squeeze(-1) for a in args
            ), as this is the format most torcheval metrics expect
            logger (Callable | None | False, optional): Will log with self.logger when flush if configured. Set this to None to have Trainer configure this.
        """
        self.save_attr(ignore=["name"])
        self._count = 0

        self._name = name or None

        for c in columns:
            if getattr(c, "label") is None:
                logging.warning(
                    f"{c} does not have a label, imputing from c.__class__.__name__"
                )
                c.label = c.__class__.__name__

        self.dict = {col.label: [] for col in self.columns}
        if xlabel is not None:
            self.dict[xlabel] = []

        self.df = None
        self.board = None

        # see set_flush_unit
        self._flush_every = flush_every
        self._batches_per_epoch = None

    def append(self, *columns: Metric):
        assert all((v == [] for v in self.dict.values())) and self.df is None
        self.columns.extend(columns)
        self.dict = {col.label: [] for col in self.columns}
        if self.xlabel is not None:
            self.dict[self.xlabel] = []

    def rename(self, from_label: str, to_label: str):
        assert all((v == [] for v in self.dict.values())) and self.df is None
        for c in self.columns:
            if c.label == from_label:
                c.label = to_label
                self.dict[to_label] = self.dict.pop(from_label)
                break

    def flush(self, index=None, log_metric=True):
        """Computes and resets all columns.
        If self.logger is configured, will also log computed values (requires xlabel to be set).

        Args:
            index (int): index to associate with row
        """
        should_log = log_metric and callable(self.logger)
        log_dict = {}
        if index is None:
            assert self.xlabel is None
        for c in self.columns:
            val = self.to_out(c.compute())
            self.dict[c.label].append(val)
            if should_log:
                log_dict[self.logging_prefix + c.label] = val
            c.reset()
        if self.xlabel is not None:
            self.dict[self.xlabel].append(index)
            if should_log:
                log_dict[self.xlabel] = index
                self.logger(log_dict)

    def compute(self):
        """Computes columns

        Args:
            index (int): index to associate with row
        """
        return {c.label: c.compute() for c in self.columns}

    def reset(self):
        for c in self.columns:
            c.reset()

    def clear(self):
        for c in self.columns:
            c.reset()
        self.dict = {col.label: [] for col in self.columns}
        if self.xlabel is not None:
            self.dict[self.xlabel] = []

        self.df = None

    def to(self, device):
        for m in self.columns:
            try:
                m.to(device)
            except:  # noqa: E722
                pass

    def to_out(self, val):
        if isinstance(val, torch.Tensor):
            return val.tolist()  # do we want lists or arrays by default?
        return val

    @property
    def name(self):
        return self._name or ", ".join([col.label for col in self.columns]) + (
            " (training)" if self.train else " (validation)"
        )

    @torch.inference_mode
    def update(self, *args, batch_num=None):
        args = self.pred_fun(*args)
        for c in self.columns:
            c.update(*args)

        self._count += 1

        if self._flush_every != 0 and self._count % self._flush_every == 0:
            index = (
                self._count
                if self._batches_per_epoch is None or batch_num is None
                else self.index_fn(batch_num, self._batches_per_epoch)
            )  # maybe should not be optional
            self.flush(index)

    def set_flush_unit(self, batch_num):
        self._flush_every = min(1, int(self.flush_every * batch_num))
        self._batches_per_epoch = batch_num

    def init_plot(self, title=None, **kwargs):
        """Convenience method to create a board linked to this to draw on"""
        title = self.name if title is None else title
        if self.board is None:
            logging.info(f"Creating plot for {self.name}")
            self.board = ProgressBoard(xlabel=self.xlabel, title=title, **kwargs)
            self.board.add(self)
        return self.board

    def plot(self, df=False):
        """Plots our graph on our board"""
        assert self.xlabel is not None
        self.init_plot()

        self.board.ax.clear()  # Clear off other graphs such as that may also be associated to our board.
        if df:  # draw dataframe
            self.board._draw(self.df, self.xlabel)
        else:
            logging.info(f"Displaying dictionary of {self.name}")
            self.board._draw(self.dict, self.xlabel)

    def record(self):
        if getattr(self, "df") is None:
            self.df = pl.DataFrame(self.dict)

        new_df = pl.DataFrame(self.dict)
        try:
            self.df = self.df.extend(new_df)
            self.dict = {k: [] for k, _ in self.dict.items()}
        except pl.ShapeError:
            logging.info(
                "ShapeError encountered. One or more columns may not compute as scalars. Attempting overwrite of self.df."
            )
            self.df = new_df
        if self.plot_on_record:
            self.plot(df=True)
        return self.df


# todo: set title and y_label
class ProgressBoard(Base):
    def __init__(
        self,
        width=800,
        height=600,
        xlim=None,
        ylim=None,
        title=None,
        xlabel="X",
        ylabel="Y",
        xscale="linear",
        yscale="linear",
        labels=None,
        display=True,
        draw_every=5,  # draw every n epochs
        update_df_every=5,  # update df ever n points
        interactive=None,
        save=False,
        train_prefix="train/",
    ):
        self.save_attr()

        ## Graphical params
        sns.set_style("whitegrid")

        self.fig, self.ax = plt.subplots(
            figsize=(width / 100, height / 100)
        )  # Adjust size for Matplotlib

        if not isinstance(interactive, bool):
            self.interactive = is_notebook()
        if self.interactive:
            self.dh = IPython.display.display(self.fig, display_id=True)

        # Set log scaling based on the provided xscale and yscale
        if xscale == "log":
            self.ax.set_xscale("log")
        if yscale == "log":
            self.ax.set_yscale("log")
        if title:
            self.ax.set_title(title)

        ## Initialize data structures
        assert draw_every >= 1

        if labels:
            self.schema = pl.Schema(
                [(xlabel, pl.Float64), (ylabel, pl.Float64), ("Label", pl.Enum(labels))]
            )

        else:
            self.schema = pl.Schema(
                [(xlabel, pl.Float64), (ylabel, pl.Float64), ("Label", pl.String())]
            )

        self._count = 0
        self._points_count = 0

        self.mfs = []

        self.data = pl.DataFrame(
            schema=self.schema, data=[]
        )  # To store processed data (mean of every n)

        self._clear_buffer()

        ## Further config
        plt.close()

        # legend_labels = []
        # for orbit in self.data['Label'].unique():
        #     legend_labels.append(f"{orbit}")

        # handles, _ = self.ax.get_legend_handles_labels()
        # self.ax.legend(handles, legend_labels, loc="lower left", bbox_to_anchor=(1.01, 0.29), title="Orbit")

    def _redef_ax(self):
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)
        self.ax.set_yscale(self.yscale)

    def _draw(self, data, xlabel):
        if isinstance(data, pl.DataFrame):
            for col in data.columns:
                if col != xlabel:
                    sns.lineplot(
                        x=xlabel, y=data[col], label=col, data=data, ax=self.ax
                    )
        elif isinstance(data, Dict):
            for col in data.keys():
                if col != xlabel and data[col]:
                    sns.lineplot(x=xlabel, y=col, label=col, data=data, ax=self.ax)
        else:
            raise TypeError
        self._redef_ax()

    def draw_mfs(self, force=False):
        self._count += 1

        if not force and (not self.display or self._count % self.draw_every) != 0:
            return
        self.ax.clear()

        for mf in self.mfs:
            if mf.train:
                self._draw(
                    {
                        self.train_prefix + key if key != mf.xlabel else key: value
                        for key, value in mf.dict.items()
                    },
                    mf.xlabel,
                )
            else:
                self._draw(mf.dict, mf.xlabel)
        self._redef_ax()
        self.iupdate()

    def add_mf(self, *mfs: MetricsFrame):
        for mf in mfs:
            self.mfs.append(mf)
            if mf.board is None:
                mf.board = self

    def _clear_buffer(self):
        self.buffer = {k: [] for k in (self.xlim, self.ylim, "Labels")}

    def close(self):
        plt.close()

    # todo: improved aggregation
    def draw_points(self, x, y, label, every_n=5, force=False, clear=False):
        """Update plot with new points (arrays) and redraw."""

        self.buffer[self.xlim].append(x)
        self.buffer[self.ylim].append(x)
        self.buffer["Labels"].append(label)

        if len(self.buffer["Labels"]) >= self.update_df_every or force:
            new_df = pl.DataFrame(self.buffer)
            self.data = self.data.extend(new_df)
            self._clear_buffer()

        if not self.display:
            return

        # # X-axis values (common for all lines)
        # x_values = [0, 1, 2, 3, 4]

        # # Plot using the dictionary directly
        # sns.lineplot(data=data, palette='tab10')

        # # Setting x-values explicitly
        # plt.xticks(ticks=range(len(x_values)), labels=x_values)
        # Redraw the plot

        if True:
            if clear:
                self.ax.clear()
                sns.scatterplot(x=self.xlabel, y=self.ylabel, hue="Label", data=new_df)
            else:
                sns.scatterplot(
                    x=self.xlabel, y=self.ylabel, hue="Label", data=self.data
                )

        else:
            for label in self.labels:
                label_data = self.data.filter(pl.col("Label") == label)
                sns.lineplot(
                    x="X",
                    y="Y",
                    data=label_data,
                    ax=self.ax,
                    label=label,
                    linestyle=self.line_styles[label],
                    color=self.line_colors[label],
                )

        self.iupdate()

    def flush(self):
        for key in self.raw_points.keys():
            self.draw([], [], key, force=True)
        self.draw_mfs(force=True)
        if self.save:
            plt.savefig("updated_plot.png")

    def iupdate(self):
        if self.interactive:
            self.dh.update(self.fig)


def plot_2dheatmap(arr, close_last=True, **kwargs):
    # if close_last:  # useful for single plots
    #     plt.close("all")
    plt.figure()
    sns.heatmap(
        [[int(el) for el in row] for row in arr],
        annot=True,
        fmt=".2f",
        cmap="Blues",
    )
    kwargs.setdefault("xlabel", "Predicted")
    kwargs.setdefault("ylabel", "Ground truth")
    plt.gca().set(**kwargs)
    plt.show()
