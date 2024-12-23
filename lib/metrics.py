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
        name=None,
        xlabel=None,  # not related to board
        unit: Optional[int] = None,
        train=False,  # used to prefix when plotting
        out_device=torch.device("cpu"),
        pred_fun=_default_pred,
        logger=False,
        logging_prefix="",
    ):
        """_summary_

        Args:
            columns (List[Metric]): Instance of torcheval.metrics.metric.Metric with a label property set
            flush_every (int, optional): _description_. Defaults to 1. Flush every n update calls. 0 to never flush.
            name (str, optional): Metric frame name, used for plot title. Can be computed from columns by default.
            xlabel (str, optional): Name of column of xlabel, used for plotting. Defaults to "epoch" or "batch".
            unit: If None, trainer will configure this frame so that all provided values are interpreted in epoch units. None behaves so that unit=num_train_batches in a Trainer with mf_epoch_units=True and 1 otherwise.
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
        if xlabel is None:
            self.xlabel = "batch"  # Technically we could use "" or None to denote no xlabel after initialization xlabel is not None, but we use "" to denote no xlabel for clarity
        if self.xlabel:
            self.dict[self.xlabel] = []

        self.df = None
        self.board = None

        self._flush_every = flush_every

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

    def flush(self, idx=None, log_metric=True, keep_idx=False):
        """Computes and resets all columns.
        If self.logger is configured, will also log computed values (requires xlabel to be set).
        Will use self._count for the idx by default, similar to update(). NOTE that you want
        to be consistent between providing idx or not throughout these two functions.
        Explicitly, using ._count allows us to accumulate statistics across multiple training steps.
        Passing in the index explicitly allows continuing the index from continued training runs.
        That is, choosing between one or the other depends on if you want to
        reuse the metric frame (single graph) vs reuse the setup (multiple graphs).

        Args:
            idx (int): idx to associate with row
            keep_idx (bool): Whether to scale idx by configured units.
        """
        should_log = log_metric and callable(self.logger)
        log_dict = {}

        for c in self.columns:
            val = self.to_out(c.compute())
            self.dict[c.label].append(val)
            if should_log:
                log_dict[self.logging_prefix + c.label] = val
            c.reset()
        if self.xlabel:
            if not keep_idx:
                idx = self.get_idx_per_unit(idx)
            self.dict[self.xlabel].append(idx)
            if should_log:
                log_dict[self.xlabel] = idx
                self.logger(log_dict)

    def compute(self):
        """Computes columns

        Args:
            idx (int): idx to associate with row
        """
        return {c.label: c.compute() for c in self.columns}

    def reset(self):
        for c in self.columns:
            c.reset()

    def clear(self):
        for c in self.columns:
            c.reset()
        self.dict = {col.label: [] for col in self.columns}
        if self.xlabel:
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

    # refactored to allow easier override
    def get_idx_per_unit(self, idx):
        if (
            self.unit is None or self.unit == 1
        ):  # Default unit acts as 1, is this inefficient to check...
            return idx or self._count
        else:
            return (idx or self._count) / self.unit

    @torch.inference_mode
    def update(self, *args, idx=None):
        args = self.pred_fun(*args)
        for c in self.columns:
            c.update(*args)

        self._count += 1

        if self._flush_every != 0 and self._count % self._flush_every == 0:
            if not self.xlabel:
                self.flush()
            else:
                self.flush(idx)

    def set_unit(
        self, unit, xlabel, scale_flush_interval=True
    ):  # trainer calls this with xlabel=epoch
        if unit:
            if scale_flush_interval:
                self._flush_every = max(1, int(self.flush_every * unit))
            self.unit = unit
        if xlabel:
            if self.xlabel:
                self.dict[xlabel] = self.dict.pop(self.xlabel)
            else:
                assert all(
                    isinstance(value, list) and not value
                    for value in self.dict.values()
                )  # should not set xlabel for initialized dict
                self.dict[xlabel] = []
            self.xlabel = xlabel

    def init_plot(self, title=None, **kwargs):
        """Convenience method to create a board linked to this to draw on"""
        title = self.name if title is None else title
        if self.board is None:
            logging.info(f"Creating plot for {self.name}")
            self.board = ProgressBoard(xlabel=self.xlabel, title=title, **kwargs)
            self.board.add_mf(self)
        return self.board

    def plot(self, df=False, kind="line"):
        """Plots our graph on our board"""
        assert self.xlabel
        self.init_plot()
        dbg(self.board)

        self.board.ax.clear()  # Clear off other graphs such as that may also be associated to our board.
        if df:  # draw dataframe
            self.board._draw(self.df, self.xlabel, update=True, kind=kind)
        else:
            logging.info(f"Displaying dictionary of {self.name}")
            self.board._draw(self.dict, self.xlabel, update=True, kind=kind)

    def record(self, plot=False):
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
            if plot:
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

        self.init_plot()

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

        # legend_labels = []
        # for orbit in self.data['Label'].unique():
        #     legend_labels.append(f"{orbit}")

        # handles, _ = self.ax.get_legend_handles_labels()
        # self.ax.legend(handles, legend_labels, loc="lower left", bbox_to_anchor=(1.01, 0.29), title="Orbit")

    def init_plot(self):
        if (fig := getattr(self, "fig", None)) is not None:
            plt.close(fig)
        self.fig, self.ax = plt.subplots(
            figsize=(self.width / 100, self.height / 100)
        )  # Adjust size for Matplotlib

        if not isinstance(self.interactive, bool):
            self.interactive = is_notebook()
        if self.interactive:
            self.dh = IPython.display.display(self.fig, display_id=True)
            plt.close()  # ipython ipdate handles the plot

        # Set log scaling based on the provided xscale and yscale
        if self.xscale == "log":
            self.ax.set_xscale("log")
        if self.yscale == "log":
            self.ax.set_yscale("log")
        if self.title:
            self.ax.set_title(self.title)

    def _redef_ax(self):
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)
        self.ax.set_yscale(self.yscale)

    def _draw(self, data, xlabel, update=False, kind="line"):
        # not sure how to handle overlapping lines
        if isinstance(data, pl.DataFrame):
            for col in data.columns:
                if col != xlabel:
                    if kind == "scatter":
                        sns.scatterplot(
                            x=xlabel,
                            y=data[col],
                            label=col,
                            data=data,
                            ax=self.ax,
                        )
                    else:
                        sns.lineplot(
                            x=xlabel,
                            y=data[col],
                            label=col,
                            data=data,
                            ax=self.ax,
                        )
        elif isinstance(data, Dict):
            for i, col in enumerate(data.keys()):
                if col != xlabel and data[col]:
                    if kind == "scatter":
                        sns.scatterplot(
                            x=xlabel,
                            y=col,
                            label=col,
                            data=data,
                            ax=self.ax,
                        )
                    else:
                        sns.lineplot(
                            x=xlabel,
                            y=col,
                            label=col,
                            data=data,
                            linewidth=1.5 - 0.2 * i,
                            ax=self.ax,
                        )
        else:
            raise TypeError

        if update:
            self._redef_ax()
            self.iupdate()

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


# plot 2d array


def plot_2dheatmap(arr, close_last=True, ticklabels="auto", **kwargs):
    # if close_last:  # useful for single plots
    #     plt.close("all")
    # alternative to using plt.subplots because sns uses gca() as default ax
    plt.figure()
    sns.heatmap(
        [[int(el) for el in row] for row in arr],
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=ticklabels,
        yticklabels=ticklabels,
        ax=plt.gca(),
    )
    kwargs.setdefault("xlabel", "Predicted")
    kwargs.setdefault("ylabel", "Ground truth")
    plt.gca().set(**kwargs)
    plt.show()


def plot_hist(list, kde=True, **kwargs):
    plt.figure()
    binwidth = 1 if isinstance(list[0], int) else None
    sns.histplot(list, binwidth=binwidth, kde=False)
    kwargs.setdefault("xlabel", "Value")
    kwargs.setdefault("ylabel", "Count")
    plt.gca().set(**kwargs)


def plot_bar(
    dict, stack_label="variant", xlabel="class", ylabel="count", x_rot=0, **kwargs
):
    fig, ax = plt.subplots()

    if isinstance(next(iter(dict.keys())), tuple):
        df = pd.DataFrame(list(dict.items()), columns=["Tuple", "Count"])
        df[["a", stack_label]] = pd.DataFrame(df["Tuple"].tolist(), index=df.index)

        # Pivot the DataFrame to get counts by 'a' and 'b'
        pivot_df = df.pivot_table(
            index="a", columns=stack_label, values="Count", aggfunc="sum", fill_value=0
        )
        # Plot a stacked bar chart
        pivot_df.plot(kind="bar", stacked=True, rot=x_rot, ax=ax)
        # ax.tick_params(axis="x", rotation=0) doesn't work
    else:
        sns.barplot(dict)

    kwargs.setdefault("xlabel", "Class")
    kwargs.setdefault("ylabel", "Count")
    ax.set(**kwargs)
    plt.show()


# plot dict


# def plot_points(*point_lists, default_time=None, **kwargs):
#     for point_list in point_lists:
#         if isinstance(point_list[0], tuple):
#             data_points, time_values = zip(*point_list)
#             plt.plot(time_values, data_points)
#         else:
#             time = (
#                 default_time[: len(point_list)]
#                 if default_time is not None
#                 else np.arange(1, len(point_list) + 1)
#             )
#             plt.plot(time, point_list)  # Plot with range on x-axis

#     kwargs.setdefault("xlabel", "Value")
#     kwargs.setdefault("ylabel", "Index")
#     plt.gca().set(**kwargs)

#     # Show the plot
#     plt.show()


def plot_points(
    *point_lists,
    kind="scatter",  # or line
    default_time=None,
    set_labels=None,
    **kwargs,
):
    fig, ax = plt.subplots()
    kwargs.setdefault("xlabel", "Value")
    kwargs.setdefault("ylabel", "Index")
    legend = "Legend"

    all_data = []

    set_labels = set_labels or [f"Set{i}" for i in range(len(point_lists))]

    for point_list, label in zip(point_lists, set_labels):
        if len(point_list) == 0:
            if vb(7):
                print("label is empty")
            continue
        if isinstance(point_list[0], tuple):
            data_points, time_values = zip(*point_list)
            df = pd.DataFrame(
                {
                    kwargs["xlabel"]: time_values,
                    kwargs["ylabel"]: data_points,
                    legend: [label] * len(point_list),
                }
            )
        else:
            time = (
                default_time[: len(point_list)]
                if default_time is not None
                else np.arange(1, len(point_list) + 1)
            )
            df = pd.DataFrame(
                {
                    kwargs["xlabel"]: time,
                    kwargs["ylabel"]: point_list,
                    legend: [label] * len(point_list),
                }
            )

        all_data.append(df)

    combined_df = pd.concat(all_data)

    # Plot using Seaborn
    if kind == "scatter":
        sns.scatterplot(
            data=combined_df,
            x=kwargs["xlabel"],
            y=kwargs["ylabel"],
            hue=legend,
            style=legend,
            ax=ax,
        )  # s=100 for size
    else:
        sns.lineplot(
            data=combined_df,
            x=kwargs["xlabel"],
            y=kwargs["ylabel"],
            hue=legend,
            style=legend,
            ax=ax,
        )
    # alternatively relplot creates its own figure

    ax.set(**kwargs)

    # Show the plot
    plt.show()