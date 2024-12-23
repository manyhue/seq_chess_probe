import inspect
from contextlib import contextmanager
import os
import numpy as np
import pandas as pd
import torch
import polars as pl
import math
from collections.abc import Mapping, Iterable
from dataclasses import is_dataclass, fields
from functools import reduce
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union


class RunEveryNth:
    def __init__(self, n, func):
        self.n = n
        self.func = func
        self.counter = 0

    def __call__(self, *args, **kwargs):
        self.counter += 1
        if self.counter % self.n == 0:
            return self.func(*args, **kwargs)
        return None

def is_notebook():
    # credit -> https://stackoverflow.com/a/39662359
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def apply_kwargs(func, kwargs):
    """
    Filters a dictionary of keyword arguments to only include those
    accepted by the given function, then applies them to the function.

    Parameters:
        func (callable): The function to which arguments will be applied.
        kwargs (dict): Dictionary of keyword arguments.

    Returns:
        The result of calling `func` with the filtered keyword arguments.
    """
    sig = inspect.signature(func)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(**valid_kwargs)


# autodecorator?
# # Define a custom metaclass that applies @dataclass
# class AutoDataclassMeta(type):
#     def __new__(cls, name, bases, dct):
#         # Create the class
#         new_cls = super().__new__(cls, name, bases, dct)
#         # Apply the @dataclass decorator automatically
#         return dataclass(new_cls)

# class Config(metaclass=AutoDataclassMeta):


@dataclass
class Config:
    @classmethod
    def _filter_kwargs(cls, *dicts, **kwargs):
        def arg_to_dict(d):
            if isinstance(d, type):  # Check if d is a class
                return vars(d)
            #  Returns False for some reason
            # if isinstance(d, Config):
            #     d = d.__dict__
            if isinstance(d, Dict):
                return d
            else:
                return d.__dict__

        all_dicts = tuple(arg_to_dict(d) for d in dicts) + (kwargs,)

        merged_dict = {
            key: value
            for d in all_dicts
            for key, value in d.items()
            if key in cls.__dataclass_fields__
        }

        return merged_dict

    def update(self, *dicts, **kwargs):
        """Use one or more positional Dict/Config/Class's and/or kwargs to update only existing attributes"""
        for key, value in self._filter_kwargs(*dicts, **kwargs).items():
            setattr(self, key, value)
        return self

    @classmethod
    def create(cls, *dicts, **kwargs):
        """Use one or more positional Dict/Config/Class's and/or kwargs to fill in the fields of and create a Config."""
        return cls(**cls._filter_kwargs(*dicts, **kwargs))

    @classmethod
    def dict_from(cls, *dicts, **kwargs):
        """create but outputs a Dict"""
        return asdict(cls.create(*dicts, **kwargs))

    # def show(self, indent=4):
    #     """
    #     Pretty prints a (possibly deeply-nested) dataclass.
    #     Each new block will be indented by `indent` spaces (default is 4).
    #     """
    #     print(stringify(self, indent))

    def dict(self, include_none=False):
        return {k: v for k, v in asdict(self).items() if include_none or v is not None}

    # a hack disguising the fact that we diverge from __dataclass_fields__, but its just for updating dict()
    # not sure about whether del self.__dict__[key] does anything
    def _del(self, key, warn=True):
        if key in self.__dict__:
            setattr(self, key, None)
        else:
            if warn:
                print(f"Warning: Key '{key}' not found.")


class Base:
    def save_parameters(self, ignore=[]):
        """Save function arguments into self.p"""

        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.p = {
            k: v
            for k, v in local_vars.items()
            if k not in set(ignore + ["self"]) and not k.startswith("_")
        }
        # for k, v in self.hparams.items():
        # setattr(self, k, v)

    def save_attr(self, ignore=[], clobber=True, expand_kwargs=True):
        """Save function arguments into class attributes."""

        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.p = {
            k: v
            for k, v in local_vars.items()
            if k not in set(ignore + ["self"]) and not k.startswith("_")
        }
        if expand_kwargs:
            kwargs = self.p.pop("kwargs", None)
            if isinstance(kwargs, Dict):
                for k, v in kwargs.items():
                    if k not in set(ignore + ["self"]) and not k.startswith("_"):
                        self.p[k] = v
        for k, v in self.p.items():
            if clobber or getattr(self, k, None) is None:
                setattr(self, k, v)

    def save_config(self, c: Config, ignore=[], clobber=True):
        for k, v in c.__dict__.items():
            if k not in ignore:
                if clobber or getattr(self, k, None) is None:
                    setattr(self, k, v)
        self.c = c

    @property
    def device(self):
        try:
            return self.gpus[0]
        except:  # noqa: E722
            return "cpu"

    def defined_or_self(self, variable_names):
        """
        Checks if the variable corresponding to `variable_name` is None.
        If it's None, sets it to the value of `self.variable_name` using getattr(self, variable_name).

        Args:
            variable_name (str): The name of the variable to check and potentially update.

        Returns:
            The value of the variable, or `None` if it's not defined.
        """
        # Get the current local variables in the function
        frame = inspect.currentframe().f_back
        local_vars = frame.f_locals

        if not isinstance(variable_names, list):
            variable_names = [variable_names]

        for vn in variable_names:
            # Check if the variable is defined and is None
            value = local_vars.get(vn, None)

            if value is None:
                # If it's None, set it to self.variable_name using getattr(self, variable_name)
                value = getattr(self, vn, None)
                local_vars[vn] = value


def get_gpus(gpus: int | List[int] = -1, vendor="cuda"):
    """Given num_gpus or array of ids, returns a list of torch devices

    Args:
        gpus (int | List[int], optional): [] for cpu. Defaults to -1 for all gpus.
        vendor (str, optional): vendor_string. Defaults to "cuda".

    Returns:
        _type_: _description_
    """
    if isinstance(gpus, list):
        assert [int(i) for i in gpus]
    elif gpus == -1:
        gpus = range(torch.cuda.device_count())
    else:
        assert gpus <= torch.cuda.device_count()
        gpus = range(gpus)
    return [torch.device(f"{vendor}:{i}") for i in gpus]


@contextmanager
def change_dir(target_dir):
    original_dir = os.getcwd()
    try:
        os.chdir(target_dir)
        yield
    finally:
        os.chdir(original_dir)


def mean_of_dicts(dict_list):
    keys = dict_list[0].keys()
    total_dict = reduce(
        lambda acc, val: {key: acc.get(key, 0) + val.get(key, 0) for key in keys},
        dict_list,
    )
    mean_dict = {key: total_dict[key] / len(dict_list) for key in keys}

    return mean_dict


def k_level_list(q, k=1):
    test = q
    while k > 0 and isinstance(test, list):
        if len(test) == 0:
            k -= 1
            break
        test = test[0]
        k -= 1
    for _ in range(k):
        q = [q]
    return q


def compute_inverse_permutation(lst: List[int], tensor: torch.Tensor = None):
    """Given a list of indices, computes the inverse permutation
    If a tensor is supplied, will apply the permutation to the tensor so that the order of dimensions in the tensor matches the order given by permutation corresponding to the list

    Args:
        lst (List):
        tensor (torch.Tensor, optional): . Defaults to None.

    Returns:
        _type_: Corresponding inverse permutation
    """
    if tensor is not None:
        lst = [idx if idx >= 0 else len(tensor.shape) + idx for idx in lst]
    sorted_lst = sorted(lst)
    value_to_index = {v: i for i, v in enumerate(sorted_lst)}

    n = len(lst)
    inverse_permutation = [0] * n
    for i in range(n):
        inverse_permutation[value_to_index[lst[i]]] = i

    if tensor is not None:
        return tensor.permute(inverse_permutation)

    return tuple(inverse_permutation)


def inverse_permute_tensor(lst: List, tensor: torch.Tensor):
    """Given a list of indices, computes the inverse permutation
    If a tensor is supplied, will apply the permutation to the tensor so that the order of dimensions in the tensor matches the order given by permutation corresponding to the list

    Args:
        lst (List)
        tensor (torch.Tensor)

    Returns:
        _type_: Corresponding inverse permutation
    """
    m = len(tensor.shape)
    lst = [idx if idx >= 0 else m + idx for idx in lst]

    inverse_permutation = [None] * m
    for i in range(len(lst)):
        inverse_permutation[lst[i]] = i
    c = len(lst)
    for i, e in enumerate(inverse_permutation):
        if e is None:
            inverse_permutation[i] = c
            c += 1

    tensor.permute(inverse_permutation)

    return tuple(inverse_permutation)


def vb(n):
    """Shorthand to allow: if vb(n): print()"""
    try:
        return verbosity >= n  # type: ignore
    except NameError:
        return False


# logging alias
def ll(*args):
    import logging

    return logging.getLogger("ll")


def dbg(*args):
    frame = inspect.currentframe().f_back
    print(f"funcname = {frame.f_code.co_name} -", *args)


# Categoricals should be dealt with in initialization, operations that may pollute the validation set are generally numeric and thus can be applied after.


def to_tensors(arr, **kwargs):
    """Coerce into Tensor"""
    if isinstance(arr, torch.Tensor):
        return arr.to(**kwargs)
    kwargs.setdefault("dtype", torch.float32)  # np is 64 but torch is 32
    if isinstance(arr, np.ndarray):
        return torch.tensor(arr, **kwargs)
    if isinstance(arr, pl.DataFrame):
        return arr.to_torch(**kwargs)
    if isinstance(arr, pd.DataFrame):
        return torch.tensor(arr.values, **kwargs)


def row_index(arr, indices):
    """Index rows of various types

    Args:
        arr: array_type or table
        indices

    Returns:
        array_type
    """
    if isinstance(arr, pd.DataFrame):
        return arr.iloc[indices, :].values
    elif isinstance(arr, pl.DataFrame):
        try:
            return arr.to_torch(dtype=pl.Float32)[indices]
        except:  # mixed requires setting correct pl types i.e. int64, float32
            return arr.to_torch()[indices]
    elif isinstance(arr, torch.Tensor):
        return arr[indices]
    else:
        return np.array(arr)[indices]


def factorize(n):
    for i in range(math.isqrt(n), 0, -1):
        d, r = divmod(n, i)
        if r == 0:
            return (d, i)


def stringify(obj, indent=4, _indents=0):
    if isinstance(obj, str):
        return f"'{obj}'"

    if not is_dataclass(obj) and not isinstance(obj, (Mapping, Iterable)):
        return str(obj)

    this_indent = indent * _indents * " "
    next_indent = indent * (_indents + 1) * " "
    start, end = (
        f"{type(obj).__name__}(",
        ")",
    )  # dicts, lists, and tuples will re-assign this

    if is_dataclass(obj):
        body = "\n".join(
            f"{next_indent}{field.name}="
            f"{stringify(getattr(obj, field.name), indent, _indents + 1)},"
            for field in fields(obj)
        )

    elif isinstance(obj, Mapping):
        if isinstance(obj, dict):
            start, end = "{}"

        body = "\n".join(
            f"{next_indent}{stringify(key, indent, _indents + 1)}: "
            f"{stringify(value, indent, _indents + 1)},"
            for key, value in obj.items()
        )

    else:  # is Iterable
        if isinstance(obj, list):
            start, end = "[]"
        elif isinstance(obj, tuple):
            start = "("

        body = "\n".join(
            f"{next_indent}{stringify(item, indent, _indents + 1)}," for item in obj
        )

    return f"{start}\n{body}\n{this_indent}{end}"


def get_project_name():
    import subprocess

    repo_name = (
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode("utf-8")
        .strip()
    )
    repo_name = os.path.basename(repo_name)

    caller_frame = inspect.stack()[1]
    caller_file = caller_frame.filename
    filename = os.path.splitext(os.path.basename(caller_file))[0]
    return f"{repo_name}_{filename}"


def tolist(series):
    if isinstance(series, pl.Series):
        return series.to_list()
    elif isinstance(series, pd.Series):
        return series.to_list()
    elif isinstance(series, np.ndarray):
        return series.tolist()
    elif isinstance(series, torch.Tensor):
        return series.tolist()
    else:
        return list(series)


def _grad(out, inp):
    return torch.autograd.grad(
        out,
        inp,
        grad_outputs=torch.ones_like(inp),
        create_graph=True,
        allow_unused=True,
    )[0]


def numpy_flatten(x):
    x = x.numpy() if isinstance(x, torch.Tensor) else x
    return x.flatten()
