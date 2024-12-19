from collections import Counter
import random
import re
from sklearn.model_selection import KFold
import torch
from lib import dfs
from lib.datasets.datasets import SubDataset
from lib.utils import *
import polars as pl
import numpy as np
import torch.utils.data as td
import pandas as pd
from lib.dfs import *
from typing import Tuple
from sklearn.preprocessing import LabelEncoder

import glob


#
@dataclass(kw_only=True)
class DataConfig(Config):
    data: Optional[Any] = None
    dataset: Optional[Any] = None
    batch_size: int = 32
    num_workers: int = field(default_factory=lambda: os.cpu_count() - 2)
    sampler: Optional[Any] = None
    transform: Any = None
    shuffle: Optional[bool] = None


# used in wandb
@dataclass(kw_only=True)
class DataloaderConfig(Config):
    batch_size: int = None
    num_workers: int = None
    sampler: Optional[Any] = None
    shuffle: bool = None
    sampler: Optional[Any] = None
    batch_sampler: Optional[Any] = None
    collate_fn: Optional[Any] = None
    pin_memory: bool = None
    drop_last: bool = None
    timeout: int = None
    worker_init_fn: Optional[Any] = None
    multiprocessing_context: Optional[Any] = None
    generator: Optional[Any] = None
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = None
    pin_memory_device: Optional[str] = None


class Data(Base):
    """The base class of td."""

    def __init__(self):
        super().__init__()

        # For folds
        self._data_inds = None
        self._folds = None
        self._folder = None
        self.processor = None

        # Subclasses define self.data = (X, y), or dataset/(train_set, val_set) directly
        # Be sure to include set_folds

    def _to_dataset(self, data):
        """Converts tensor tuples to dataset"""

        tensors = (to_tensors(x) for x in data)
        return td.TensorDataset(*tensors)

    # prefer not to use this as it doesn't apply to test set
    # by default allow defining scikit processors
    def _fit_transforms(self, tensors):
        if self.processor is not None:
            tensors[0] = self.processor.fit_transform(tensors[0])
            return [lambda x: self.processor.transform(x)]
        else:
            return []

    def _transform(self, tensors, transforms):
        for i, tt in enumerate(zip(tensors, transforms)):
            t, tr = tt
            tensors[i] = tr(t)

    def set_data(
        self,
        *args,
        test=False,
    ):
        data = tuple(
            np.array(a) if isinstance(a, List) else a for a in args
        )  # To allow advanced indexing

        if test:
            self.test_data = data
            self.test_dataset = None
        else:
            self.data = data
            self.dataset = None

    @property
    def data_inds(self):
        if not self._data_inds:
            if self._folds:
                self._data_inds = next(self._folds)
        return self._data_inds or None

    # todo: memory management
    def loaders(
        self,
        train_set=None,
        val_set=None,
        split: Tuple[Union[np.ndarray, List[int]], Union[np.ndarray, List[int]]]
        | None
        | False = None,
        dtypes=[],
        loader_args: List[Dict] = None,
        **kwargs,
    ):
        """returns train_loader, val_loader
        **kwargs: passed to train DataLoader
        split: Tuple of indices that determines how to split the dataset. Prevents splitting if set to False. Controlled by set_folds otherwise.
        loader_args: List of kwargs zipped to loaders. If there are more loaders than loader_args, the last dict is repeated. If not supplied, loader_args=[kwargs].
            - By default, shuffle is False for loaders after the first. If shuffle/sampler is set on self, it will be a default value for the first kwargs only. sampler can be set to True on later args to use the same method as for the first arg.
            - shuffle=True is not set for train if sampler is given.
        """

        def _loaders(*sets):
            nonlocal loader_args
            nonlocal split
            nonlocal kwargs

            # either apply same kwargs to all loaders, or supply a list
            loader_args = loader_args or [kwargs]
            loader_args += [loader_args[-1].copy()] * (len(sets) - len(loader_args))

            # Allow setting some loader args on data class itself
            for ix, la in enumerate(loader_args):
                # Note: batch_sampler is identical to sampler+fixed batch_size
                for key in ["batch_size", "num_workers", "sampler", "shuffle"]:
                    if key not in la.keys():
                        if key == "sampler":
                            if ix == 0:
                                if callable(la.get(key, None)):
                                    la[key] = la[key](self, split[ix])
                                elif getattr(self, key, None):
                                    la[key] = self.sampler(self, split[ix])
                        elif key == "shuffle":
                            if (
                                ix == 0
                                and "sampler" not in la.keys()
                                and not isinstance(sets[ix], td.IterableDataset)
                            ):
                                _shuffle = getattr(self, "shuffle", None)
                                la[key] = (
                                    _shuffle if _shuffle is not None else True
                                )  # shuffle=True shuffles the data after every epoch
                        elif (attr := getattr(self, key, None)) is not None:
                            la[key] = attr

            return tuple(
                td.DataLoader(set, **kwargs) for set, kwargs in zip(sets, loader_args)
            )

        # Allow setting train/val_sets directly
        train_set = (
            train_set
            if isinstance(train_set, td.Dataset)
            else getattr(self, "train_set", None)
        )
        val_set = (
            val_set
            if isinstance(val_set, td.Dataset)
            else getattr(self, "val_set", None)
        )

        if split is None:
            if self.data_inds:
                split = self.data_inds
            else:
                split = False

        train_set = train_set or self.dataset
        if isinstance(train_set, td.Dataset):
            if val_set is not None:
                return _loaders(train_set, val_set)
            elif split is False:
                return _loaders(train_set)
            return _loaders(*(SubDataset(train_set, split[i]) for i in split))

        else:
            # Create Dataset from self.data(*X, y) by converting to tensors
            assert self.data is not None
            if val_set is not None or split is False:
                split = [slice(0, None)]
            dtypes += [torch.float32] * (len(self.data) - len(dtypes))

            train_arr = list(row_index(a, split[0]) for a in self.data)
            # if isinstance(self, ClassifierData):
            #     train_arr[-1] = train_arr[-1].to(torch.int64)
            transforms = self._fit_transforms(train_arr)
            train_set = self._to_dataset(train_arr)

            if val_set is not None:
                return _loaders(train_set, val_set)
            elif split is False:
                return _loaders(train_set)
            else:

                def to_val_set(indices):
                    val_array = [row_index(a, indices) for a in self.data]
                    self._transform(val_array, transforms)
                    return self._to_dataset(val_array)

                return _loaders(train_set, *(to_val_set(ixs) for ixs in split[1:]))

    def test_loader(self, test_set=None, **kwargs):
        test_set = test_set or self.test_dataset or self.test_data
        assert test_set is not None

        if not isinstance(test_set, td.Dataset):
            test_set = self._to_dataset(test_set)

        # Allow setting some loader args on data class itself
        # todo: auto detect best batch size
        for key in ["batch_size", "num_workers"]:
            if (attr := getattr(self, key, None)) is not None:
                kwargs[key] = attr

        return td.DataLoader(test_set, **kwargs)

    def data_range(
        self, loader_index, batch_index, range=None
    ):  # assuming tensors, todo: improve
        loader = iter(self.loaders(shuffle=False)[loader_index])
        outs = next(loader)[batch_index]
        while True:
            try:
                next_batch = next(loader)[batch_index]
                outs = torch.cat((outs, next_batch), dim=0)
            except StopIteration:
                break
        if range:
            return outs[range]
        else:
            return outs

    def set_folds(self, split_method):
        """Configure folds

        Args:
            split_method: from sklearn.model_selection
        """

        self._folder = split_method
        try:
            if self.dataset is not None:
                self._folds = self._folder.split(np.arange(len(self.dataset)))
            if self.data is not None:
                self._folds = self._folder.split(*self.data)
        except AttributeError:
            raise AttributeError("dataset/data not found. Have you called set_data?")

    @property
    def raw_len(self):
        if l := getattr(self, "dataset", None):
            return len(l)
        else:
            try:
                return len(self.data[0])
            except:  # noqa: E722
                raise AttributeError

    # todo: shuffle when train_set is given
    def folds(self, continue_existing=True):
        """Provides an iterator that changes self.data_inds, used by self.loaders(), using iterator provided by self._folds, to be used to loop over folds.

        >>> from sklearn.model_selection import KFold
        >>> t = Data()
        >>> t.data = ([1, 2, 3], [1, 2, 3])
        >>> t.set_folds(KFold(n_splits=3))
        >>> for i in t.folds(): print(t.data_inds)
        (array([1, 2]), array([0]))
        (array([0, 2]), array([1]))
        (array([0, 1]), array([2]))
        """

        def initialize():
            assert self._folder is not None and self.data is not None
            self.set_folds(self._folder)

        if not self._folds or not continue_existing:
            initialize()

        fold_num = 0
        while True:
            try:
                self._data_inds = next(self._folds)
                fold_num += 1
                yield fold_num
            except StopIteration:
                initialize()
                # logging.debug("Reinitializing folds")
                return

    ## Data Preview section

    def sample_batch(
        self,
        batch_index=0,
        train=True,
    ):
        loader = self.loaders()[0 if train else 1]

        for _ in range(batch_index + 1):
            b = next(iter(loader))

        return b

    def preview_batch(self, batch, samples=5, header=True):
        if isinstance(batch, Dict):
            batch_vals = batch.values()
            batch_items = batch.items()
        else:
            batch_vals = batch
            batch_items = enumerate(batch)

        if header:
            # Print shapes for all tensors in the batch
            print("Constituent shapes:")
            for i, tensor in batch_items:
                print(f"batch[{i}]: {tensor.shape}, {tensor.dtype}")

        # Print sample values
        samples = min(samples, len(next(iter(batch_vals))))
        if header:
            print(f"\nFirst {samples} samples:")
        for i in range(samples):
            print(f"\nSample {i}: ")
            for j in batch_vals:
                print(f"\n{j[i].squeeze()}")

    def preview(self, samples=5):
        """
        Preview the data by showing dimensions and sample rows from both training and validation sets.

        Args:
            samples (int): Number of samples to display from each dataset
        """

        def loader_len(loader):
            if isinstance(loader.dataset, td.IterableDataset):
                return "IterableDataset"
            else:
                return f"{len(loader)} batches"

        loaders = self.loaders()

        for i, l in enumerate(loaders):
            print(f"\nLoader {i} ({loader_len(loaders[0])}) Preview:")
            print("-" * 50)
            self.preview_batch(next(iter(l)), samples, header=True)

    def preview_df(self, index):
        dfs.preview_df(self.data[index], head=False)

    def describe_index(
        self, index, batch_index=0, feature_dims=slice(1, None), train=True
    ):
        """
        Provide statistics on features.
        """

        print(f"\n{"Training" if train else "Validation"} Batch[{index}] Statistics:")
        print("-" * 50)
        describe_tensor(
            self.sample_batch(
                batch_index,
                train,
            )[index],
            feature_dims,
        )


class DataFromLoader(Data):
    def __init__(self, train_loader, val_loader, batch_size=32, num_workers=1):
        super().__init__()
        self.save_attr()

        self._folds = None

    def loaders(self, train_set=None, val_set=None, val_kwargs=None, **kwargs):
        return self.train_loader, self.val_loader


class ClassifierData(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_attr()
        # Use set_daata

    def decode_label(self, label):
        """Returns name of original label for interpretation. Usually overriden or provisioned by a method below."""
        if self.le:
            try:
                return self.le.inverse_transform(label)
            except ValueError:
                return self.le.inverse_transform([label])[0]
        return label

    def encode_label(self, label):
        """Returns value of encoded label for interpretation. Usually overriden or provisioned by a method below."""
        if self.le:
            try:
                return self.le.transform(label)
            except ValueError:
                return self.le.transform([label])[0]
        return label

    @property
    def classes(self):
        return len(self.le.classes_)

    def set_data(self, *args, test=False, show_counts=True):
        """Initializes self.data

        Args:
            test (bool, optional): Initialize test set. Defaults to False.
            show_counts (bool, optional): Displays data counts on non-test initialization. Defaults to True.
        """
        self.le = LabelEncoder()
        labels = tolist(args[-1])  # counter needs a hashable type

        data = tuple(np.array(a) if isinstance(a, List) else a for a in args[:-1]) + (
            torch.tensor(self.le.fit_transform(labels), dtype=torch.int64),
        )

        if test:
            self.test_data = data
            self.test_dataset = None
        else:
            self.data = data
            self.dataset = None

        # counts

        self.counts = sorted(
            list(Counter(labels).items()), key=lambda x: self.encode_label(x[0])
        )

        if show_counts and test is False:
            self.show_counts(test=test)

    def show_counts(self, test=False):
        assert all((self.counts, self.data, self.le))
        print("data[-1] counts in label order from 0:")
        for k, v in self.counts:
            print(f"{k}: {v}")

    @classmethod
    def make_inverse_sampler(cls, factor=1):
        """Creates a factory method for creating an weighted random sampler which weights each class according 1/class_size^alpha

        Args:
            factor (int, optional): _description_. Defaults to 1.
        """

        def inner(self, inds, **kwargs):
            nonlocal factor
            assert all((self.data, self.counts))

            def idx_to_label(n):
                return self.data[-1][inds[n]]

            weights = [
                pow(self.counts[idx_to_label(i)][1], -factor) for i in range(len(inds))
            ]

            return td.WeightedRandomSampler(
                weights,
                num_samples=kwargs.pop("num_samples", len(inds)),
                **kwargs,
            )

        return inner


class ImageData(ClassifierData):
    # Use set_data
    def _paths_and_labels_from_folder(
        self,
        folder,
        glob_string,
        label_fn=lambda folder_name: folder_name,
        label_encode=True,
    ):
        paths = []
        labels = []
        for label in os.listdir(folder):
            label_path = os.path.join(folder, label)
            if os.path.isdir(label_path):
                # Get all .jpg files ending with 'Ch3.ome.jpg' in this folder
                image_paths = glob.glob(os.path.join(label_path, glob_string))
                paths.extend(image_paths)
                labels.extend([label_fn(label)] * len(image_paths))
        return paths, labels

    def _paths_and_labels_from_file(self, file, file_label_regex):
        paths = []
        labels = []
        pattern = re.compile(file_label_regex)
        with open(file, "r") as f:
            for line in f.readlines():
                match = pattern.search(line)
                try:
                    paths.append(match.group(1))
                    labels.append(match.group(2))
                except (IndexError, AttributeError):
                    logging.warning(f"Line {line} doesn't match regex")

        return paths, labels

    def visualize(self, imgs, labels=None, grid=(4, 4), title=None):
        """
        input imgs can be single or multiple tensor(s), this function uses matplotlib to visualize.
        Single input example:
        show(x) gives the visualization of x, where x should be a torch.Tensor
            if x is a 4D tensor (like image batch with the size of b(atch)*c(hannel)*h(eight)*w(eight), this function splits x in batch dimension, showing b subplots in total, where each subplot displays first 3 channels (3*h*w) at most.
            if x is a 3D tensor, this function shows first 3 channels at most (in RGB format)
            if x is a 2D tensor, it will be shown as grayscale map

        Multiple input example:
        show(x,y,z) produces three windows, displaying x, y, z respectively, where x,y,z can be in any form described above.
        """

        import numpy as np
        import matplotlib.pyplot as plt

        flag = True
        if isinstance(imgs, (torch.Tensor, np.ndarray)):
            imgs = imgs.detach().cpu()

            if imgs.dim() == 4:  # 4D tensor
                bz = imgs.shape[0]
                c = imgs.shape[1]
                if bz == 1 and c == 1:  # single grayscale image
                    imgs = [imgs.squeeze()]
                elif bz == 1 and c == 3:  # single RGB image
                    imgs = imgs.squeeze()
                    imgs = [imgs.permute(1, 2, 0)]
                elif bz == 1 and c > 3:  # multiple feature maps
                    imgs = imgs[:, 0:3, :, :]
                    imgs = [imgs.permute(0, 2, 3, 1)[:]]
                    print(
                        "warning: more than 3 channels! only channels 0,1,2 are preserved!"
                    )
                elif bz > 1 and c == 1:  # multiple grayscale images
                    imgs = imgs.squeeze()
                elif bz > 1 and c == 3:  # multiple RGB images
                    imgs = imgs.permute(0, 2, 3, 1)
                elif bz > 1 and c > 3:  # multiple feature maps
                    imgs = imgs[:, 0:3, :, :]
                    imgs = imgs.permute(0, 2, 3, 1)[:]
                    print(
                        "warning: more than 3 channels! only channels 0,1,2 are preserved!"
                    )
                else:
                    raise Exception("unsupported type!  " + str(imgs.size()))
                flag = False
            else:  # single image
                imgs = [imgs]

        if flag:

            def process_img(img):
                if img.dim() == 3:
                    c = img.shape[0]
                    if c == 1:  # grayscale
                        img = img.squeeze()
                    elif c == 3:  # RGB
                        img = img.permute(1, 2, 0)
                    else:
                        raise Exception("unsupported type!  " + str(img.size()))
                img = img.numpy().squeeze()
                return img

            imgs = [process_img(img) for img in imgs]

        if len(imgs) == 1:
            fig, axs = plt.subplots(title=title)
            axs = [axs]
            labels = k_level_list(labels, k=1)
        else:
            fig, axs = plt.subplots(*grid)
            axs = axs.flatten()
        fig.suptitle(title)  # dunno how to hide figure numbers
        if labels is not None:
            for ax, img, label in zip(axs, imgs, labels):
                ax.imshow(img, cmap="gray")
                ax.axis("off")
                ax.set_title(
                    label if isinstance(label, str) else self.decode_label(label)
                )
        else:
            for ax, img in zip(axs, imgs):
                ax.imshow(img, cmap="gray")
                ax.axis("off")

        plt.tight_layout()

    # def visualize_class(self, class_idx, idx=0):
    #     idx = np.where(self.data[1] == class_idx)[0][idx]
    #     img = self.loaders()[0].dataset[idx][0]
    #
    #     self.visualize(img, f"{self.decode_label(class_idx)} ({idx})")

    # expects a list for batch
    def visualize_class(self, class_idx, samples=1):
        train = iter(self.loaders()[0])
        batch_no = 0
        imgs = []
        labels = []
        orig_samples = samples
        while samples > 0:
            batch = next(train)
            inds = torch.nonzero(batch[1] == class_idx)
            num = min(samples, len(inds))
            samples -= num
            for i in range(num):
                imgs.append(batch[0][inds[i]].squeeze(0))
                labels.append(f"(batch {batch_no}, {inds[i].item()})")

            batch_no += 1
        return self.visualize(
            imgs,
            labels,
            grid=factorize(orig_samples),
            title=self.decode_label(class_idx),
        )


def describe_tensor(tensor, feature_dims):
    if isinstance(feature_dims, slice):
        feature_dims = list(range(len(tensor.shape))[feature_dims])
    else:
        feature_dims = [
            idx if idx >= 0 else len(tensor.shape) + idx for idx in feature_dims
        ]

    complement_dims = [i for i in range(tensor.dim()) if i not in feature_dims]

    min_value = compute_inverse_permutation(
        feature_dims, tensor.amin(dim=complement_dims)
    )
    max_value = compute_inverse_permutation(
        feature_dims, tensor.amax(dim=complement_dims)
    )
    mean_value = compute_inverse_permutation(
        feature_dims, tensor.mean(dim=complement_dims)
    )
    std_value = compute_inverse_permutation(
        feature_dims, tensor.std(dim=complement_dims)
    )

    # Print the results
    print(f"\nfeature_dims {feature_dims}: ")
    print(f"Type: {tensor.dtype}")
    print("Min:")
    print(min_value)
    print("Max:")
    print(max_value)
    print("Mean:")
    print(mean_value)
    print("Std:")
    print(std_value)
