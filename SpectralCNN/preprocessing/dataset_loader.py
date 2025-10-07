import sys
import torch
from pytorch_lightning import LightningDataModule
from typing import List, Dict, Optional
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import copy
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split
from typing import Sequence, List, Tuple
from collections import defaultdict
import random
import logging
import pandas as pd
import json
from typing import Optional
from thesis.utils.tools import build_transforms_pipeline
from thesis.utils.mappings import LabelMapper
import hashlib
import itertools
from pytorch_lightning.trainer.states import TrainerFn

# import time # for debugging and runtime calculations (to find out where bottlenecks reside)

sys.setrecursionlimit(10000)

from bnatools.tools import SpectrumHandler
from thesis.preprocessing.transforms import (
    Concat2TensorSoftRealData,
    # Concat2TensorSoft,
)
from thesis.configs.config_reader import load_config
from thesis.utils.wrappers import deprecated

# -- load config
config = load_config("../configs/config.yml")
logger = logging.getLogger(__name__)


def safe_split_indices(
    indices: Sequence[int],
    labels: Sequence[int],
    seed: int = 42,
) -> Tuple[list[int], list[int]]:
    """
    Manual per-class train/val split
        - If a class has 1 example: it goes to both train & val
        - Else: hold out 1 example for val, rest --> train
    """
    label_to_indices = defaultdict(list)
    for idx, lbl in zip(indices, labels):
        label_to_indices[lbl].append(idx)

    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []

    for lbl, idxs in label_to_indices.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)  # shuffle indices for this class
        if len(idxs) == 1:
            # singleton case: include in both sets
            train_idx.append(idxs[0])
            val_idx.append(idxs[0])
        else:
            # hold out one for validation
            val_idx.append(idxs[-1])
            train_idx.extend(idxs[:-1])  # all but last go to train

    return train_idx, val_idx


##################################################
### Dataset classes for spectral data handling ###
##################################################
class SpectraData(Dataset):
    def __init__(
        self,
        h5_path: str,
        data_names: list[str],
        rnd_N: int = None,
        transforms: bool = None,
        verbose: bool = False,
        num_classes: int = 10,
        label_smoothing: float = 0.0,  # for soft labels
    ):
        super().__init__()

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"File {h5_path} not found.")

        self.h5_path = h5_path
        self.transforms = transforms
        self.verbose = verbose
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.data_names = data_names

        traindata_name = "spectra"
        gtdata_name = "labels_idx"

        len_tmp = self.get_data_length(h5_path)
        rnd_N = self._draw_random(rnd_N, len_tmp)

        self.spectrum_handler = SpectrumHandler(
            h5_file=h5_path,
            h5_names=data_names,
            rnd_N=rnd_N,
        )

        self.len = len(self.spectrum_handler)
        self._static_transforms()
        self._consistency_check()

        if self.verbose:
            print(self.spectrum_handler.print_summary())

    def _to_list(self, data_names):
        return [data_names] if isinstance(data_names, str) else data_names

    def _draw_random(self, rnd_N, len_tmp):
        if rnd_N is not None:
            if isinstance(rnd_N, (int, float)):
                return np.random.choice(
                    np.arange(len_tmp), size=min([len_tmp, int(rnd_N)]), replace=False
                )
            elif isinstance(rnd_N, (list, tuple, np.ndarray)):
                return np.array(rnd_N)
            else:
                raise ValueError("rnd_N must be int, float or array-like")

    def filter(
        self, filter_dict: dict = None, in_containers: str = "all"
    ) -> np.ndarray:
        """
        Filters the data based on the provided filter dictionary.

        Parameters
        ----------
        filter_dict : dict, optional
            Dictionary containing the filter criteria. The default is None.
        in_containers : str, optional
            Specifies whether to include data present in all containers ('all') or any container ('any').
            The default is 'all'.

        Returns
        -------
        indices : np.ndarray
            Array of indices that match the filter criteria.
        """
        if filter_dict is None:
            return np.arange(self.len)

        idx = []
        for name in self.data_names:
            try:
                filter_idx = self.spectrum_handler.scs[name].get_filter_idx(filter_dict)
            except Exception as e:
                print(f"Error in filter for {name}: {e}")
                continue
            idx.append(filter_idx)

        if not idx:
            return np.array([])

        try:
            indices = np.array(idx)
        except Exception as e:
            print(f"Error in index conversion: {e}")
            return np.array([])

        if len(indices.shape) > 1:
            indices = np.ravel(indices)

        return indices if len(indices) > 0 else np.array([])

    @classmethod
    def get_data_length(cls, file_name: str, root_path: str = ""):
        return SpectrumHandler.get_data_length(file_name=file_name, root_path=root_path)

    def _static_transforms(self) -> None:
        pass

    def _consistency_check(self) -> None:
        pass

    def __len__(self):
        return self.len

    def get_raw(self, idx):
        """Return the raw **untransformed** data for the given index."""
        raw_batch = [
            {k: v[idx] for k, v in self.spectrum_handler.scs[name].data.items()}
            for name in self.data_names
        ]
        # print(f"Raw batch: {raw_batch}")
        return raw_batch[0]

    def __getitem__(self, idx):
        sample = self.get_raw(idx)
        if self.transforms:
            sample = self.transforms(sample)

        if "soft_labels" not in sample:
            # build one-hot labels
            onehot = torch.zeros(self.num_classes, dtype=torch.float32)
            hard_idx = sample["labels_idx"]
            if isinstance(hard_idx, torch.Tensor):
                hard_idx = hard_idx.item()  # ensuring int
            onehot[hard_idx] = 1.0

            # apply label smoothing
            eps = self.label_smoothing
            if eps > 0:
                onehot = onehot * (1.0 - eps) + eps / self.num_classes
            sample["soft_labels"] = onehot

        if isinstance(sample, list):
            return sample[0]
        return sample


class SpectraDataReal(Dataset):
    """
    Used for training.
    This is the dataset that contains real spectra data and is used when config["DATA"]["DATASET_KWARGS"][1]['label'] == "real".
    Here, the data is prepared to nicely fit the test case (when using trainer.test()).
    """

    def __init__(
        self,
        sample_path: str,
        mappings_json_path: str,
        image_path: str = None,
        x_col: str = "xSample",
        y_col: str = "ySample",
        num_classes: int = None,
        spectra_col: str = "spectrum",
        name_col: str = "name_cleaned",
        transforms=None,
    ):
        super().__init__()

        self.df = pd.read_parquet(sample_path, engine="pyarrow")

        # load numpy image if provided
        self.image: Optional[np.ndarray] = np.load(image_path) if image_path else None

        # extract raw x,y to reconstruct
        self.xs = self.df[x_col].to_numpy(dtype=np.int32)
        self.ys = self.df[y_col].to_numpy(dtype=np.int32)
        self.names = self.df[name_col].to_numpy(dtype=str)
        self.specs = np.stack(self.df[spectra_col].to_numpy()).astype(np.float32)
        self.width = int(self.xs.max() + 1)
        self.height = int(self.ys.max() + 1)

        # -- load mapping file
        # mappings_json_path = os.path.join(sample_path, "mappings", "label_mapping.json")
        with open(mappings_json_path, "r") as f:
            self.mapping = json.load(f)
        self.numeral_mapping = self.mapping["numeral_mapping"]
        # print(f"Loaded numeral mapping: {self.numeral_mapping}")

        self.num_classes = (
            len(self.numeral_mapping) if num_classes is None else num_classes
        )
        self.transforms = transforms
        self.unk_idx = self.numeral_mapping.get("UNK")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        spec = torch.from_numpy(self.specs[idx])  # [4096]
        soft = torch.tensor(self.df["soft_labels"].iloc[idx], dtype=torch.float32)
        hard = int(torch.argmax(soft).item())
        acq_ms = (
            float(self.df["acq_time"].iloc[idx])
            if "acq_time" in self.df.columns
            else 100.0
        )

        # create item
        item = {
            "spectra": spec,
            "labels_idx": torch.tensor(hard),
            "soft_labels": soft,
            "x": int(self.xs[idx]),
            "y": int(self.ys[idx]),
            "mix_weight": self.df["mix_weight"].iloc[idx],
            "acq_time": acq_ms,
        }

        if self.transforms:
            item = self.transforms([item], is_real_data=True)

        return item


class PureDataset(Dataset):
    """
    Given a SpectraData instance and a list of indices, __getitem__ returns
    exactly the pure (unmixed) version of each index, passed through the same
    transforms pipeline.
    """

    def __init__(self, raw_ds, indices, pipeline, num_classes: int):
        super().__init__()
        self.raw_ds = raw_ds
        self.indices = list(indices)
        self.pipeline = pipeline
        self.num_classes = num_classes

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx0 = self.indices[index]
        raw_sample = copy.deepcopy(self.raw_ds.get_raw(idx0))

        assert (
            raw_sample["spectra"] is not None and len(raw_sample["spectra"]) > 0
        ), f"Error: Sample at index {idx0} has no spectra data."

        hard_lbl = raw_sample["labels_idx"]
        if isinstance(hard_lbl, torch.Tensor):
            hard_lbl = hard_lbl.item()

        # build one‐hot soft_labels
        soft_labels = torch.zeros(self.num_classes, dtype=torch.float32)
        soft_labels[hard_lbl] = 1.0
        # raw_sample["soft_labels"] = soft_labels
        spec_tensor = raw_sample["spectra"]
        if not torch.is_tensor(spec_tensor):
            spec_tensor = torch.as_tensor(spec_tensor, dtype=torch.float32)

        standardized_sample = {
            "spectra": raw_sample["spectra"],
            "labels_idx": torch.tensor(hard_lbl, dtype=torch.long),
            "soft_labels": soft_labels,
            "mix_weight": 1.0,
            "mix_orig_a": spec_tensor.clone(),
            "mix_orig_b": spec_tensor.clone(),
        }
        final_sample = self.pipeline(standardized_sample, is_real_data=False)
        return final_sample


class MixOnlyDataset(Dataset):
    def __init__(
        self,
        raw_ds,
        indices: Sequence[int],
        pipeline: callable,
        augment_fraction: float,
        num_classes: int,
        weight_step: float,
        positive_class_indices: Optional[Sequence[int]] = None,
    ):
        """
        Creates a dataset of mixed samples with specific controls.

        Args:
            raw_ds: The base dataset containing raw, unmixed samples.
            indices: A sequence of indices to be used from the raw_ds.
            pipeline: A callable (e.g., a series of transforms) to be applied to each sample.
            augment_fraction: The ratio of mixed samples to generate relative to the
                              number of original samples. E.g., 1.0 means if you have
                              100 original samples, you will generate 100 mixed ones.
            num_classes: The total number of classes in the dataset.
            weight_step: The discrete step for mixing weights, e.g., 0.2 for
                         steps of 20% (0.2, 0.4, 0.6, 0.8).
            positive_class_indices: An optional list of class indices that *must* be
                                    mixed together. All unique pairs from this list
                                    will be generated.
        """
        super().__init__()
        self.raw_ds = raw_ds
        self.indices = list(indices)
        self.pipeline = pipeline
        self.augment_fraction = augment_fraction
        self.num_classes = num_classes
        self.weight_step = weight_step
        self.positive_class_indices = (
            positive_class_indices if positive_class_indices else []
        )
        self.rng = np.random.default_rng()  # default rng

        # --- Parameter Validation ---
        if not (0 < self.weight_step <= 0.5):
            raise ValueError(
                f"weight_step must be between 0 and 0.5, but got {self.weight_step}. "
                "A value > 0.5 would create redundant weights (e.g., 0.8 is the same as 1.0 - 0.2)."
            )

        # --- precompute class-to-index mapping ---
        self.class_to_indices = defaultdict(list)
        for i in self.indices:
            sample = self.raw_ds.get_raw(i)
            lbl = sample["labels_idx"]
            if isinstance(lbl, torch.Tensor):
                lbl = lbl.item()
            self.class_to_indices[lbl].append(i)

        # check if all positive classes are actually present in the dataset subset
        logger.info(f"Positive class indices specified: {self.positive_class_indices}")
        for c in self.positive_class_indices:
            if c not in self.class_to_indices:
                logger.warning(
                    f"Warning: Positive class {c} is not found in the provided indices."
                )

        # --- precompute weights
        # e.g., if step is 0.2, weights will be [0.2, 0.4, 0.6, 0.8]
        self.mixing_weights = np.arange(self.weight_step, 1.0, self.weight_step)

        # --- How many mixed examples we will create each epoch ---
        total = len(self.indices)
        self.num_augmented = int(np.floor(self.augment_fraction * total))

        # Build the initial batch of mixed samples:
        self.mixed_samples = []
        if self.num_augmented > 0:
            self._create_mixed_list()

    def _mix_one_sample(self, class_A_idx: int, class_B_idx: int, w: float):
        """Helper function to create a single mixed sample dictionary."""
        # Pick a random sample from each class
        idxA = self.rng.choice(self.class_to_indices[class_A_idx])
        idxB = self.rng.choice(self.class_to_indices[class_B_idx])

        sampA = self.raw_ds.get_raw(idxA)
        sampB = self.raw_ds.get_raw(idxB)

        assert (
            sampA["spectra"] is not None and len(sampA["spectra"]) > 0
        ), f"Error: Sample at index {idxA} has no spectra data."
        assert (
            sampB["spectra"] is not None and len(sampB["spectra"]) > 0
        ), f"Error: Sample at index {idxB} has no spectra data."

        specA = torch.as_tensor(sampA["spectra"], dtype=torch.float32)
        specB = torch.as_tensor(sampB["spectra"], dtype=torch.float32)

        # linear interpolation
        mixed_spec = w * specA + (1.0 - w) * specB

        # label corresponds to the class with the higher weight
        final_label = class_A_idx if w >= 0.5 else class_B_idx

        # Build soft-labels vector
        soft = torch.zeros(self.num_classes, dtype=torch.float32)
        soft[class_A_idx] = w
        soft[class_B_idx] = 1.0 - w
        return {
            "spectra": mixed_spec,
            "labels_idx": torch.tensor(final_label, dtype=torch.long),
            "soft_labels": soft,
            "mix_weight": w,
            "mix_orig_a": specA.clone(),
            "mix_orig_b": specB.clone(),
        }

    def _create_mixed_list(self):
        """(Re)build self.mixed_samples as a fresh list of length self.num_augmented."""
        out = []

        # --- generate all mandatory mixes from the positive list ---
        if self.positive_class_indices and len(self.positive_class_indices) >= 2:
            # itertools.combinations ensures we don't get both (A, B) and (B, A)
            positive_pairs = itertools.combinations(self.positive_class_indices, 2)

            for class_A, class_B in positive_pairs:
                # ensure we have samples for both classes before trying to mix
                if (
                    class_A in self.class_to_indices
                    and class_B in self.class_to_indices
                ):
                    for w in self.mixing_weights:
                        mixed_sample = self._mix_one_sample(class_A, class_B, w)
                        out.append(mixed_sample)

        # --- fill the rest with random mixes until we reach num_augmented ---
        all_available_classes = list(self.class_to_indices.keys())
        num_remaining = self.num_augmented - len(out)

        if num_remaining > 0 and len(all_available_classes) >= 2:
            for i in range(num_remaining):
                # pick two distinct random classes
                class_A, class_B = self.rng.choice(all_available_classes, 2)

                # pick a random weight from our predefined steps
                w = self.rng.choice(self.mixing_weights)
                mixed_sample = self._mix_one_sample(class_A, class_B, w)
                out.append(mixed_sample)

        # --- Finalize the list ---
        if len(out) > self.num_augmented:
            random.shuffle(out)
            self.mixed_samples = out[: self.num_augmented]
        else:
            self.mixed_samples = out

    def regenerate(self):
        """Call this at the start of each epoch so that a brand‐new random
        set of mixed samples is created."""
        self._create_mixed_list()

    def __len__(self):
        return len(self.mixed_samples)

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.mixed_samples[idx])
        final_sample = self.pipeline(sample, is_real_data=False)
        return final_sample


class RealSpectraDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transforms: List,
        label_mapper: dict,
        image_path: str,
    ):
        super().__init__()
        self.df = df
        self.transforms = transforms
        self.label_mapper = label_mapper
        self.xs = self.df["xSample"].to_numpy(dtype=np.int32)
        self.ys = self.df["ySample"].to_numpy(dtype=np.int32)
        self.image = np.load(image_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spectrum = row["spectrum"]
        label_idx = row["labels_idx"]
        soft_labels = row["soft_labels"]
        acq_ms = float(row["acq_time"])
        total_counts_per_second = row["total_counts_per_second"]

        x = (
            torch.zeros((1, 4096), dtype=torch.float32)
            if spectrum is None or len(spectrum) == 0 or pd.isna(spectrum).all()
            else torch.from_numpy(spectrum).float().unsqueeze(0)
        )
        y = torch.tensor(label_idx, dtype=torch.long)

        # create item
        sample = {
            "spectra": x,
            "labels_idx": y,
            "soft_labels": torch.as_tensor(soft_labels, dtype=torch.float32),
            "x": int(self.xs[idx]),
            "y": int(self.ys[idx]),
            "acq_time": acq_ms,
            "total_counts_per_second": total_counts_per_second,
        }

        if self.transforms:
            sample = self.transforms(sample, is_real_data=True)

        return sample


###############################################
### DataModule classes for loading datasets ###
###############################################


class FineTuneDataModule(LightningDataModule):
    """
    This is the data module for loading real spectra data.
    It is used in fine-tune-model, so that real-world-measurements are used for training.
    """

    def __init__(
        self,
        parquet_path: str,
        label_mapper: dict,
        config: dict,
        rng: np.random.Generator,
        batch_size: int = 128,
        num_workers: int = 0,
        val_size: float = 0.2,
        random_state: int = 42,
        image_path: str = None,
    ):
        super().__init__()
        self.parquet_path = parquet_path
        self.label_mapper = label_mapper
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_path = image_path
        self.val_size = val_size
        self.random_state = random_state
        self.rng = rng
        self.config = config
        self.train_pure = None
        self.val_pure = None
        self.mineral_image = np.load(image_path) if image_path else None

    def setup(self, stage: Optional[str] = None):
        n_classes = self.label_mapper.get_num_classes()
        logger.info("Building transformation pipelines for fine-tuning...")
        self.train_pipeline, _ = build_transforms_pipeline(
            rng=self.rng, stage="finetune_train", num_classes=n_classes, cfg=self.config
        )
        self.val_pipeline, _ = build_transforms_pipeline(
            rng=self.rng, stage="finetune_val", num_classes=n_classes, cfg=self.config
        )
        logger.info("Transformation pipelines built.")

        # load from parquet
        df = pd.read_parquet(self.parquet_path, engine="pyarrow")
        print(f"Loaded fine-tuning data: {len(df)} samples from {self.parquet_path}")
        if df["labels_idx"].isna().any():
            unk_idx = self.label_mapper.get_num_classes() - 1
            df["labels_idx"] = df["labels_idx"].fillna(unk_idx).astype(int)

        train_df, val_df = train_test_split(
            df,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=df["labels_idx"],
        )
        logger.info(f"Data split: {len(train_df)} train, {len(val_df)} val")

        self.train_ds = RealSpectraDataset(
            train_df,
            transforms=self.train_pipeline,
            label_mapper=self.label_mapper,
            # image_path=self.image_path,
        )
        self.val_ds = RealSpectraDataset(
            val_df,
            transforms=self.val_pipeline,
            label_mapper=self.label_mapper,
            # image_path=self.image_path,
        )
        # settings for class weights calc
        self.train_pure = self.train_ds
        self.val_pure = self.val_ds

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class SimpleSpectraLDM(LightningDataModule):
    def __init__(
        self,
        synthetic_config: Dict,
        config: Dict,
        label_mapper: LabelMapper,
        num_classes: int,
        real_config: Optional[Dict] = None,
        batch_size: int = 128,
        num_workers: int = 0,
    ):

        super().__init__()
        self.save_hyperparameters(
            ignore=["synthetic_config", "real_config", "config", "label_mapper"]
        )

        # Store the configurations and dependencies
        self.synthetic_config = synthetic_config
        self.real_config = real_config
        self.config = config
        self.label_mapper = label_mapper

        # Placeholders for datasets that will be created in setup()
        self.dataset_train = self.dataset_val = self.dataset_test = None
        self.train_pure = self.train_mixed = self.val_pure = self.val_mixed = None
        self.train_summary = self.val_summary = self.real_summary = {}

    def get_transform_summaries(self) -> tuple[dict, dict, dict]:
        return self.train_summary, self.val_summary, self.real_summary

    def pipeline_signature(self) -> tuple[str, str]:
        stages = {
            "train_pure": getattr(self.train_pure, "transforms", None)
            and getattr(self.train_pure.transforms, "chosen_names", []),
            "train_mixed": getattr(self.train_mixed, "transforms", None)
            and getattr(self.train_mixed.transforms, "chosen_names", []),
            "val": getattr(self.val_pure, "transforms", None)
            and getattr(self.val_pure.transforms, "chosen_names", []),
            "test": getattr(self.dataset_test, "transforms", None)
            and getattr(self.dataset_test.transforms, "chosen_names", []),
        }
        sig_str = json.dumps(stages, sort_keys=True)
        sig_hash = hashlib.md5(sig_str.encode()).hexdigest()[:6]
        return sig_str, sig_hash

    def setup(self, stage: str = None):
        # leaving hook empty.
        pass

    def mixture_setup(self, stage: str = None):
        logger.info(f"Setting up datasets for stage: {stage}...")
        base_seed = self.config["MODEL"]["LEARNING"]["TRAINER"].get("TRAINING_SEED", 42)
        self.rng = np.random.default_rng(base_seed)
        if stage in (TrainerFn.FITTING, "fit") and self.dataset_train is not None:
            print(
                f"Train dataset already set up with {len(self.dataset_train)} samples from {stage}."
            )
            return  # Setup for training/validation is already done.

        if stage == "test" and self.dataset_test is not None:
            print(f"Test dataset already set up with {len(self.dataset_test)} samples.")
            return  # Setup for testing is already done.
        if self.dataset_train is None and stage in ("fit", "validate", None):
            # --- get params from main cfg ---
            mix_augment_cfg = self.config["DATA"]["MIX_AUGMENT"]
            split_ratios = self.config["DATA"]["SPLITS"]["synthetic"]
            positive_list = self.config["DATA"].get("POSITIVE_LIST", [])

            # --- build transforms pipelines ---
            self.train_pipeline, self.train_summary = build_transforms_pipeline(
                rng=self.rng,
                stage="train",
                num_classes=self.hparams.num_classes,
                cfg=self.config,
            )
            self.val_pipeline, self.val_summary = build_transforms_pipeline(
                rng=self.rng,
                stage="val",
                num_classes=self.hparams.num_classes,
                cfg=self.config,
            )

            # --- create and split synth dataset ---
            ds_class = self.synthetic_config["ds_class"]
            ds_kwargs = self.synthetic_config["ds_kwargs"]
            ds_kwargs.pop("num_classes", None)  # avoid conflict
            raw_ds = ds_class(num_classes=self.hparams.num_classes, **ds_kwargs)

            all_labels = [raw_ds.get_raw(i)["labels_idx"] for i in range(len(raw_ds))]
            train_indices, val_indices = safe_split_indices(
                indices=list(range(len(raw_ds))),
                labels=all_labels,
                seed=base_seed,
            )

            # check empty spectra
            empty_spectra = [
                raw_ds.get_raw(i)["spectra"] is None
                or len(raw_ds.get_raw(i)["spectra"]) == 0
                for i in range(len(raw_ds))
            ]

            # --- assemble final training and validation datasets ---
            self.train_pure = PureDataset(
                raw_ds,
                train_indices,
                self.train_pipeline,
                num_classes=self.hparams.num_classes,
            )
            self.train_mixed = MixOnlyDataset(
                raw_ds=raw_ds,
                indices=train_indices,
                pipeline=self.train_pipeline,
                augment_fraction=mix_augment_cfg["AUGMENT_FRACTION"],
                num_classes=self.hparams.num_classes,
                weight_step=mix_augment_cfg["WEIGHT_STEP"],
                positive_class_indices=positive_list,
            )
            self.dataset_train = ConcatDataset([self.train_pure, self.train_mixed])

            # validation datasets
            self.val_pure = PureDataset(
                raw_ds,
                val_indices,
                self.val_pipeline,
                num_classes=self.hparams.num_classes,
            )
            self.val_mixed = MixOnlyDataset(
                raw_ds=raw_ds,
                indices=val_indices,
                pipeline=self.val_pipeline,
                num_classes=self.hparams.num_classes,
                augment_fraction=mix_augment_cfg["AUGMENT_FRACTION"],
                weight_step=mix_augment_cfg["WEIGHT_STEP"],
                positive_class_indices=positive_list,
            )
            self.dataset_val = ConcatDataset([self.val_pure, self.val_mixed])

            self.raw_synthetic = raw_ds
            self.xform_synthetic = self.train_pure
            self.mixed_synthetic = self.train_mixed

            logger.info("===TRAIN===")
            logger.info(
                f"Pure: {len(self.train_pure)}  Mixed: {len(self.train_mixed)}  Total: {len(self.dataset_train)}"
            )
            logger.info("===VAL===")
            logger.info(
                f"Pure: {len(self.val_pure)}  Mixed: {len(self.val_mixed)}  Total: {len(self.val_pure) + len(self.val_mixed)}"
            )
            if self.dataset_test:
                logger.info(f"===TEST=== \nSize: {len(self.dataset_test)}")

            logger.info("Data modules setup complete.")

        if stage == "test" and self.dataset_test is None:
            logger.info("Setting up test dataset...")
            # --- setup test dataset if cfg provided ---
            if self.real_config:
                test_ds_class = self.real_config["ds_class"]
                test_ds_kwargs = self.real_config["ds_kwargs"]
                self.real_pipeline, _ = build_transforms_pipeline(
                    rng=self.rng,
                    stage="test",
                    num_classes=self.hparams.num_classes,
                    cfg=self.config,
                )
                test_ds_kwargs["transforms"] = self.real_pipeline
                test_ds_kwargs.pop("num_classes", None)  # avoid conflict
                print(f"Test dataset kwargs: {test_ds_kwargs}")
                self.dataset_test = test_ds_class(
                    mappings_json_path=self.label_mapper.filepath,
                    num_classes=self.hparams.num_classes,
                    **test_ds_kwargs,
                )
                self.mineral_image = getattr(self.dataset_test, "image", None)
                self.mineral_shape = (
                    self.mineral_image.shape[:2]
                    if self.mineral_image is not None
                    else None
                )

    # --- worker init function hooked into DataLoader
    def _worker_init_fn(self, worker_id: int):
        # epoch-safe reseed. Lightning should expose current_epoch via trainer.
        epoch = getattr(self, "_current_epoch", 0)
        self.reseed_pipelines(epoch=epoch, worker_id=worker_id)
        np.random.seed(self._base_seed + 1000 * epoch + worker_id)

        # make pytorch deterministic per worker
        torch.manual_seed(self._base_seed + 1000 * epoch + worker_id)

    def get_train_len(self):
        return self.train_len

    def get_val_len(self):
        return self.val_len

    def get_test_len(self):
        return self.test_len

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=(self.hparams.num_workers > 0),
            shuffle=True,
            pin_memory=True,
            worker_init_fn=self._worker_init_fn,
        )

    def val_dataloader(self):
        loaders = []
        # always adding pure val loader
        pure_loader = DataLoader(
            self.val_pure,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            persistent_workers=(self.hparams.num_workers > 0),
            pin_memory=True,
            worker_init_fn=self._worker_init_fn,
        )
        loaders.append(pure_loader)

        # add mix only if non-empty
        if len(self.val_mixed) > 0:
            mix_loader = DataLoader(
                self.val_mixed,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                persistent_workers=(self.hparams.num_workers > 0),
                shuffle=False,
                pin_memory=True,
                worker_init_fn=self._worker_init_fn,
            )
            loaders.append(mix_loader)
        return loaders

    def test_dataloader(self):
        if not self.dataset_test:
            return None
        return DataLoader(
            self.dataset_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=True,
            worker_init_fn=self._worker_init_fn,
        )
