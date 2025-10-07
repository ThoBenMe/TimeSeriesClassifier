import pywt
import math
import torch
import numpy as np
import torch.nn.functional as F
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d

from sklearn.decomposition import PCA
from thesis.utils.wrappers import deprecated
from matplotlib import pyplot as plt
from thesis.utils.plotting import plot_transforms_spectra

"""
Preferred order of transforms:
1. SamplePreparation          # Setup metadata
2. AddPileup                  # Multi-photon spectrum distortion
3. AddKCPS                    # Scale normalized spec to real photon counts
4. AddNoise                   # Poisson photon counting (detection)
5. Shift                      # Simulate detector drift
6. Gain                       # Global gain/beam instability
7. WaveletTransform           # Optional: noise suppression AFTER noise
8. Smoothing                  # Simulate resolution broadening
9. BaselineCorrection         # Remove slow-varying background
10. NormIT                    # Normalize by t_life and tube_current
11. LogTransform / NormSqrt   # Compress dynamic range
12. FinalConversion           # Convert labels
"""


class Concat2TensorSoftRealData(object):
    """used for training on multiple phases to concatenate spectra"""

    def __init__(
        self,
        data_name,
        ground_truth_name,
        concat_gt=False,
        use_soft_labels=False,
        num_classes=10,
    ):
        self.data_name = data_name
        self.ground_truth_name = ground_truth_name
        self.concat_gt = concat_gt
        self.use_soft_labels = use_soft_labels
        self.num_classes = num_classes

    def __call__(self, batch):
        # Extract coordinates before processing other fields
        print(f"Batch Type: {type(batch)}")
        print(f"Batch Size: {len(batch)}")
        print(f"BATCH: {batch}")

        coords = torch.stack([item["coords"] for item in batch])

        # Process data
        data = torch.stack([item[self.data_name] for item in batch], dim=0)

        images = torch.stack([item["image_val"] for item in batch], dim=0)  # image data

        if self.ground_truth_name in batch[0]:
            labels = []
            for item in batch:
                lbl = item[self.ground_truth_name]
                if not isinstance(lbl, torch.Tensor):
                    lbl = torch.tensor(lbl, dtype=torch.float32)
                labels.append(lbl)
            labels = torch.stack(labels, dim=0)
            if self.use_soft_labels:
                labels = F.one_hot(labels.long(), num_classes=self.num_classes).float()
        else:
            # labels = None
            # use uniform probability distribution as dummy labels
            batch_size = data.shape[0]
            labels = (
                torch.ones((batch_size, self.num_classes), dtype=torch.float32)
                / self.num_classes
            )
            # (should not interfere with the output of the model)

        return {"data": data, "label": labels, "coords": coords, "image_val": images}
        # "meta": meta_list}

    def __repr__(self):
        return self.__class__.__name__ + "()"


##################################################################
###  TRANSFORMS FOR DATA AUGMENTATION AND PREPROCESSING        ###
##################################################################
class OrderedTransformPipeline(object):
    """
    A configurable and ordered transformation pipeline that applies
    preprocessing, optional, and postprocessing transforms to input batches.

    This pipeline supports dynamic or fixed selection of optional transforms,
    allowing randomized or reproducible augmentations. Transforms are applied
    in a strict order: required_pre -> optional_subset -> required_post.

    Parameters
    ----------
    required_pre : list
        List of transformations to always apply before any optional transforms.

    optional_transforms : list
        List of candidate transformations from which a random subset is selected.

    required_post : list
        List of transformations to always apply after the optional transforms.

    min_optionals : int
        Minimum number of optional transforms to select.

    max_optionals : int
        Maximum number of optional transforms to select.

    dynamic : bool, default=True
        If True, a new subset of optional transforms is sampled on each call.
        If False, a fixed subset is sampled once and reused.

    rng : np.random.Generator, optional
        Optional NumPy random number generator for deterministic behavior.
        If None, a default generator is created.

    Methods
    -------
    __call__(batch, is_real_data=False)
        Applies the full transform pipeline to the input batch.

    chosen_names
        Returns the short names (or class names) of the currently chosen optional transforms.

    Examples
    --------
    >>> pipeline = OrderedTransformPipeline(
    ...     required_pre=[Normalize()],
    ...     optional_transforms=[AddNoise(), Flip()],
    ...     required_post=[Clip()],
    ...     min_optionals=1,
    ...     max_optionals=2,
    ...     dynamic=True
    ... )
    >>> batch = pipeline(batch)
    """

    def __init__(
        self,
        required_pre: list,
        optional_transforms: list,
        required_post: list,
        min_optionals: int,
        max_optionals: int,
        dynamic: bool = True,
        rng: np.random.Generator = None,
    ):
        self.required_pre = required_pre
        self.optional_transforms = optional_transforms
        self.required_post = required_post
        self.min_optionals = min_optionals
        self.max_optionals = max_optionals
        self.dynamic = dynamic
        self._fixed_optional_subset = None
        self.rng = rng if rng is not None else np.random.default_rng()

    def _choose_optional_subset(self):
        num = self.rng.integers(self.min_optionals, self.max_optionals + 1)
        if num == 0 or not self.optional_transforms:
            return []
        if num > len(self.optional_transforms):
            num = len(self.optional_transforms)
        return list(self.rng.choice(self.optional_transforms, size=num, replace=False))

    def __call__(self, batch, is_real_data: bool = False):
        optional_subset = (
            self._choose_optional_subset()
            if self.dynamic
            else self._fixed_optional_subset
        )
        if optional_subset is None:
            self._fixed_optional_subset = optional_subset = (
                self._choose_optional_subset()
            )

        full_pipeline = self.required_pre + optional_subset + self.required_post

        for t in full_pipeline:
            batch = t(batch, is_real_data=is_real_data)
        return batch

    def reseed(self, seed: int):
        self.rng = np.random.default_rng(seed)
        for t in self.required_pre + self.optional_transforms + self.required_post:
            if hasattr(t, "rng"):
                t.rng = np.random.default_rng(seed)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"required_pre={self.required_pre}, "
            f"optional_transforms={self.optional_transforms}, "
            f"required_post={self.required_post}, "
            f"min_optionals={self.min_optionals}, max_optionals={self.max_optionals})"
        )

    @property
    def chosen_names(self) -> list[str]:
        """Return names of a current random subet (not cached)."""
        if self._fixed_optional_subset is None:
            self._fixed_optional_subset = self._choose_optional_subset()

        return [
            t.short_name() if hasattr(t, "short_name") else t.__class__.__name__
            for t in self._fixed_optional_subset
        ]


@deprecated
class PCATransform(object):
    def __init__(self, pca_model: PCA = None, n_components=2):
        self.n_components = n_components
        if pca_model is not None:
            self.pca_model = pca_model
            self.is_fitted = True
        else:
            self.pca_model = PCA(n_components=n_components, whiten=True)
            self.is_fitted = False

    def fit(self, data):
        self.pca_model.fit(data)
        self.is_fitted = True

    def __call__(self, batch, is_real_data: bool = False):
        if is_real_data:
            # Skip PCA for real data
            # ! NEEDS TO BE DONE SPECIFICALLY ON REAL DATA!
            return batch

        if not self.is_fitted:
            raise RuntimeError(
                "PCA model is not fitted. Pre-fit the PCA on larger dataset."
            )

        for i in range(len(batch)):
            spec = batch[i]["spectra"]
            if isinstance(spec, torch.Tensor):
                spec_np = spec.cpu().numpy()
            else:
                spec_np = np.array(spec)

            # Apply PCA transformation (expecting a 2D array: reshape to [1, 4096])
            transformed_spec = self.pca_model.transform(spec_np.reshape(1, -1))
            batch[i]["spectra"] = torch.tensor(
                transformed_spec.flatten(), dtype=torch.float32
            )

        return batch

    def plot_batch(self, batch, sample_idx=0):
        """
        Plot the PCA transformed spectra for a specific sample in the batch.
        """
        import matplotlib.pyplot as plt

        # set orig spectrum based on mix_weight
        w = batch[sample_idx]["mix_weight"].item()
        orig_spec = (
            batch[sample_idx]["mix_orig_a"].cpu().numpy()
            if w >= 0.5
            else batch[sample_idx]["mix_orig_b"].cpu().numpy()
        )

        # calc PCA coefficients using orig spec
        pca_coeffs = self.pca_model.transform(orig_spec.reshape(1, -1))
        # reconstruct spec using inverse transform
        reconstructed = self.pca_model.inverse_transform(pca_coeffs)

        # plotting
        fix, axs = plt.subplots(3, 1, figsize=(10, 12))

        # orig spec
        axs[0].plot(orig_spec, label="Original Spectrum")
        axs[0].set_title("Original Spectrum")
        axs[0].legend()

        # PCA coefficients
        axs[1].plot(range(len(pca_coeffs.flatten())), pca_coeffs.flatten())
        axs[1].set_title("PCA Coefficients")
        axs[1].set_xlabel("Component Index")
        axs[1].set_ylabel("Coefficient Value")
        axs[1].legend()

        # reconstructed spectrum
        axs[2].plot(
            reconstructed.flatten(), label="Reconstructed Spectrum", color="orange"
        )
        axs[2].set_title("Reconstructed Spectrum")
        axs[2].legend()

        plt.tight_layout()
        plt.show()

    def __repr__(self):
        return f"{self.__class__.__name__}(n_components={self.n_components})"


class WaveletTransform(object):
    """
    Applies a wavelet transform to the spectrum.
    This transform performs a discrete wavelet transform (DWT) using the specified wavelet
    and reconstruction method (if desired) to reduce noise while preserving locality.
    """

    def __init__(
        self,
        wavelet: str = "db4",
        level: int = 4,
        apply_threshold: bool = False,
        threshold: float = None,
        rng: np.random.Generator = None,
    ):
        self.wavelet = wavelet
        self.level = level
        self.apply_threshold = apply_threshold
        self.threshold = threshold
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, batch, is_real_data: bool = False):
        if is_real_data:
            # Skip wavelet transform for real data
            # ! NEEDS TO BE DONE SPECIFICALLY ON REAL DATA - APPLY THE SAME WAY AS FOR TRAINING!
            return batch

        assert isinstance(batch, dict), "Batch should be a dictionary."
        if batch["_has_wavelet"] == True:
            print(f"Sample has already been wavelet transformed. Returning it as is.")
            return batch

        epsilon = 1e-8  # small constant to avoid division by zero
        spec = batch["spectra"]
        if isinstance(spec, torch.Tensor):
            spec = spec.cpu().numpy()
        else:
            spec = np.array(spec)

        # assuming the spectra is 1D of length 4096
        coeffs = pywt.wavedec(spec.flatten(), self.wavelet, level=self.level)

        if self.apply_threshold:
            # Calculate the median absolute deviation (MAD)
            mad = np.median(np.abs(coeffs[-1]))

            # Ensure MAD is not zero to avoid division by zero
            if mad == 0:
                thresh = 0.0  # If MAD is zero, set threshold to zero
            else:
                thresh = mad / (0.6745 + epsilon)

            if self.threshold is not None:
                thresh = self.threshold  # Use the provided threshold if available

            # Apply thresholding
            coeffs[1:] = [pywt.threshold(c, thresh, mode="soft") for c in coeffs[1:]]

        # Reconstruct signal using approx. coeffs (denoised)
        reconstructed = pywt.waverec(coeffs, self.wavelet)
        # Ensure output size is 4096
        if reconstructed.shape[0] > 4096:
            reconstructed = reconstructed[:4096]
        elif reconstructed.shape[0] < 4096:
            reconstructed = np.pad(
                reconstructed, (0, 4096 - reconstructed.shape[0]), mode="constant"
            )

        batch["spectra"] = torch.tensor(reconstructed, dtype=torch.float32)
        batch["_has_wavelet"] = True
        return batch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(wavelet={self.wavelet}, level={self.level}, "
            f"apply_threshold={self.apply_threshold}, threshold={self.threshold})"
        )


class BaselineCorrection(object):
    """
    Subtracts an estimated baseline from the spectrum using a median filter.
    Kernel_Size controls the window over which the baselinen is estimated.

    Where: Before any final value transforms (e.g., NormSqrt, LogTransform).
    Why? -> log and sqrt are nonlinear and will distort baseline if applied before.
    """

    def __init__(self, kernel_size=51):
        self.kernel_size = kernel_size

    def __call__(self, batch, is_real_data: bool = False):
        assert isinstance(batch, dict), "Batch should be a dictionary."

        if batch["_has_baseline_correction"] == True:
            print(f"Sample has already been baseline corrected. Returning it as is.")
            return batch

        spec = batch["spectra"]
        if isinstance(spec, torch.Tensor):
            spec_np = spec.cpu().numpy()
        else:
            spec_np = np.array(spec)

        # if shape (1, 4096) -> (4096,), and if (4096,) -> (4096,)
        spec_np = spec_np.squeeze()

        if len(spec_np) < self.kernel_size:
            print("!!! PROBLEM DETECTED !!!")
            print(
                f"Spectrum length is {len(spec_np)}, less than kernel size: {self.kernel_size}"
            )
            print(f"SPEC TYPE: {type(spec_np)}")
            print(f"SPEC: {spec_np}")
            raise ValueError("Found a spectrum shorter than the baseline kernel.")

        baseline = medfilt(spec_np, kernel_size=self.kernel_size)
        # Ensure non-negative values
        spec_corrected = np.clip(
            spec_np - baseline, a_min=0, a_max=None
        )  # Ensure non-negative values
        batch["spectra"] = torch.tensor(spec_corrected, dtype=torch.float32)

        batch["_has_baseline_correction"] = True
        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size})"


class Smoothing(object):
    """
    Applies Gaussian smoothing to the spectrum.
    G is gaussian kernel with standard deviation sigma.
    Good for:
    - smoothing out noisy sharp features
    - aproximating energy resolution of real spectrometers
    - preventing overfitting to narrow peaks

    Where:
    - Option 1: After AddNoise --> Simulate Detector Resolution
    --- Good when trying to simulate real detector resolution broadening
    - Option 2: Before AddNoise --> Denoising very sharp, synthetic peaks
    --- Good when trying to pre-smooth synthetic spectra to look more like real measurements
    - Option 3: Both
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, batch, is_real_data: bool = False):
        # # Skip smoothing for real data
        # if is_real_data:
        #     return batch

        assert isinstance(batch, dict), "Batch should be a dictionary."
        if batch["_has_smoothing"] == True:
            print(f"Sample has already been smoothed. Returning it as is.")
            return batch

        spec = batch["spectra"]
        if isinstance(spec, torch.Tensor):
            spec_np = spec.cpu().numpy()
        else:
            spec_np = np.array(spec)

        spec_smoothed = gaussian_filter1d(spec_np, sigma=self.sigma)
        spec_smoothed[spec_smoothed < 0] = 0.0  # Ensure non-negative values
        batch["spectra"] = torch.tensor(spec_smoothed, dtype=torch.float32)

        # plot for debugging, side by side with original
        # plot_transforms_spectra(
        #     spec1=spec,
        #     spec2=spec_smoothed,
        #     log=True,
        #     kev_style=True,
        #     label=batch["labels"],
        #     transform="Smoothing",
        # )

        batch["_has_smoothing"] = True
        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}(sigma={self.sigma})"


class Shift(object):
    """
    Shifts the spectrum by a random int number of indices within range [-max_shift, max_shift]
    Shifts should be kept VERY small to simulate calibration uncertainty.
    """

    def __init__(self, max_shift=3, rng: np.random.Generator = None):
        self.max_shift = max_shift
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, batch, is_real_data: bool = False):
        # Skip shifting for real data
        if is_real_data:
            return batch

        assert isinstance(batch, dict), "Batch should be a dictionary."
        if batch["_has_shift"] == True:
            print(f"Sample has already been shifted. Returning it as is.")
            return batch

        spec = batch["spectra"]

        # ensure numpy
        if isinstance(spec, torch.Tensor):
            spec_np = spec.cpu().numpy()
        else:
            spec_np = np.array(spec)

        shift = self.rng.integers(-self.max_shift, self.max_shift + 1)
        if shift > 0:
            spec_shifted = np.zeros_like(spec_np)
            spec_shifted[shift:] = spec_np[:-shift]
        elif shift < 0:
            spec_shifted = np.zeros_like(spec_np)
            spec_shifted[:shift] = spec_np[-shift:]
        else:
            spec_shifted = spec_np.copy()

        # converting back to torch tensor
        batch["spectra"] = torch.tensor(spec_shifted, dtype=torch.float32)

        # plot_transforms_spectra(
        #     spec1=spec,
        #     spec2=spec_shifted,
        #     log=True,
        #     kev_style=True,
        #     label=batch["labels"],
        #     transform="Shift",
        # )

        batch["_has_shift"] = True
        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}(max_shift={self.max_shift})"


class Gain(object):
    """
    Randomly scale intensity of the spectrum to simulate variations in experimental conditions.
    Gain_Range is specified as tuple (min, max)
    """

    def __init__(self, gain_range: tuple = (0.8, 1.2), rng: np.random.Generator = None):
        self.gain_range = gain_range
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, batch, is_real_data: bool = False):
        # Skip gain for real data
        if is_real_data:
            return batch

        if batch["_has_gain"] == True:
            print(f"Sample has already been gain scaled. Returning it as is.")
            return batch

        assert isinstance(batch, dict), "Batch should be a dictionary."
        gain = self.rng.uniform(*self.gain_range)
        spec = batch["spectra"]
        batch["spectra"] = spec * gain

        # plot_transforms_spectra(
        #     spec1=spec,
        #     spec2=batch["spectra"],
        #     log=True,
        #     kev_style=True,
        #     label=batch["labels"],
        #     transform="Gain",
        # )

        batch["_has_gain"] = True
        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}(gain_range={self.gain_range})"


class LogarithmicTransform(object):
    """
    Applies logarithmic transform to comporess the dynamic range of the spectrum.
    Uses log1p to avoid issues with zero values.
    Where: After AddNoise
    """

    def __call__(self, batch, is_real_data: bool = False):
        assert isinstance(batch, dict), "Batch should be a dictionary."

        if batch["_has_log_transform"] == True:
            print(f"Sample has already been log transformed. Returning it as is.")
            return batch

        spec = batch["spectra"]
        spec_orig = spec
        if isinstance(spec, torch.Tensor):
            spec = torch.log1p(spec)
        else:
            spec = np.log1p(spec)
        batch["spectra"] = spec
        # if is_real_data:
        #     plot_transforms_spectra(
        #         spec1=spec_orig,
        #         spec2=spec,
        #         log=True,
        #         kev_style=True,
        #         label=batch["labels_idx"],
        #         transform="LogarithmicTransform",
        #     )

        # batch["_has_log_transform"] = True
        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class SamplePreparation(object):
    """Intention of this class is to prepare each sample by setting default values for pileup parameters
    and saving a copy of the original spectrum for later reference (e.g., for PCA visualization).
    This transform is intended to be called early in the pipeline, before any other transformations are applied.
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        meta_real = self.cfg.get("REAL_METADATA", {})
        meta_synth = self.cfg.get("SYNTH_METADATA", {})
        self.acq_unit = str(meta_real.get("ACQ_TIME_UNIT", "s")).lower()
        self.real_tube_default = float(meta_real.get("TUBE_CURRENT", 0.2))
        self.synth_tube_default = float(meta_synth.get("TUBE_CURRENT", 0.2))
        self.synth_t_life_def = float(meta_synth.get("T_LIFE", 1.1))  # seconds

    def _to_seconds(self, t):
        if t is None:
            return None
        t = float(t)
        return (
            t
            if self.acq_unit == "s"
            else (t * 1e-3 if self.acq_unit == "ms" else t * 1e-6)
        )

    def __call__(self, batch, is_real_data: bool = False):
        if isinstance(batch, list):
            batch = batch[0]
        assert isinstance(batch, dict), "Batch should be a dictionary."

        if "spectra" not in batch:
            raise KeyError("Input batch must contain 'spectra' key.")

        for k in (
            "_has_KCPS",
            "_has_noise",
            "_has_pileup",
            "_has_shift",
            "_has_gain",
            "_has_smoothing",
            "_has_baseline_correction",
            "_has_log_transform",
            "_has_norm_sqrt",
            "_has_normIT",
            "_has_wavelet",
            "_has_crop",
            "_has_normSqrt",
            "_has_anscombe",
        ):
            batch.setdefault(k, False)

        # --- standardize optional metadata ---
        batch.setdefault("pu_1ph", 0.0)
        batch.setdefault("pu_2ph", 0.0)
        batch.setdefault("pu_3ph", 0.0)
        batch.setdefault("pileup_lam", 0.0)
        batch.setdefault("mix_weight", 1.0)

        if is_real_data:
            # use per-pixel acq_time
            acq_time = batch.get("acq_time", None)
            lifetime = self._to_seconds(acq_time)
            if lifetime is not None:
                batch["t_life"] = float(lifetime)  # seconds
            batch.setdefault("tube_current", self.real_tube_default)
            batch.setdefault(
                "total_counts_per_second", batch.get("total_counts_per_second", None)
            )
            # print(
            #     f"REAL DATA: t_life={batch['t_life']}, tube_current={batch['tube_current']}, total_counts_per_second={batch.get('total_counts_per_second')}"
            # )
        else:
            # batch.setdefault("t_life", self.synth_t_life_def)  # e.g., 1.0s
            batch["t_life"] = float(
                self.synth_t_life_def
            )  # hard override of t_life for downstream calcs.
            batch.setdefault("tube_current", self.synth_tube_default)  # e.g., 0.2 A
            batch.setdefault("total_counts_per_second", None)
            # print(
            #     f"SYNTH DATA: t_life={batch['t_life']}, tube_current={batch['tube_current']}, total_counts_per_second={batch['total_counts_per_second']}"
            # )
        print(
            f"SamplePreparation: t_life={batch['t_life']}, tube_current={batch['tube_current']}, total_counts_per_second={batch['total_counts_per_second']}, is_real_data={is_real_data}"
        )
        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class FinalConversion(object):
    def __init__(self, use_soft_labels=False, num_classes=10):
        self.use_soft_labels = use_soft_labels
        self.num_classes = num_classes

    def __call__(self, sample, is_real_data: bool = False):
        # -- convert the things we care about --
        sample["spectra"] = torch.as_tensor(sample["spectra"], dtype=torch.float32)
        sample["labels_idx"] = torch.as_tensor(sample["labels_idx"], dtype=torch.long)

        if self.use_soft_labels:
            if "soft_labels" not in sample:
                n = int(self.num_classes)
                one = torch.zeros(n, dtype=torch.float32)
                li = int(sample["labels_idx"])
                if 0 <= li < n:
                    one[li] = 1.0
                sample["soft_labels"] = one
        else:
            # ensure downstream code can still read the key if it expects it
            if "soft_labels" not in sample:
                n = int(self.num_classes)
                one = torch.zeros(n, dtype=torch.float32)
                li = int(sample["labels_idx"])
                if 0 <= li < n:
                    one[li] = 1.0
                sample["soft_labels"] = one

        if "orig_a" in sample and "mix_orig_a" not in sample:
            sample["mix_orig_a"] = sample.pop("orig_a")
        if "orig_b" in sample and "mix_orig_b" not in sample:
            sample["mix_orig_b"] = sample.pop("orig_b")
        if "mix_orig_a" not in sample:
            # use the transformed spectra as both branches
            sample["mix_orig_a"] = sample["spectra"].clone()
            sample["mix_orig_b"] = sample["spectra"].clone()
        if "mix_weight" not in sample:
            # pure = weight 1.0
            sample["mix_weight"] = 1.0

        # cast remaining tensors
        sample["soft_labels"] = torch.as_tensor(
            sample["soft_labels"], dtype=torch.float32
        )
        sample["mix_orig_a"] = torch.as_tensor(
            sample["mix_orig_a"], dtype=torch.float32
        )
        sample["mix_orig_b"] = torch.as_tensor(
            sample["mix_orig_b"], dtype=torch.float32
        )
        sample["mix_weight"] = torch.as_tensor(
            sample["mix_weight"], dtype=torch.float32
        )
        if "x" in sample:
            sample["x"] = torch.as_tensor(sample["x"], dtype=torch.float32)
        if "y" in sample:
            sample["y"] = torch.as_tensor(sample["y"], dtype=torch.float32)
        return {k: v for k, v in sample.items() if torch.is_tensor(v)}

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(use_soft_labels={self.use_soft_labels}, num_classes={self.num_classes})"
        )


class AddKCPS(object):
    """
    Scales the spectrum by a randomly chosen kcps factor, which is then multiplied by a scale factor.
    This is used to simulate the effect of different count rates on the spectrum.
    This should be applied after AddPileup.
    """

    def __init__(self, cfg, rng: np.random.Generator = None):
        # cfg is always from SYNTH_METADATA: AddKCPS is not added for real data.
        self.rng = rng if rng is not None else np.random.default_rng()

        self.scale_factor = cfg["SCALE_FACTOR"]
        self.rnd_kcps = cfg["RANGE"]

    def __call__(self, sample, is_real_data: bool = False):
        if is_real_data:
            return sample

        # re-set values in sample
        if isinstance(sample, list):
            sample = sample[0]
        assert isinstance(sample, dict), "Batch should be a dictionary."

        if sample.get("_has_KCPS", False):
            print(f"Sample has already been KCPS scaled. Returning it as is.")
            return sample

        t_life = sample.get("t_life", None)
        if t_life is None:
            raise KeyError("Sample must contain 't_life' key for KCPS scaling.")

        # kcps scaling
        kcps = self.rng.uniform(*self.rnd_kcps)
        total_counts = kcps * float(t_life)
        # total_counts = kcps * self.scale_factor * float(t_life)
        sample["spectra"] = sample["spectra"] * total_counts  # scale spectra
        sample["count_rate_kcps"] = kcps
        sample["_has_KCPS"] = True
        print(
            f"Applied KCPS scaling: {kcps} kcps, total_counts={total_counts}. T_life={t_life}"
        )
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f"({self.rnd_kcps}, {self.scale_factor})"


class AddPileup(object):
    """
    Applies a multiphoton pileup effect using 1-ph, 2-ph, and 3-photon spectra.
    Pileup is applied with a given probability (`prob`).
    """

    def __init__(
        self, rnd_lam, rnd_var=None, prob=1.0, rng: np.random.Generator = None
    ):
        self.rnd_lam = rnd_lam
        self.rnd_var = rnd_var or [0.8, 1.2]
        self.prob = prob
        self.rng = rng if rng is not None else np.random.default_rng()

    def make_pileup(self, spec1ph, spec2ph, spec3ph):
        lam = self.rng.uniform(*self.rnd_lam)
        var = self.rng.uniform(*self.rnd_var)
        if lam == 0:
            facs = [1.0, 0.0, 0.0]
        else:
            # This math is fine
            facs = [math.exp(-lam) * (lam**n) / math.factorial(n) for n in range(1, 4)]
        spec = facs[0] * spec1ph + facs[1] * spec2ph + facs[2] * spec3ph * var
        return spec, lam

    def __call__(self, batch, is_real_data: bool = False):
        # Skip pileup for real data
        if is_real_data or self.rng.uniform() > self.prob:
            return batch

        if batch["_has_pileup"] == True:
            print(f"Sample has already been pileup scaled. Returning it as is.")
            return batch

        # Get 1-photon spectrum.
        spec1ph = batch["spectra"]
        if isinstance(spec1ph, torch.Tensor):
            spec1ph = spec1ph.cpu().numpy()

        # Retrieve pu_2ph and pu_3ph if they exist; else default to zeros.
        spec2ph_raw = batch.pop("pu_2ph", None)
        spec3ph_raw = batch.pop("pu_3ph", None)

        if (
            spec2ph_raw is not None
            and hasattr(spec2ph_raw, "__len__")
            and len(spec2ph_raw) == len(spec1ph)
        ):
            spec2ph = np.asarray(spec2ph_raw, dtype=np.float64)
        else:
            spec2ph = np.zeros_like(spec1ph, dtype=np.float64)  # Default to zeros

        # Validate spec3ph
        if (
            spec3ph_raw is not None
            and hasattr(spec3ph_raw, "__len__")
            and len(spec3ph_raw) == len(spec1ph)
        ):
            spec3ph = np.asarray(spec3ph_raw, dtype=np.float64)
        else:
            spec3ph = np.zeros_like(spec1ph, dtype=np.float64)  # Default to zeros
        # ---------------------------------------------------------------

        spec, lam = self.make_pileup(spec1ph, spec2ph, spec3ph)

        batch["spectra"] = torch.as_tensor(spec, dtype=torch.float32)
        batch["pileup_lam"] = lam

        # Reinsert default keys so every sample has them.
        batch["pu_2ph"] = np.zeros_like(batch["spectra"])
        batch["pu_3ph"] = np.zeros_like(batch["spectra"])

        batch["_has_pileup"] = True
        return batch

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Crop(object):
    def __init__(self, crop_lr):
        self.crop_lr = crop_lr

    def __call__(self, batch, is_real_data: bool = False):
        # Todo: Check if Crop is needed on real data. If not, deactivate for real data.
        assert isinstance(batch, dict), "Batch should be a dictionary."
        if self.crop_lr is None:
            return batch
        else:
            if batch["_has_crop"] == True:
                print(f"Sample has already been cropped. Returning it as is.")
                return batch
            spec = batch["spectra"]
            spec_orig = spec
            spec[: self.crop_lr[0]] = 0
            spec[len(spec) - self.crop_lr[1] :] = 0
            batch["spectra"] = spec
            batch["_has_crop"] = True

            # plot_transforms_spectra(
            #     spec1=spec_orig,
            #     spec2=spec,
            #     log=True,
            #     kev_style=True,
            #     label=batch["labels"],
            #     transform="Crop",
            # )
            return batch

    def __repr__(self):
        return self.__class__.__name__ + "()"


class AddNoise(object):
    """
    Poisson noise with realistic dwell-time variation.

    Dual use:
      1) Single-range: pass 'rnd_lt=["0.098s","0.103s"]' (or numbers; seconds by default).
      2) Bimodal: pass 'modes={"inner":["0.098s","0.103s"], "edge":["45ms","70ms"]}'
         and 'p_edge=0.05' to occasionally sample short edge lifetimes.

    Why dual-mode:
      In AMICS maps the left edge often has shorter acquisition times (~45–70 ms),
      while the interior is tightly clustered around ~98–103 ms. The bimodal option
      mirrors this by drawing a rare "edge" lifetime with probability `p_edge`,
      so train data reflects border vs interior SNR differences.

    What it does:
      - Reads the current spectrum (assumed in counts).
      - Samples a virtual acquisition time 't_virtual' from the given range(s).
      - Forms Poisson rate λ = spec * (t_virtual / t_life_orig) and draws noise.
      - Writes back 't_life = t_virtual' so that downstream 'NormIT' divides by
        the same dwell time it used to scale counts. Result: mean amplitude after
        'NormIT' matches validation; only SNR changes.

    Units:
      Accepts "s", "ms", "us" suffixes (e.g., "45ms"). Bare numbers are seconds.

    Train-only. Deterministic if you pass a seeded RNG.
    """

    def __init__(
        self,
        rnd_lt: list | tuple | None = None,  # e.g. ["0.098s","0.103s"]
        rng: np.random.Generator = None,
    ):
        super().__init__()
        self.rng = rng or np.random.default_rng()
        self.rnd_lt = self._parse_range(rnd_lt)

    @staticmethod
    def _parse_val(x):
        if isinstance(x, (int, float)):
            # assume seconds
            return float(x)
        s = str(x).strip().lower()
        if s.endswith("ms"):
            return float(s[:-2]) * 1e-3  # milliseconds to seconds
        if s.endswith("us"):
            return float(s[:-2]) * 1e-6  # microseconds to seconds
        if s.endswith("s"):
            return float(s[:-1])  # seconds
        return float(s)  # default: seconds

    def _parse_range(self, range):
        assert (
            isinstance(range, (list, tuple)) and len(range) == 2
        ), "Range must be a list or tuple of two values."
        a, b = self._parse_val(range[0]), self._parse_val(range[1])
        low, high = (a, b) if a <= b else (b, a)
        return float(low), float(high)

    def __call__(self, batch, is_real_data: bool = False):
        if is_real_data or batch.get("_has_noise", False):
            return batch
        if isinstance(batch, list):
            batch = batch[0]
        assert isinstance(batch, dict), "Batch must be dict"

        spec = batch["spectra"]
        spec_np = (
            spec.cpu().numpy()
            if isinstance(spec, torch.Tensor)
            else np.asarray(spec, dtype=np.float64)
        )

        t_orig = float(batch.get("t_life", 0.0))
        if not np.isfinite(t_orig) or t_orig <= 0:
            raise ValueError(f"T_LIFE invalid: {t_orig}.")

        t_virtual = self.rng.uniform(*self.rnd_lt)  # seconds
        lam = spec_np * (t_virtual / t_orig)
        if np.any(lam < 0) or not np.all(np.isfinite(lam)):
            raise ValueError(
                f"Invalid lambda values for Poisson: {lam}. Ensure spectra and T_LIFE are valid."
            )

        noisy = self.rng.poisson(lam).astype(np.float64)
        print("lam:", lam[300:400])
        print("noisy:", noisy[300:400])

        batch["spectra"] = torch.as_tensor(noisy, dtype=torch.float32)
        batch.setdefault("_t_life_orig", float(t_orig))
        batch["t_life"] = float(t_virtual)  # update t_life to virtual one
        batch["_has_noise"] = True
        print(f"AddNoise: T_LIFE changed from {t_orig:.4f}s to {t_virtual:.4f}s")
        # print(
        #     f"T_LIFE ORIGINAL (before Noise): {batch['_t_life_orig']}, T_LIFE VIRTUAL (after Noise): {batch['t_life']}"
        # )
        return batch

    def __repr__(self):
        return f"AddNoise(rnd_lt={self.rnd_lt})"


class NormIT(object):
    """
    Normalization by acquisition time and beam strength.
    Converts from raw detected counts to counts persecond per unit current.
    This makes spectra comparable across different detector conditions.
    Where: After AddNoise
    """

    def __init__(self, cfg):
        real = cfg.get("REAL_METADATA", {})
        synth = cfg.get("SYNTH_METADATA", {})
        self.defs = {
            True: dict(
                t_life=float(real["T_LIFE"]),
                tube=float(real["TUBE_CURRENT"]),
                acq_time_unit=str(real["ACQ_TIME_UNIT"]).lower(),
                tube_unit=str(real["TUBE_CURRENT_UNIT"]).lower(),
            ),
            False: dict(
                t_life=float(synth["T_LIFE"]),
                tube=float(synth["TUBE_CURRENT"]),
                acq_time_unit=str(synth["ACQ_TIME_UNIT"]).lower(),
                tube_unit=str(synth["TUBE_CURRENT_UNIT"]).lower(),
            ),
        }

    @staticmethod
    def _to_seconds(t, unit):
        u = (unit or "s").lower()
        t = float(t)
        return t * 1e-3 if u == "ms" else t * 1e-6 if u == "us" else t

    @staticmethod
    def _to_amps(i, unit):
        u = (unit or "a").lower()
        i = float(i)
        if u == "ua":
            return i * 1e-6
        if u == "ma":
            return i * 1e-3
        return i  # "a"

    def __call__(self, batch, is_real_data: bool = False):
        # TODO: Tube Current and T_Life are not part of the metadata. Now it is set to 0.2 and 1 respectively as a default but needs further checking.
        if isinstance(batch, list):
            batch = batch[0]
        d = self.defs[is_real_data]
        t_life = batch.get("t_life")
        if t_life is None and "acq_time" in batch:
            t_life = self._to_seconds(batch["acq_time"], d["acq_time_unit"])  # real
        if t_life is None:
            t_life = d["t_life"]  # default value

        tube = batch.get("tube_current", d["tube"])
        tube = self._to_amps(tube, d["tube_unit"])  # convert to Amps

        if not batch.get("_has_normIT", False):
            denom = max(float(t_life) * tube, 1e-12)  # avoid division by zero
            batch["spectra"] = batch["spectra"] / denom
            batch["_has_normIT"] = True
            batch["t_life"] = float(t_life)
            batch["tube_current"] = float(tube)
        return batch

    def __repr__(self):
        return self.__class__.__name__ + "()"


class NormSqrt(object):
    """
    Variance-Stabilizing Transformation (VST) using square root transformation.
    Used for:
    - Poisson-distributed data (e.g. photon counts)
    - Reducing dynamic range while preserving noise structure
    - Making spectra easier for model to learn (less skew)

    Where: After everything else, including AddNoise and NormIT and LogTransform.
    """

    def __init__(self):
        pass

    def __call__(self, batch, is_real_data: bool = False):
        assert isinstance(batch, dict), "Batch should be a dictionary."

        if batch["_has_normSqrt"] == True:
            print(f"Sample has already been NormSqrt scaled. Returning it as is.")
            return batch

        spec_orig = batch["spectra"]

        if isinstance(batch["spectra"], torch.Tensor):
            batch["spectra"] = torch.sqrt(torch.clamp(batch["spectra"], min=0))
        else:
            batch["spectra"] = np.sqrt(np.maximum(batch["spectra"], 0))
        batch["_has_normSqrt"] = True

        return batch

    def __repr__(self):
        return self.__class__.__name__ + "()"


class AnscombeTransform(object):
    """
    Anscombe Variance-Stabilizing Transformation (VST) for Poisson-distributed data.

    Purpose:
    - Stabilizes variance across intensity range, making noise more homoscedastic.
    - Especially effective for photon count data with low to moderate counts.
    - Reduces dynamic range while preserving relative peak intensities.

    Formula:
        y = 2 * sqrt(x + 3/8)

    Where:
        x : input spectrum values (counts), must be >= 0 after baseline correction.

    Usage:
    - Place after baseline correction and NormIT in the preprocessing pipeline.
    - For real data: Smoothing -> BaselineCorrection -> NormIT -> Anscombe
    - For synthetic data: augmentations -> Smoothing -> BaselineCorrection -> NormIT -> Anscombe

    Notes:
    - Do not combine with NormSqrt or LogTransform (choose one VST only).
    - Clamp negative values to 0 before transformation to avoid NaNs.
    """

    def __call__(self, batch, is_real_data: bool = False):
        assert isinstance(batch, dict), "Batch should be a dictionary."

        if batch.get("_has_anscombe", False):
            return batch

        spec = batch["spectra"]
        if isinstance(spec, torch.Tensor):
            spec = torch.sqrt(torch.clamp(spec, min=0) + 3.0 / 8.0) * 2.0
        else:
            spec = np.sqrt(np.maximum(spec, 0) + 3.0 / 8.0) * 2.0

        batch["spectra"] = spec
        batch["_has_anscombe"] = True

        return batch

    def __repr__(self):
        return self.__class__.__name__ + "()"


class MinMaxScaler(object):
    """
    Applies Min-Max scaling to each spectrum independently to a [0, 1] range.
    This preserves the shape and relative importance of small peaks.
    """

    def __init__(self):
        pass

    def __call__(self, batch, is_real_data: bool = False):
        if not batch.get("_has_minmaxscaled", False):
            spectra = batch["spectra"]

            # Keep batch dimension, find min/max along the feature dimension (dim=1)
            if spectra.dim() == 1:
                spectra = spectra.unsqueeze(0)  # [4096] -> [1, 4096]
            min_vals = torch.min(spectra, dim=1, keepdim=True)[0]
            max_vals = torch.max(spectra, dim=1, keepdim=True)[0]

            # Add a small epsilon to avoid division by zero for flat spectra
            denom = max_vals - min_vals + 1e-8

            scaled_spec = (spectra - min_vals) / denom
            batch["spectra"] = (
                scaled_spec.squeeze(0) if scaled_spec.dim() == 2 else scaled_spec
            )
            batch["_has_minmaxscaled"] = True
        return batch

    def __repr__(self):
        return self.__class__.__name__ + "()"
