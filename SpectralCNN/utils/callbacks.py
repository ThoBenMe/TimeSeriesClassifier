import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import shutil
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.metrics import (
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
    classification_report,
    top_k_accuracy_score,
)
from thesis.utils.wrappers import deprecated
import re
from thesis.preprocessing.pca_tsne import DataInspector
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize, LogNorm
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from thesis.utils.mappings import LabelMapper
from pathlib import Path
import ujson
from typing import Callable, Any, Optional, Union, Dict, List
from scipy.stats import entropy
import logging

logger = logging.getLogger(__name__)


class ConfusionMatrixCallback(pl.Callback):
    def __init__(
        self,
        split: str = "pure",
        num_classes: int = 10,
        class_names: list[str] | None = None,
        frequency: int = 10,
    ):
        super().__init__()
        assert split in (
            "pure",
            "aug",
            "combined",
        ), "split must be one of 'pure', 'aug', or 'combined'"
        self.split = split
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.frequency = frequency

    def on_validation_epoch_end(self, trainer, pl_module):
        """Computation is quite heavy, thus we might want to run it less frequently."""
        epoch = trainer.current_epoch
        if (epoch + 1) % self.frequency != 0:
            return

        # picking DL
        val_loaders = trainer.datamodule.val_dataloader()
        if not isinstance(val_loaders, (list, tuple)):
            val_loaders = [val_loaders]

        # pick right one
        loader_idx = {"pure": 0, "aug": 1, "combined": 2}[self.split]
        if loader_idx >= len(val_loaders):
            # only have 'pure' in this dm
            return

        # iterate dl and build conf matrix
        dl = val_loaders[loader_idx]
        preds_all = []
        targets_all = []
        pl_module.eval()

        with torch.no_grad():
            for batch in dl:
                x = (
                    batch["spectra"]
                    .flatten(start_dim=1)
                    .unsqueeze(1)
                    .to(pl_module.device)
                )
                targets = batch["labels_idx"].to(pl_module.device)
                (
                    _,
                    logits_h,
                    _,
                ) = pl_module(x)
                pred = logits_h.argmax(dim=1)
                preds_all.append(pred.cpu())
                targets_all.append(targets.cpu())

        preds = torch.cat(preds_all, dim=0).numpy()
        targets = torch.cat(targets_all, dim=0).numpy()

        # compute conf matrix
        cm = confusion_matrix(targets, preds, labels=list(range(self.num_classes)))

        # plotting
        n = self.num_classes
        per_class = 0.4
        min_size = 8
        fig_w = max(min_size, n * per_class)
        fig_h = max(min_size, n * per_class)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)

        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        # plt.colorbar(im, ax=ax)
        ax.set(
            xticks=range(n),
            yticks=range(n),
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            xlabel="Predicted",
            ylabel="True",
            title=f"Confusion Matrix - {self.split.capitalize()} @ epoch {epoch+1}",
        )

        # rotate and shrink tick labels so they do not overlap
        plt.setp(
            ax.get_xticklabels(),
            rotation=90,
            ha="center",
            fontsize=max(6, 12 - n // 10),
        )
        plt.setp(ax.get_yticklabels(), fontsize=max(6, 12 - n // 10))
        plt.tight_layout()

        # log it
        logger = trainer.logger
        if hasattr(logger, "experiment") and hasattr(logger.experiment, "log_figure"):
            logger.experiment.log_figure(
                run_id=logger.run_id,
                figure=fig,
                artifact_file=f"confusion_matrix_{self.split}.png",
            )
        elif hasattr(logger, "log_figure"):
            logger.log_figure(
                f"Confusion_Matrix/{self.split}", fig, global_step=epoch + 1
            )
        else:
            plt.show()

        plt.close(fig)


class PCAAccuracyCallback(pl.Callback):
    def __init__(
        self,
        hdf5_path: str,
        dataset_key: str,
        n_components: int = 2,
        max_samples: int = None,
        figsize: tuple = (10, 8),
    ):
        super().__init__()
        self.inspector = DataInspector(hdf5_path, dataset_key)
        self.n_components = n_components
        # print(
        #     f"Using {self.n_components} PCA components, Type: {type(self.n_components)}"
        # )
        self.max_samples = max_samples
        self.figsize = figsize
        self.pca = None
        self._val_raw_batches = []
        self._val_pred_batches = []
        self._val_true_batches = []

    def on_fit_start(self, trainer, pl_module):
        raw_data = self.inspector.load_data(max_samples=self.max_samples)
        if hasattr(self.inspector, "pca") and self.inspector.pca is not None:
            self.pca = self.inspector.pca
        else:
            self.pca = PCA(n_components=self.n_components)
            self.pca.fit(raw_data)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        raw_spec = batch["spectra"].detach().cpu().numpy()  # [B, 4096]

        x_tensor = (
            batch["spectra"]
            .flatten(start_dim=1)
            .unsqueeze(1)
            .float()
            .to(pl_module.device)
        )

        with torch.no_grad():
            _, logits_h, _ = pl_module(x_tensor)
            preds = logits_h.argmax(dim=1).cpu().numpy()  # [B]

        trues = batch["labels_idx"].detach().cpu().numpy()  # [B]

        # handle shape mismatch in fine-tune settings
        if raw_spec.ndim > 2:
            raw_spec = raw_spec.squeeze(1)
        self._val_raw_batches.append(raw_spec)
        self._val_pred_batches.append(preds)
        self._val_true_batches.append(trues)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._val_raw_batches:
            return

        # stack into arrays
        all_raw = np.vstack(self._val_raw_batches)  # [N_valid, 4096]
        all_preds = np.concatenate(self._val_pred_batches, axis=0)  # [N_valid]
        all_trues = np.concatenate(self._val_true_batches, axis=0)  # [N_valid]

        pcs = self.pca.transform(all_raw)
        correct = all_preds == all_trues

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.scatter(
            pcs[correct, 0],
            pcs[correct, 1],
            c="tab:blue",
            label="correct",
            s=10,
            alpha=0.7,
        )
        ax.scatter(
            pcs[~correct, 0],
            pcs[~correct, 1],
            c="tab:red",
            label="incorrect",
            s=10,
            alpha=0.7,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PCA of Validation Spectra (epoch: {trainer.current_epoch})")
        ax.legend(loc="best")
        plt.tight_layout()

        # logging
        logger = trainer.logger
        if isinstance(logger, MLFlowLogger):
            logger.experiment.log_figure(
                run_id=logger.run_id,
                figure=fig,
                artifact_file="PCA/accuracy_scatter.png",
            )
        elif hasattr(logger, "experiment") and hasattr(logger.experiment, "add_figure"):
            logger.experiment.add_figure(
                "PCA/accuracy_scatter", fig, global_step=trainer.current_epoch
            )
        else:
            plt.show()

        plt.close(fig)
        self._val_raw_batches.clear()
        self._val_pred_batches.clear()
        self._val_true_batches.clear()


@deprecated
class AttentionWeightsCallback(pl.Callback):
    def __init__(self, val_dataloader, attention_layer_name, num_images=1):
        super().__init__()
        self.val_dataloader = val_dataloader
        self.attention_layer_name = attention_layer_name
        self.num_images = num_images

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.model.eval()

        # grab one batch
        batch = next(iter(self.val_dataloader))
        data = batch["spectra"].unsqueeze(1).to(pl_module.device)
        labels = batch["labels_idx"].to(pl_module.device)

        # find the attention module
        modules = dict(pl_module.model.named_modules())
        attn_layer = modules.get(self.attention_layer_name)
        if attn_layer is None:
            print(f"Attention layer '{self.attention_layer_name}' not found!")
            return

        attn_weights_list = []

        def _hook(_module, _input, _output):
            if hasattr(_module, "attention_weights"):
                attn_weights_list.append(_module.attention_weights.clone().detach())

        handle = attn_layer.register_forward_hook(_hook)

        # **unpack all three outputs** to trigger the hook
        with torch.no_grad():
            feats, logit_h, logit_s = pl_module(data)

        handle.remove()

        if not attn_weights_list:
            print("No attention weights captured.")
            return

        attn_weights = attn_weights_list[0]  # [B, heads, L]
        B, heads, L = attn_weights.shape

        for i in range(min(self.num_images, B)):
            w = attn_weights[i].cpu().numpy()  # [heads, L]
            fig, axs = plt.subplots(1, heads, figsize=(3 * heads, 3))
            if heads == 1:
                axs = [axs]
            lbl = labels[i].item() if labels.dim() == 1 else labels[i].argmax().item()
            for h in range(heads):
                ax = axs[h]
                arr = w[h : h + 1]  # keep 2D
                ax.imshow(arr, aspect="auto", cmap="viridis")
                ax.set_title(f"Head {h}")
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f"Attention weights for sample {i}, label {lbl}")

            # log or show
            logger = trainer.logger
            if hasattr(logger, "experiment") and hasattr(
                logger.experiment, "add_figure"
            ):
                logger.experiment.add_figure(f"Attention/{i}", fig, trainer.global_step)
            else:
                plt.show()
            plt.close(fig)

        pl_module.train()


class MulticlassificationMetricsCallback(pl.Callback):
    """
    PyTorch Lightning callback to compute and log multiclass classification metrics
    at the end of each validation epoch, using only the 'pure' validation loader.

    For each validation batch (dataloader_idx == 0), it:
      - Retrieves the model's raw logits on that batch
      - Accumulates logits and true labels in memory

    At the end of validation:
      - Concatenates all buffered logits and targets
      - Computes:
          o Macro-F1: unweighted mean of per-class F1 scores (equal performance on all classes, also rare ones)
          o Weighted-F1: mean of per-class F1 scores weighted by class support (overall performance, reflecting actual class distribution)
          o Top-K Accuracy: fraction of samples whose true label is among the top K logits
      - Logs 'val_macro_f1', 'val_weighted_f1', and 'val_top{K}_acc' to the Lightning recorder
      - Clears the buffers for the next epoch

    Args:
        num_classes (int): Total number of classes (C).
        class_names (list[str], optional): Human-readable names for each class, for reporting.
            Defaults to stringified integers ["0", "1", …].
        top_k (int, optional): K for top-K accuracy. If K >= num_classes, top-K is set to 0.0.
    """

    def __init__(
        self,
        num_classes: int,
        class_names: list[str] | None = None,
        top_k: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.top_k = top_k

        # buffers
        self.logits_buffer = []
        self.targets = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        # only run on PURE samples (dataloader_idx == 0)
        if dataloader_idx != 0:
            return

        logits_h = outputs["logits"]
        target = batch["labels_idx"].to(pl_module.device)

        # store for epoch-end
        self.logits_buffer.append(logits_h.cpu())
        self.targets.append(target.cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        # concatenate
        logits = torch.cat(self.logits_buffer, dim=0)  # [N, C]
        targets = torch.cat(self.targets, dim=0)  # [N]
        assert logits.shape[0] == targets.shape[0]
        logits_np = logits.cpu().numpy()
        targets_np = targets.cpu().numpy()

        # preds + F1
        preds_np = logits_np.argmax(axis=1)
        macro_f1 = f1_score(targets_np, preds_np, average="macro", zero_division=0)
        weighted_f1 = f1_score(
            targets_np, preds_np, average="weighted", zero_division=0
        )

        # top-K
        if self.top_k < self.num_classes:
            all_labels = list(range(self.num_classes))
            topk_acc = top_k_accuracy_score(
                y_true=targets_np,
                y_score=logits_np,
                k=self.top_k,
                labels=all_labels,
            )
        else:
            topk_acc = 0.0

        # log
        pl_module.log(
            "val_macro_f1", macro_f1, prog_bar=True, on_epoch=True, on_step=False
        )
        pl_module.log("val_weighted_f1", weighted_f1, prog_bar=True, on_epoch=True)
        pl_module.log(f"val_top{self.top_k}_acc", topk_acc, prog_bar=True)

        # clear
        self.logits_buffer.clear()
        self.targets.clear()


class RegenerateMixedCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        dm = trainer.datamodule
        if dm is not None and getattr(dm, "train_mixed", None) is not None:
            dm.train_mixed.regenerate()

    # ! validation set can also be regenerated if needed, but preferably kept static
    # def on_fit_start(self, trainer, pl_module):
    #     dm = trainer.datamodule
    #     if dm is not None and getattr(dm, "val_mixed", None) is not None:
    #         torch.manual_seed(1234)
    #         dm.val_mixed.regenerate()  # regenerate mixed samples for validation


class TopKPredictionCallback(pl.Callback):
    def __init__(
        self, k: int = 3, label_mapper=None, out_path: str = "./top_k_predictions.txt"
    ):
        self.k = k
        self.label_mapper: LabelMapper = label_mapper
        self._entries = []
        self.out_path = out_path

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=1
    ):
        if dataloader_idx != 1:
            return

        device = pl_module.device
        x = batch["spectra"].unsqueeze(1).float().to(device)
        with torch.no_grad():
            _, _, logits_soft = pl_module(x)

        probs = F.softmax(logits_soft, dim=1)
        topk_vals, topk_idxs = torch.topk(probs, self.k, dim=1)

        for i in range(len(topk_idxs)):
            labels = [
                self.label_mapper.get_classname_by_idx(idx.item())
                for idx in topk_idxs[i]
            ]

            scores = topk_vals[i].tolist()
            line = (
                f"Epoch {trainer.current_epoch} | "
                f"Batch {batch_idx} | Sample {i}: "
                f"Top‐{self.k} = {list(zip(labels, scores))}"
            )
            self._entries.append(line)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._entries:
            return

        with open(self.out_path, "w", encoding="utf-8") as f:
            for line in self._entries:
                f.write(line + "\n")
        self._entries.clear()


class SegmentationMapCallback(pl.Callback):
    """
    Callback to visualize and log segmentation results after testing.

    This callback constructs and logs:
    - The original grayscale image.
    - Ground-truth and predicted segmentation maps.
    - Overlay maps (segmentation mask overlaid on the original image).
    - A separate class distribution figure showing counts and percentages per class.

    It supports both MLflow and generic loggers and uses a custom colormap to ensure
    non-contiguous class indices are handled correctly.

    Args:
        label_mapper: Object providing a `get_classnames()` method and access to the `numeral_mapping` dictionary.
    """

    def __init__(
        self,
        image: np.ndarray,
        label_mapper=None,
        topk: int = 3,
        save_dir="./segmentation_output",
    ):
        pass
