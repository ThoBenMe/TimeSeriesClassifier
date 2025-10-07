import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F
import logging
import math
import ujson
import thesis.models.architectures.cnn_v2 as archs
from thesis.configs.config_reader import load_config
from thesis.utils.mappings import LabelMapper
import time
from thesis.models.architectures.cnn_v2 import ArcFace

config = load_config("../configs/config.yml")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


### -- Lightning Wrapper for Models -- ###
class XRFClassifier(pl.LightningModule):
    def __init__(
        self,
        model,
        label_mapper: LabelMapper,
        num_training_steps: int,
        model_cfg: dict = {},
        use_soft_labels: bool = True,
        smoothing_eps: float = 0.1,
        use_focal: bool = False,
        learning_rate: float = 1e-3,
        automatic_uncertainty_weighting: bool = False,  # whether to use uncertainty weighting for multi-task loss
        use_arcface_head: bool = False,  # whether to use ArcFace head instead of linear layer
        warmup_epochs: int = 0,  # 0 = no curriculum learning
        class_counts: torch.Tensor | None = None,
    ):
        super().__init__()
        self.model = model
        self.use_arcface_head = use_arcface_head

        # self.model = torch.compile(self.model)  # speed up with torch 2.0 compiler
        self.log_vars = None
        self.label_mapper = label_mapper
        self.validation_step_outputs = []
        self.smoothing_eps = smoothing_eps
        self.learning_rate = learning_rate
        self.use_soft_labels = use_soft_labels
        self.num_training_steps = num_training_steps
        self.mappings_file_path = self.label_mapper.filepath
        self.num_classes = self.label_mapper.get_num_classes()
        self.loss_cfg = model_cfg["LEARNING"]["LOSS"]
        self.optim_cfg = model_cfg["LEARNING"]["OPTIMIZER"]
        self.warmup_epochs = warmup_epochs
        with open(self.mappings_file_path, "r") as f:
            self.mappings = ujson.load(f)

        # --- Class Balancing Weights ---
        cb = (
            self._compute_cb_weights(class_counts) if class_counts is not None else None
        )
        self.register_buffer("cb_weights", cb)

        # --- Loss Functions Setup ---
        self.criterion_soft = F.kl_div  # for soft labels, always KL Divergence
        if self.use_arcface_head:
            # --- for ArcFace ---
            logger.info("Using ArcFace head for hard labels.")
            final_channels = model.head_hard[-1].in_features  # feature dim
            self.arcface_hard = ArcFace(
                in_features=final_channels, out_features=self.num_classes
            )
            self.arcface_soft = ArcFace(
                in_features=final_channels, out_features=self.num_classes
            )
            self.criterion_hard = nn.CrossEntropyLoss(
                label_smoothing=0.0,  # NO label smoothing with ArcFace
                weight=self.cb_weights,
            )
        elif use_focal:
            # --- for Linear Head ---
            logger.info("Using Focal Loss for hard labels.")
            self.criterion_hard = lambda logits, y: self.focal_loss(
                logits,
                y,
                alpha=(self.cb_weights if self.cb_weights is not None else 1.0),
                gamma=2.0,
                reduction="mean",
            )
        else:
            logger.info(
                f"Using Cross-Entropy Loss with {self.smoothing_eps} smoothing episodes for hard labels."
            )
            self.criterion_hard = lambda logits, y: F.cross_entropy(
                logits,
                y,
                weight=self.cb_weights,
                label_smoothing=(
                    self.smoothing_eps if self.smoothing_eps > 0.0 else 0.0
                ),
            )

        # --- Other Setup ---
        self.num_epochs = config.get("TRAIN", {}).get("MAX_EPOCHS", 100)

        if automatic_uncertainty_weighting:
            logger.info("Using automatic uncertainty weighting for multi-task loss.")
            # learnable params representing log-precision of each task
            self.log_vars = nn.Parameter(torch.zeros(2))

        # --- Save Hyperparameters ---
        self.save_hyperparameters(
            ignore=["mappings_file", "class_counts", "label_mapper", "model"]
        )

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        dm = self.trainer.datamodule
        if hasattr(dm, "_current_epoch"):
            dm._current_epoch = self.current_epoch
        if hasattr(dm, "reseed_pipelines"):
            dm.reseed_pipelines(epoch=self.current_epoch, worker_id=0)

    def on_fit_start(self):
        if self.trainer is not None:
            self.num_epochs = self.trainer.max_epochs

    def training_step(self, batch: dict, batch_idx):
        # unpack
        spec = batch["spectra"]
        x = spec.flatten(start_dim=1).unsqueeze(1).float()
        feats, logits_hard, logits_soft = self(x)

        # labels
        y_hard = batch.get("labels_idx")
        y_soft = batch.get("soft_labels", None)

        # check which samples are mixed
        is_mixed = (
            batch.get("mix_weight", torch.ones(x.size(0), device=x.device)).view(-1)
            < 1.0
        )
        pure_mask = ~is_mixed
        zero = torch.tensor(0.0, device=x.device)

        # --- Calculate component losses ---
        if pure_mask.any():
            if self.use_arcface_head:
                logits = self.arcface_hard(feats[pure_mask], y_hard[pure_mask])
            else:
                logits = logits_hard[pure_mask]

            loss_hard = self.criterion_hard(logits, y_hard[pure_mask])
            acc_hard = (logits.argmax(1) == y_hard[pure_mask]).float().mean()
            self.log("train_acc", acc_hard, on_step=True, on_epoch=True, prog_bar=True)
        else:
            loss_hard = zero

        # --- Soft loss calculation ---
        if is_mixed.any():
            log_probs = F.log_softmax(logits_soft[is_mixed], dim=1)
            loss_soft = self.criterion_soft(
                log_probs, y_soft[is_mixed], reduction="batchmean"
            )
        else:
            loss_soft = zero

        # --- Combine losses based on training phase ---
        if self.current_epoch < self.warmup_epochs:
            # Warmup Phase of Curriculum Learning
            loss = loss_hard
        else:
            # Multi-task phase
            if self.log_vars is not None:
                # automatic uncertainty weighting
                precision_hard = torch.exp(-self.log_vars[0])
                loss_hard_weighted = precision_hard * loss_hard + self.log_vars[0]
                precision_soft = torch.exp(-self.log_vars[1])
                loss_soft_weighted = precision_soft * loss_soft + self.log_vars[1]
                loss = loss_hard_weighted + loss_soft_weighted

                self.log(
                    "loss_weights/hard", precision_hard, on_step=False, on_epoch=True
                )
                self.log(
                    "loss_weights/soft", precision_soft, on_step=False, on_epoch=True
                )
            else:
                # ! manual ramping
                max_beta = self.loss_cfg.get("MAX_BETA", 0.5)
                initial_beta = 0.05  # not zero, make model aware of mixtures
                ramp_epochs = self.loss_cfg.get("RAMP_EPOCHS", 20)

                ramp_value = (
                    (max_beta - initial_beta) * (self.current_epoch + 1) / ramp_epochs
                )
                beta = min(max_beta, initial_beta + ramp_value)
                alpha = 1.0 - beta
                loss = alpha * loss_hard + beta * loss_soft

        # --- Final Logging ---
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_soft_loss", loss_soft, on_epoch=True, sync_dist=True, on_step=False
        )
        return loss

    def validation_step(self, batch: dict, batch_idx, dataloader_idx=0):
        spec = batch["spectra"]
        x = spec.flatten(start_dim=1).unsqueeze(1).float()  # shape [B, 1, 4096]
        _, logits_hard, logits_soft = self(x)
        UNK_IDX = self.num_classes - 1  # last class is 'unknown'

        if dataloader_idx == 0:
            # PURE samples
            y_hard = batch["labels_idx"]
            loss = self.criterion_hard(logits_hard, y_hard)
            preds = logits_hard.argmax(dim=1)

            # EXCLUDE UNK SAMPLES FROM ACC CALCULATION
            mask = y_hard != UNK_IDX
            if mask.any():
                strict_match = preds[mask] == y_hard[mask]
                alias_match = self.is_alias_match(y_hard[mask], preds[mask])
                combined_match = strict_match | alias_match
                acc_combined = combined_match.float().mean()
                count = mask.sum()
            else:
                acc_combined = torch.tensor(0.0, device=self.device)
                count = torch.tensor(0, device=self.device)

            # some debug stats
            num_unk = (y_hard == UNK_IDX).sum().item()
            if num_unk > 0:
                logger.info(
                    f"Excluded {num_unk} 'UNK' samples from pure validation accuracy calculation."
                )

            output_dict = {
                "loss": loss,
                "acc": acc_combined,
                "count": count,
                "logits": logits_hard,
                "dataloader_idx": dataloader_idx,
            }
            self.validation_step_outputs.append(output_dict)
            return output_dict

        elif dataloader_idx == 1:
            # MIXED samples
            y_soft = batch.get("soft_labels")
            log_probs = F.log_softmax(logits_soft, dim=1)
            loss = self.criterion_soft(log_probs, y_soft, reduction="batchmean")
            preds = logits_soft.argmax(dim=1)
            true_labels: torch.Tensor = y_soft.argmax(dim=1)
            # EXCLUDE UNKs FROM ACC CALCULATION
            mask = true_labels != UNK_IDX

            if mask.any():
                strict_match = preds[mask] == true_labels[mask]
                alias_match = self.is_alias_match(true_labels[mask], preds[mask])
                combined_match: torch.Tensor = strict_match | alias_match
                acc_combined = combined_match.float().mean()
                count = mask.sum()
            else:
                acc_combined = torch.tensor(0.0, device=self.device)
                count = torch.tensor(0, device=self.device)

            output_dict = {
                "loss": loss,
                "acc": acc_combined,
                "count": count,
                "logits": logits_soft,
                "dataloader_idx": dataloader_idx,
            }
            self.validation_step_outputs.append(output_dict)
            return output_dict

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if not outputs:
            logger.warning("No validation outputs to aggregate.")
            self.validation_step_outputs.clear()
            return

        # separate outputs based on structure (pure have logits, mixed don't)
        pure_outputs = [o for o in outputs if o["dataloader_idx"] == 0]
        mix_outputs = [o for o in outputs if o["dataloader_idx"] == 1]

        # aggregate pure outputs
        if pure_outputs:
            total_pure_count = sum(x["count"] for x in pure_outputs)
            if total_pure_count > 0:
                weighted_pure_loss = (
                    sum(x["loss"] * x["count"] for x in pure_outputs) / total_pure_count
                )
                weighted_pure_acc = (
                    sum(x["acc"] * x["count"] for x in pure_outputs) / total_pure_count
                )
            else:
                weighted_pure_loss = torch.tensor(0.0, device=self.device)
                weighted_pure_acc = torch.tensor(0.0, device=self.device)
        else:
            weighted_pure_loss = torch.tensor(0.0, device=self.device)
            weighted_pure_acc = torch.tensor(0.0, device=self.device)

        # aggregate mixed outputs
        if mix_outputs:
            total_mix_count = sum(x["count"] for x in mix_outputs)
            if total_mix_count > 0:
                weighted_mix_loss = (
                    sum(x["loss"] * x["count"] for x in mix_outputs) / total_mix_count
                )
                weighted_mix_acc = (
                    sum(x["acc"] * x["count"] for x in mix_outputs) / total_mix_count
                )
            else:
                weighted_mix_loss = torch.tensor(0.0, device=self.device)
                weighted_mix_acc = torch.tensor(0.0, device=self.device)
        else:
            weighted_mix_loss = torch.tensor(0.0, device=self.device)
            weighted_mix_acc = torch.tensor(0.0, device=self.device)

        # --- Log aggregated metrics ---
        self.log("val_pure_loss", weighted_pure_loss, prog_bar=True)
        self.log("val_pure_acc", weighted_pure_acc, prog_bar=True)
        self.log("val_mix_loss", weighted_mix_loss, prog_bar=True)
        self.log("val_mix_acc", weighted_mix_acc, prog_bar=True)

        # overall metrics
        safe_pure_acc = max(weighted_pure_acc.item(), 0)
        safe_mix_acc = max(weighted_mix_acc.item(), 0)
        overall_acc = math.sqrt(safe_pure_acc * safe_mix_acc)
        self.log("val_overall_acc", overall_acc, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch: dict, batch_idx):
        # should be a dict. If not, we unpack it
        if isinstance(batch, list):
            batch = batch[0]
            assert isinstance(batch, dict), f"Expected dict, got {type(batch)}"

        spectra = batch["spectra"]  # shape [B, 4096]
        labels_idx = batch.get("labels_idx", None)  # shape [B] (or None)
        x = spectra.flatten(start_dim=1).unsqueeze(1).float()  # shape [B, 1, 4096]
        assert x.ndim == 3 and x.size(1) == 1, f"test_step: wrong x shape: {x.shape}"

        _, logits_hard, _ = self(x)
        preds = logits_hard.argmax(dim=1)  # shape [B]
        coords = (
            torch.stack([batch["x"], batch["y"]], dim=1)
            if "x" in batch and "y" in batch
            else None
        )

        if labels_idx is None:
            acc = torch.tensor(float("nan"), device=self.device)
        else:
            unk_idx = self.num_classes - 1
            mask = labels_idx != unk_idx  # filter out 'unknown' labels
            if mask.any():
                strict_match = preds[mask] == labels_idx[mask]
                alias_match = self.is_alias_match(labels_idx[mask], preds[mask])
                combined = strict_match | alias_match
                acc = combined.float().mean()
            else:
                acc = torch.tensor(0.0, device=self.device)

        self.log(
            "test_acc",
            acc,
            prog_bar=True,
        )

        return {
            "preds": preds,
            "labels_idx": labels_idx,
            "logits_hard": logits_hard,
            "coords": coords,
            "spectra": spectra,
        }

    def is_alias_match(
        self, true_idx: torch.Tensor, pred_idx: torch.Tensor
    ) -> torch.Tensor:
        idx_to_name = {v: k for k, v in self.mappings["numeral_mapping"].items()}
        alias_map = self.mappings.get("alias_map", {})
        matches = []
        for true, pred in zip(true_idx.tolist(), pred_idx.tolist()):
            true_name = idx_to_name.get(true)
            pred_name = idx_to_name.get(pred)
            if true_name is None or pred_name is None:
                matches.append(False)
                continue
            aliases = alias_map.get(true_name, [true_name])
            matches.append(pred_name in aliases)
        return torch.tensor(matches, device=true_idx.device)

    def _compute_cb_weights(
        self, counts: torch.Tensor, beta: float = 0.999
    ) -> torch.Tensor:
        """
        Compute class balancing weights based on counts.
        CB weights handle long-tail class imbalance (some classes with 10, some with 10000 samples).
        Normally: w_c = 1 / n_c, but overpenalizes rare classes.
        Class-balanced weighting uses effective number of samples:
          - effective_num_c = 1 - beta^(n_c)
          - w_c = (1 - beta) / effective_num_c --- with n_c being the number of samples in class c.
        If n_c is big: beta^(n_c) is small -> effective number ~1 -> small weight
        if n_c is small: beta^(n_c) is big -> effective number ~0 -> big weight

        Args:
            counts (torch.Tensor): _description_
            beta (float, optional): _description_. Defaults to 0.999.

        Returns:
            torch.Tensor: _description_
        """
        counts = counts.float().clamp_min(1.0)  # avoid division by zero
        eff_num = 1.0 - (beta**counts)
        w = (1.0 - beta) / eff_num
        return (w / w.mean()).to(torch.float32)

    def focal_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute the focal loss between inputs and targets.
        See: https://medium.com/data-scientists-diary/implementing-focal-loss-in-pytorch-for-class-imbalance-24d8aa3b59d9
        Why: with class imbalance, CE will be dominated by easy/majority class examples.
        FL: downweights easy examples, focuses on hard-to-classify samples.
        Formula: CE = -log( p_t ), where p_t is the probability of the true class.

        Args:
            inputs (torch.Tensor): The input tensor (logits).
            targets (torch.Tensor): The target tensor (ground truth labels).
            alpha (float): Weighting factor for the class. heavy-class-imbalance -> alpha=[0.75;0.9] for emphasis on rarer classes.
            In Multi-class setup, alpha can generalize to per-class weighting vector, like alpha=torch.tensor([0.1, 0.3, 0.6]) (needs to know imbalance ratio).
            gamma (float): Focusing parameter to reduce the relative loss for well-classified examples. 0 -> CE, increase -> shrink loss contrib. for well-classified examples. This helps to focus more on hard-to-classify examples.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

        Returns:
            torch.Tensor: Computed focal loss.
        """
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        # gather: picks probability p_t for each sample's correct class
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_(1e-8, 1.0)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        if isinstance(alpha, torch.Tensor):
            # alpha can be scalar or per-sample from class weights
            if alpha.ndim == 1:  # per class alpha
                alpha = alpha.to(inputs.device)[targets]
            alpha = alpha.to(inputs.device)
        weight = (alpha if isinstance(alpha, torch.Tensor) else alpha) * (
            (1 - pt) ** gamma
        )
        loss = -weight * log_pt
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss

    def configure_optimizers(self):
        """
        Configures optimizers and schedulers with support for differential learning
        rates (for the ParallelHybridModel architecture)
        """
        weight_decay = self.optim_cfg.get("WEIGHT_DECAY", 0.01)
        base_lr = self.learning_rate
        param_groups = []

        if isinstance(self.model, archs.ParallelHybridModel):
            logger.info(f"Configuring optimizer with differential learning rates...")

            # Define the learning rates for each part of the hybrid model
            lrs = self.optim_cfg.get("LEARNING_RATES", {})
            param_group_lrs = {
                self.model.cnn_branch: base_lr,
                self.model.transformer_embedding: lrs.get(
                    "TRANSFORMER_BRANCH", base_lr
                ),
                self.model.transformer_encoder: lrs.get("TRANSFORMER_BRANCH", base_lr),
                self.model.fusion_neck: lrs.get("FUSION_HEAD", base_lr),
                self.model.head_hard: lrs.get("FUSION_HEAD", base_lr),
                self.model.head_soft: lrs.get("FUSION_HEAD", base_lr),
            }

            # Create parameter groups for each module with its specific LR
            for module, lr in param_group_lrs.items():
                param_groups.append(
                    {
                        "params": [
                            p
                            for n, p in module.named_parameters()
                            if not any(nd in n.lower() for nd in no_decay)
                            and p.requires_grad
                        ],
                        "weight_decay": weight_decay,
                        "lr": lr,
                    }
                )
                param_groups.append(
                    {
                        "params": [
                            p
                            for n, p in module.named_parameters()
                            if any(nd in n.lower() for nd in no_decay)
                            and p.requires_grad
                        ],
                        "weight_decay": 0.0,
                        "lr": lr,
                    }
                )

        else:
            logger.info(
                f"[{self.model.__class__.__name__}]: Configuring optimizer with single base LR. Base LR: {base_lr}, Weight Decay: {weight_decay}."
            )
            # --- PARAM GROUP RULES ---
            no_decay = ["bias", "norm"]
            # params that should NOT have weight decay applied
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if not any(nd in n.lower() for nd in no_decay)
                        and p.requires_grad
                    ],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if any(nd in n.lower() for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]

            # --- CREATE OPTIMIZER ---
            optim_name = self.optim_cfg.get("OPTIM_NAME", "adamw").lower()
            if optim_name == "adamw":
                optimizer = optim.AdamW(optimizer_grouped_parameters, lr=base_lr)
            elif optim_name == "sgd":
                momentum = self.optim_cfg.get("MOMENTUM", 0.9)
                optimizer = optim.SGD(
                    optimizer_grouped_parameters, lr=base_lr, momentum=momentum
                )
            else:
                raise ValueError(f"Unsupported optimizer: {optim_name}")

            # --- Scheduler Setup ---
            # ReduceLROnPlateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",  # minimize validation loss
                factor=self.optim_cfg.get(
                    "LR_REDUCE_FACTOR", 0.2
                ),  # reduce LR by factor (e.g., 0.2 = 5)
                patience=self.optim_cfg.get(
                    "LR_REDUCE_PATIENCE", 5
                ),  # wait epochs before reducing
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_pure_loss",  # metric to monitor for ReduceLRO
                    "frequency": 1,
                    "interval": "epoch",
                },
            }
