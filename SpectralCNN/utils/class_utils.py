import os
import torch
import mlflow
import logging
from typing import Tuple
from pathlib import Path
import thesis.utils.tools as tools
import thesis.utils.callbacks as clb
from thesis.utils.mappings import LabelMapper
import thesis.models.architectures.cnn_v2 as archs
from thesis.preprocessing import dataset_loader as dl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import numpy as np
import pytorch_lightning as pl

# -- setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -- Model architecture mapping
_MODEL_MAPPING = {
    "baseline": archs.Baseline1DCNN,
    "attn-cnn": archs.Attention1DCNN,
    # "lstm": archs.CNNLSTM,
    "transformer": archs.ParallelHybridModel,
    "sanity-check": archs.SanityCheckCNN,
    "simple": archs.SimpleAttentionCNN,
}


def print_model_summary(
    model, detailed: bool = False, bunny: bool = True
) -> Tuple[int, int]:
    """
    Prints a summary of the model architecture and parameter counts, optionally a cute bunny.
    """
    print("\n----- Model Summary -----")
    if detailed:
        print(model)
        print("-" * 30)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:_}")
    print(f"Trainable parameters: {trainable_params:_}")
    print(f"Non-trainable parameters: {total_params - trainable_params:_}")
    print("-" * 30)
    if bunny:
        print("(\\_/)")
        print("(^_^)")
        print("(>.<)")
        print("\n")
    return total_params, trainable_params


def get_label_mapper(cfg: dict) -> LabelMapper:
    """
    Creates a LabelMapper instance for the currently configured dataset.
    """
    mappings_file_path = get_mappings_filepath(cfg)
    label_mapper = LabelMapper(mapping_file=mappings_file_path)
    return label_mapper


def build_trainer(
    cfg: dict,
    callbacks: list,
    run_path: str,
    logger=None,
    gradient_clip_val: float = 0.0,
    precision: str = "32-true",
    accumulate_grad_batches: int = 1,
    profiler=None,
) -> pl.Trainer:
    """Builds and configures a Torch Lightning Trainer instance."""
    acc, devs = trainer_accel(cfg)
    trainer_cfg = cfg["MODEL"]["LEARNING"]["TRAINER"]
    return pl.Trainer(
        max_epochs=trainer_cfg["MAX_EPOCHS"],
        default_root_dir=run_path,
        callbacks=callbacks,
        logger=logger if logger is not None else False,
        accelerator=acc,
        devices=devs,
        log_every_n_steps=trainer_cfg.get("LOG_EVERY_N_STEPS", 10),
        deterministic=True,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        gradient_clip_val=gradient_clip_val,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        profiler=profiler,
    )


def create_data_module(
    cfg: dict, with_test_dataloader: bool = True, stage: str = "fit"
) -> dl.SimpleSpectraLDM:
    label_mapper = get_label_mapper(cfg)
    num_classes = label_mapper.get_num_classes()
    set_cfg_num_classes(cfg, num_classes)
    set_cfg_dataset(cfg)

    base_data_path = "../data"
    all_configs = cfg["DATA"]["DATASET_KWARGS"]

    synthetic_config = next((c for c in all_configs if c["label"] == "synthetic"), None)
    real_config = next((c for c in all_configs if c["label"] == "real"), None)

    if not synthetic_config:
        raise ValueError("Synthetic data config is required!")

    dataset_name = cfg["DATA"]["SYNTHETIC_DATASET_NAME"]
    h5_path = os.path.join(base_data_path, dataset_name, "spectra", "spectra.h5")
    synthetic_config["ds_kwargs"]["h5_path"] = h5_path

    # real data config is optional
    if real_config and with_test_dataloader:
        measurement_dir = cfg["DATA"]["REAL_MEASUREMENT_DIR"]
        sample_path = os.path.join(
            base_data_path,
            "measurements",
            measurement_dir,
            dataset_name,
            "dataframe.parquet",
        )
        image_path = os.path.join(
            base_data_path,
            "measurements",
            measurement_dir,
            "combined_image.npy",
        )
        real_config["ds_kwargs"]["sample_path"] = sample_path
        real_config["ds_kwargs"]["image_path"] = image_path
    else:
        real_config = None

    # create data module
    data_module = dl.SimpleSpectraLDM(
        synthetic_config=synthetic_config,
        real_config=real_config,
        config=cfg,
        label_mapper=label_mapper,
        num_classes=num_classes,
        batch_size=cfg["MODEL"]["LEARNING"]["TRAINER"]["BATCH_SIZE"],
        num_workers=0,
    )
    data_module.mixture_setup(stage=stage)
    return data_module


def set_cfg_num_classes(cfg: dict, num_classes: int) -> None:
    # -- update config with correct number of classes ---
    if "MODEL" not in cfg:
        cfg["MODEL"] = {}
    cfg["MODEL"]["NUM_CLASSES"] = num_classes

    # --- update dataset-specific configs ---
    if "DATA" in cfg and "DATASET_KWARGS" in cfg["DATA"]:
        for ds_cfg in cfg["DATA"]["DATASET_KWARGS"]:
            if "ds_kwargs" in ds_cfg:
                ds_cfg["ds_kwargs"]["num_classes"] = num_classes


def set_cfg_dataset(cfg: dict) -> None:
    if "DATA" in cfg and "DATASET_KWARGS" in cfg["DATA"]:
        for ds_cfg in cfg["DATA"]["DATASET_KWARGS"]:
            if ds_cfg.get("label", "").lower() == "synthetic":
                ds_cfg["ds_class"] = dl.SpectraData
            if ds_cfg.get("label", "").lower() == "real":
                ds_cfg["ds_class"] = dl.SpectraDataReal


def setup_mlflow_training(
    model: str,
    cfg: dict,
    transforms_signature: str | None = None,
    is_finetuning: bool = False,
) -> tuple:
    """Outsourced setup for training model to keep the main function clean."""

    # -- Setup Paths
    os.makedirs(cfg["MODEL"]["LEARNING"]["TRAINER"]["RUN_PATH_TRAIN"], exist_ok=True)

    # -- MLFlow Setup
    if not is_finetuning:
        validation_mode = cfg["MODEL"]["LEARNING"]["TRAINER"].get(
            "VALIDATION_MODE", False
        )
        experiment_name = (
            f"{model.capitalize()}_Validation"
            if validation_mode
            else f"{model.capitalize()}"
        )
        mlflow.set_experiment(experiment_name)
        iteration = tools.get_next_iteration_number(experiment_name)
        # chosen_transforms = getattr(transforms_pipeline, "chosen_names", [])
        run_name = tools.generate_run_name(
            model, iteration, validation_mode, transforms_signature
        )
    else:
        experiment_name = f"{model.capitalize()}_FineTuning"
        mlflow.set_experiment(experiment_name)
        iteration = tools.get_next_iteration_number(experiment_name)
        run_name = f"Finetune_{model.capitalize()}_Run{iteration:02d}"

    return (
        run_name,
        iteration,
        validation_mode,
        experiment_name,
    )


def get_mappings_filepath(cfg: dict) -> str:
    """
    Constructs the canonical path to the label_mapping.json file
    based on the project's data root and the selected dataset.
    """
    data_root = Path("../data")
    dataset_name = cfg["DATA"]["SYNTHETIC_DATASET_NAME"]
    mappings_path = data_root / dataset_name / "mappings" / "label_mapping.json"
    if not mappings_path.exists():
        raise FileNotFoundError(f"Mappings file not found at: {mappings_path}")
    return str(mappings_path)


def build_model(model_name: str, cfg: dict) -> Tuple[torch.nn.Module, dict]:
    """
    Instantiates a model with a clean, model-specific configuration.
    """
    arch_cls = _MODEL_MAPPING.get(model_name)
    if arch_cls is None:
        raise ValueError(f"Unknown model name: '{model_name}'")

    # This will be a flat dictionary for logging/saving hyperparameters
    summary_kwargs = {"ARCH": arch_cls.__name__}
    model_kwargs = {}

    if model_name == "sanity-check":
        # SanityCheckCNN has no parameters
        model = arch_cls(cfg["NUM_CLASSES"])
        return model, summary_kwargs

    # --- Build model-specific kwargs ---
    if model_name == "transformer":
        # For the Transformer, we build the nested config dicts it expects.
        cnn_cfg = {
            "input_channels": 1,
            "input_length": cfg["INPUT_LENGTH"],
            "num_classes": -1,  # Headless CNN
            "dropout": 0.0,  # Dropout is handled in the main model's head
            "use_gap": False,  # Must return a sequence
            "num_convs": cfg["CNN"]["NUM_CONVS"],
            "num_filters": cfg["CNN"]["NUM_FILTERS"],
            "kernel_size": cfg["CNN"]["KERNEL_SIZE"],
            "pool_every": cfg["CNN"]["POOL_EVERY"],
            "norm_type": cfg["CNN"]["NORM"]["TYPE"],
            "gn_groups": cfg["CNN"]["NORM"]["GN_GROUPS"],
            "attention_block": cfg["ATTENTION"]["PER_BLOCK"],
            "se_reduction": cfg["ATTENTION"]["SE"]["SE_REDUCTION"],
            "eca_k": cfg["ATTENTION"]["ECA"]["ECA_KERNELS"],
            "use_bottleneck_mhsa": False,  # Transformer is the main attention block
        }

        transformer_cfg = {
            "d_model": cfg["TRANSFORMER"][
                "D_MODEL"
            ],  # The feature dimension for the transformer branch
            "nhead": cfg["TRANSFORMER"]["NUM_HEADS"],
            "num_encoder_layers": cfg["TRANSFORMER"]["NUM_ENCODER_LAYERS"],
            "dim_feedforward": cfg["TRANSFORMER"]["DIM_FEEDFORWARD"],
            "dropout": cfg["TRANSFORMER"]["DROPOUT"],
        }

        fusion_cfg = {
            "hidden_dim": cfg["FUSION"]["HIDDEN_DIM"],
            "dropout": cfg["FUSION"]["DROPOUT"],
            "num_classes": cfg["NUM_CLASSES"],
        }

        # This is the final dictionary passed to the model's __init__
        model_kwargs = {
            "cnn_cfg": cnn_cfg,
            "transformer_cfg": transformer_cfg,
            "fusion_cfg": fusion_cfg,
        }
        # For logging, we create a flat dictionary
        summary_kwargs.update(cnn_cfg)
        summary_kwargs.update(transformer_cfg)
        summary_kwargs.update(fusion_cfg)

    else:
        # For all other models, we build a single, flat kwargs dictionary.
        # Start with parameters common to all CNN-based models.

        model_kwargs = {
            "num_classes": cfg["NUM_CLASSES"],
            "num_convs": cfg["CNN"]["NUM_CONVS"],
            "dropout": cfg["HEADS"]["DROPOUT"],
            "norm_type": cfg["CNN"]["NORM"]["TYPE"],
            "kernel_size": cfg["CNN"]["KERNEL_SIZE"],
            "num_filters": cfg["CNN"]["NUM_FILTERS"],
            "attention_type": cfg["ATTENTION"]["PER_BLOCK"],
            "activation": cfg["CNN"]["NORM"]["NON_LINEARITY"],
            "group_norm_groups": cfg["CNN"]["NORM"]["GN_GROUPS"],
            "use_depthwise": cfg["CNN"]["USE_DEPTHWISE_SEPARABLE"],
            "verbose": False,
            "use_attention_pooling": cfg["HEADS"]["USE_ATTENTION_POOLING"],
        }

        if model_name == "baseline":
            # Baseline-specific params (if any) would go here
            pass

        summary_kwargs.update(model_kwargs)

    # instantiate model
    model = arch_cls(**model_kwargs)
    return model, summary_kwargs


def get_callbacks_list(objective: int, cfg: dict, cb_image: np.ndarray = None) -> list:
    """Allows to define different callbacks for different objectives.
    Objectives:
        0: Normal training
        1: Augmentation parameter search
        2: Model architecture parameter search
        3: Fine-tuning on real data
        4: testing only (no callbacks needed)

    Args:
        objective (int): Objective number.

    Returns:
        list: List of callbacks for the specified objective.
    """
    label_mapper: LabelMapper = get_label_mapper(cfg)
    monitor = cfg["MODEL"]["LEARNING"]["TRAINER"]["MONITOR_METRIC"]
    num_classes = cfg["MODEL"]["NUM_CLASSES"]
    patience = cfg["MODEL"]["LEARNING"]["TRAINER"]["PATIENCE"]
    mode = cfg["MODEL"]["LEARNING"]["TRAINER"]["MODE"]

    match objective:
        case 1:
            # objective 1: augmentation parameter search
            run_path = cfg["MODEL"]["LEARNING"]["TRAINER"]["RUN_PATH_HPO1"]
            return [
                ModelCheckpoint(
                    monitor=monitor,
                    mode=mode,
                    filename="aug-{val_macro_f1:.4f}",
                    dirpath=run_path,
                    save_top_k=1,
                ),
                EarlyStopping(monitor=monitor, mode=mode, patience=patience),
                clb.MulticlassificationMetricsCallback(
                    num_classes=num_classes,
                    class_names=label_mapper.get_classnames(),
                ),
                clb.RegenerateMixedCallback(),
            ]
        case 2:
            # objective 2: model architecture parameter search
            run_path = cfg["MODEL"]["LEARNING"]["TRAINER"]["RUN_PATH_HPO2"]
            return [
                ModelCheckpoint(
                    monitor=monitor,
                    mode=mode,
                    filename="model-{val_macro_f1:.4f}",
                    dirpath=run_path,
                    save_last=True,
                    save_top_k=1,
                ),
                EarlyStopping(
                    monitor=monitor,
                    mode=mode,
                    patience=patience,
                ),
                clb.MulticlassificationMetricsCallback(
                    num_classes=num_classes,
                    class_names=label_mapper.get_classnames(),
                ),
                clb.RegenerateMixedCallback(),
            ]
        case 3:
            # objective 3: fine-tuning on real data
            run_path = cfg["MODEL"]["LEARNING"]["TRAINER"]["RUN_PATH_FINETUNE"]
            return [
                ModelCheckpoint(
                    dirpath=run_path,
                    filename="finetuned-{epoch:02d}-{val_macro_f1:.4f}",
                    monitor=monitor,
                    mode=mode,
                    save_last=True,
                    save_top_k=1,
                ),
                EarlyStopping(
                    monitor=monitor,
                    mode=mode,
                    patience=patience,
                ),
                LearningRateMonitor(logging_interval="epoch"),
                clb.MulticlassificationMetricsCallback(
                    num_classes=num_classes,
                    class_names=label_mapper.get_classnames(),
                ),
                clb.ConfusionMatrixCallback(
                    split="pure",
                    num_classes=num_classes,
                    class_names=label_mapper.get_classnames(),
                    frequency=1,
                ),
                clb.PCAAccuracyCallback(
                    hdf5_path=cfg["DATA"]["DATASET_KWARGS"][0]["ds_kwargs"]["h5_path"],
                    dataset_key=f"{cfg['DATA']['DATASET_KWARGS'][0]['ds_kwargs'].get('data_names')[0]}/spectra",
                    n_components=2,
                    max_samples=None,
                    figsize=(12, 9),
                ),
                clb.MulticlassificationMetricsCallback(
                    num_classes=num_classes,
                    class_names=label_mapper.get_classnames(),
                    top_k=3,
                ),
                clb.SegmentationMapCallback(label_mapper=label_mapper, image=cb_image),
                # clb.RegenerateMixedCallback(), <<- should not mix around with real data
            ]
        case 4:
            # testing only, no callbacks needed
            return [
                clb.SegmentationMapCallback(
                    label_mapper=label_mapper,
                    image=cb_image,
                )
            ]
        case 0:
            # objective 0: normal training
            run_path = cfg["MODEL"]["LEARNING"]["TRAINER"]["RUN_PATH_TRAIN"]
            return [
                ModelCheckpoint(
                    dirpath=run_path,
                    filename="{epoch:02d}-{val_macro_f1:.4f}",
                    monitor=monitor,
                    mode=mode,
                    save_last=True,
                    save_top_k=1,
                ),
                EarlyStopping(
                    monitor=monitor,
                    mode=mode,
                    patience=patience,
                ),
                LearningRateMonitor(logging_interval="epoch"),
                clb.ConfusionMatrixCallback(
                    split="pure",
                    num_classes=num_classes,
                    class_names=label_mapper.get_classnames(),
                    frequency=10,
                ),
                # clb.GradCAMCallback(
                #     target_layer_name="convs.1",
                #     split="pure",
                #     num_images=4,
                #     frequency=10,
                # ),
                clb.MulticlassificationMetricsCallback(
                    num_classes=num_classes,
                    class_names=label_mapper.get_classnames(),
                    top_k=3,
                ),
                clb.SegmentationMapCallback(label_mapper=label_mapper, image=cb_image),
                clb.TopKPredictionCallback(
                    k=10,
                    label_mapper=label_mapper,
                ),
                clb.PCAAccuracyCallback(
                    hdf5_path=cfg["DATA"]["DATASET_KWARGS"][0]["ds_kwargs"]["h5_path"],
                    dataset_key=f"{cfg['DATA']['DATASET_KWARGS'][0]['ds_kwargs'].get('data_names')[0]}/spectra",
                    n_components=2,
                    max_samples=None,
                    figsize=(12, 9),
                ),
                clb.RegenerateMixedCallback(),
                # clb.TopKPredictionCallback(k=3, label_mapper=label_mapper)
            ]
        case _:
            raise ValueError(f"Unknown objective: {objective}. Must be 0, 1, 2, or 3.")


def get_key_run_params(cfg: dict) -> dict:
    """
    Extracts and flattens the most important run-specific parameters
    from the main config dictionary for clean MLflow logging.
    """
    model_cfg = cfg["MODEL"]
    learning_cfg = model_cfg["LEARNING"]
    trainer_cfg = learning_cfg["TRAINER"]
    loss_cfg = learning_cfg["LOSS"]

    params = {
        # --- Core Setup ---
        "model_arch": model_cfg["ARCH"],
        "batch_size": trainer_cfg["BATCH_SIZE"],
        "max_epochs": trainer_cfg["MAX_EPOCHS"],
        "training_seed": trainer_cfg["TRAINING_SEED"],
        # --- Loss Function ---
        "loss_use_focal": loss_cfg["USE_FOCAL"],
        "loss_smoothing_eps": model_cfg["SMOOTHING_EPS"],
        "use_soft_labels": model_cfg["USE_SOFT_LABELS"],
        # --- Optimizer & Scheduler ---
        "optimizer": learning_cfg["OPTIMIZER"]["OPTIM_NAME"],
        "weight_decay": learning_cfg["OPTIMIZER"]["WEIGHT_DECAY"],
        "scheduler": learning_cfg["SCHEDULER"]["NAME"],
    }

    # --- Learning Rate (Handles both single and differential LR) ---
    if model_cfg["ARCH"] == "transformer":
        rates = learning_cfg["OPTIMIZER"]["LEARNING_RATES"]
        params["lr_cnn"] = rates["CNN_BRANCH"]
        params["lr_transformer"] = rates["TRANSFORMER_BRANCH"]
        params["lr_fusion"] = rates["FUSION_HEAD"]
    else:
        params["learning_rate"] = learning_cfg["OPTIMIZER"]["LEARNING_RATE"]

    # --- Mix Augmentation & Soft Labels ---
    if model_cfg["USE_SOFT_LABELS"]:
        params["mix_max_beta"] = loss_cfg["MAX_BETA"]
        params["mix_ramp_epochs"] = loss_cfg["RAMP_EPOCHS"]

    return params


def calculate_class_counts(datamodule: dl.SimpleSpectraLDM, num_classes: int):
    """
    Calculate class counts (# of samples per class) from the pure training dataset.
    Note: This is only using the "pure" subset, not the mixed one in order to get the true class distribution.
    Mixtures would distort the true counts, could be considered though.
    """
    train_pure_dataset = (
        datamodule.train_pure
    )  # Or: mixed ones, then: datamodule.train_dataloader().dataset
    labels = [sample["labels_idx"] for sample in train_pure_dataset]
    unique_labels, counts = torch.unique(torch.tensor(labels), return_counts=True)
    class_counts = torch.zeros(num_classes)
    class_counts[unique_labels] = counts.float()
    return class_counts


def trainer_accel(cfg: dict):
    dev = cfg["MODEL"]["LEARNING"]["TRAINER"]["DEVICE"]
    use_gpu = (
        (dev == "cuda")
        or (dev == "gpu")
        or (dev == "auto" and torch.cuda.is_available())
    )
    if use_gpu:
        return "gpu", 1
    else:
        return "cpu", 1
