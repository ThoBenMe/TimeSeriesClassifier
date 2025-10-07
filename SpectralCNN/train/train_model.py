import os
import copy
import time
import json
import torch
import shutil
import optuna
import mlflow
import logging
import argparse
import warnings
import numpy as np
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import thesis.utils.class_utils as cutils
from thesis.utils.mappings import LabelMapper
from thesis.configs.config_reader import load_config
import thesis.preprocessing.transforms as transforms
from thesis.preprocessing import dataset_loader as dl
from thesis.models.lightning.xrf_model import (
    XRFClassifier,
)
from thesis.models.architectures.cnn_v2 import ParallelHybridModel
from thesis.utils.tools import build_transforms_pipeline
import pandas as pd
from pytorch_lightning.profilers import AdvancedProfiler

# set precision
torch.set_float32_matmul_precision("high")
profiler = AdvancedProfiler(dirpath=".", filename="perf_logs.txt")

# -- setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings(
    "ignore",
    message="The 'train_dataloader' does not have many workers",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The 'val_dataloader' does not have many workers",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples",
)
logging.getLogger("mlflow").setLevel(logging.WARNING)


def objective_phase1(
    trial: optuna.Trial,
    modelname: str,
    cfg: dict,
) -> float:
    config = copy.deepcopy(cfg)
    use_focal = config["MODEL"]["LEARNING"]["LOSS"].get("USE_FOCAL", False)
    run_path = f"{config['MODEL']['LEARNING']['TRAINER']['RUN_PATH_HPO1']}/{modelname}"
    config["MODEL"]["LEARNING"]["TRAINER"]["RUN_PATH_HPO1"] = run_path
    automatic_uncertainty_weighting = config["MODEL"].get(
        "AUTOMATIC_UNCERTAINTY_WEIGHTING", False
    )
    os.makedirs(run_path, exist_ok=True)

    # DataModule & Model instantiation
    dm = cutils.create_data_module(cfg=config, with_test_dataloader=False)
    cnn, _ = cutils.build_model(modelname, config["MODEL"])
    label_mapper = cutils.get_label_mapper(cfg)
    num_classes = label_mapper.get_num_classes()
    class_counts = cutils.calculate_class_counts(dm, num_classes)
    mappings_file_path = cutils.get_mappings_filepath(cfg)
    model_kwargs = {
        "model": cnn,
        "model_cfg": config["MODEL"],
        "use_soft_labels": config["MODEL"]["USE_SOFT_LABELS"],
        "mappings_file": mappings_file_path,
        "num_classes_cfg": num_classes,
        "class_counts": class_counts,
        "use_focal": use_focal,
        "automatic_uncertainty_weighting": automatic_uncertainty_weighting,
    }
    model = XRFClassifier(**model_kwargs)

    # 4) Callbacks: monitor macro-F1 for pruning & early stop
    callbacks = cutils.get_callbacks_list(
        objective=1, cfg=config, cb_image=dm.test_dataloader().image
    )

    trainer = cutils.build_trainer(
        cfg=config,
        run_path=run_path,
        callbacks=callbacks,
        gradient_clip_val=1.0,
    )
    # _, _ = cutils.print_model_summary(model)
    trainer.fit(model, dm)
    return float(trainer.callback_metrics["val_macro_f1"])


def objective_phase2(
    trial: optuna.Trial, cfg: dict, modelname: str = "baseline"
) -> float:
    """
    Optuna objective function for Phase 2: model architecture parameter search.

    This phase optimizes the architecture and learning rate of the CNN (or hybrid model)
    while using a fixed transformation pipeline. It evaluates the validation
    macro F1-score for model selection.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object for parameter suggestion and tracking.
    modelname : str, optional
        Name of the model architecture to optimize (default is "baseline").

    Returns
    -------
    float
        Validation macro F1-score (higher is better).
    """
    config = copy.deepcopy(cfg)
    config["MODEL"]["LEARNING"]["OPTIMIZER"]["WEIGHT_DECAY"] = trial.suggest_float(
        "weight_decay", 1e-5, 1e-2, log=True
    )

    if modelname == "transformer":
        config["MODEL"]["CNN"]["NUM_CONVS"] = trial.suggest_int(
            "cnn_num_convs", 3, 5, 7
        )
        config["MODEL"]["CNN"]["NUM_FILTERS"] = trial.suggest_categorical(
            "cnn_num_filters", [32, 64]
        )
        config["MODEL"]["CNN"]["KERNEL_SIZE"] = trial.suggest_categorical(
            "cnn_kernel_size", [3, 5, 7, 9]
        )
        config["MODEL"]["CNN"]["POOL_EVERY"] = trial.suggest_categorical(
            "cnn_pool_every", [1, 2]
        )

        # -- Transformer Branch (with dependent parameters) --
        d_model = trial.suggest_categorical("tf_d_model", [64, 128, 256])
        config["MODEL"]["TRANSFORMER"]["D_MODEL"] = d_model

        # Pro move: Ensure nhead is always a valid divisor of d_model
        valid_nheads = [h for h in [2, 4, 8, 16] if d_model % h == 0]
        nhead = trial.suggest_categorical("tf_nhead", valid_nheads)
        config["MODEL"]["TRANSFORMER"]["NUM_HEADS"] = nhead

        config["MODEL"]["TRANSFORMER"]["NUM_ENCODER_LAYERS"] = trial.suggest_int(
            "tf_num_layers", 1, 4
        )
        config["MODEL"]["TRANSFORMER"]["DIM_FEEDFORWARD"] = trial.suggest_categorical(
            "tf_dim_feedforward", [d_model * 2, d_model * 4]
        )
        config["MODEL"]["TRANSFORMER"]["DROPOUT"] = trial.suggest_float(
            "tf_dropout", 0.1, 0.3, step=0.05
        )

        # -- Fusion Head --
        config["MODEL"]["FUSION"]["HIDDEN_DIM"] = trial.suggest_categorical(
            "fusion_hidden_dim", [256, 512, 1024]
        )
        config["MODEL"]["FUSION"]["DROPOUT"] = trial.suggest_float(
            "fusion_dropout", 0.2, 0.5, step=0.1
        )

        # -- Differential Learning Rates --
        opt_cfg = config["MODEL"]["LEARNING"]["OPTIMIZER"]
        opt_cfg["LEARNING_RATES"]["CNN_BRANCH"] = trial.suggest_float(
            "lr_cnn", 1e-5, 5e-4, log=True
        )
        opt_cfg["LEARNING_RATES"]["TRANSFORMER_BRANCH"] = trial.suggest_float(
            "lr_tf", 5e-5, 1e-3, log=True
        )
        opt_cfg["LEARNING_RATES"]["FUSION_HEAD"] = trial.suggest_float(
            "lr_fusion", 1e-4, 5e-3, log=True
        )

    elif modelname == "attn-cnn":
        config["MODEL"]["CNN"]["NUM_CONVS"] = trial.suggest_int("num_convs", 3, 8)
        config["MODEL"]["CNN"]["NUM_FILTERS"] = trial.suggest_categorical(
            "num_filters", [16, 32, 64, 128]
        )
        config["MODEL"]["CNN"]["KERNEL_SIZE"] = trial.suggest_categorical(
            "kernel_size", [3, 5, 7]
        )
        config["MODEL"]["HEADS"]["DROPOUT"] = trial.suggest_float("dropout", 0.1, 0.5)

        # -- Single Learning Rate --
        config["MODEL"]["LEARNING"]["OPTIMIZER"]["LEARNING_RATE"] = trial.suggest_float(
            "lr", 1e-5, 1e-3, log=True
        )
        config["MODEL"]["ATTENTION"]["PER_BLOCK"] = trial.suggest_categorical(
            "attention_block", [None, "se", "eca", "temporal"]
        )
        if config["MODEL"]["ATTENTION"]["PER_BLOCK"] == "temporal":
            config["MODEL"]["ATTENTION"]["TEMPORAL"]["TEMPORAL_KERNELS"] = (
                trial.suggest_categorical("temporal_k", [3, 5, 7, 9])
            )
        config["MODEL"]["ATTENTION"]["BOTTLENECK_MHSA"]["ENABLED"] = (
            trial.suggest_categorical("use_bottleneck_mhsa", [True, False])
        )
        if config["MODEL"]["ATTENTION"]["BOTTLENECK_MHSA"]["ENABLED"]:
            config["MODEL"]["ATTENTION"]["BOTTLENECK_MHSA"]["HEADS"] = (
                trial.suggest_categorical("mhsa_heads", [4, 8])
            )
            config["MODEL"]["ATTENTION"]["BOTTLENECK_MHSA"]["DROPOUT"] = (
                trial.suggest_categorical("mhsa_dropout", [0.1, 0.2, 0.3])
            )
        config["MODEL"]["CNN"]["POOL_EVERY"] = trial.suggest_categorical(
            "pool_every", [1, 2]
        )
        config["AUTOMATIC_UNCERTAINTY_WEIGHTING"] = trial.suggest_categorical(
            "automatic_uncertainty_weighting", [True, False]
        )
    elif modelname == "baseline":
        pass  # skip baseline for now since we lazy

    # Update the model architecture in the config based on the function argument
    run_path = f"{config['MODEL']['LEARNING']['TRAINER']['RUN_PATH_HPO2']}/{modelname}"
    config["MODEL"]["LEARNING"]["TRAINER"]["RUN_PATH_HPO2"] = run_path
    os.makedirs(run_path, exist_ok=True)
    config["MODEL"]["ARCH"] = modelname
    num_classes = config["MODEL"]["NUM_CLASSES"]
    use_focal = config["MODEL"]["LEARNING"]["LOSS"].get("USE_FOCAL", False)
    dm = cutils.create_data_module(config, with_test_dataloader=False)
    class_counts = cutils.calculate_class_counts(dm, num_classes)
    model_arch, _ = cutils.build_model(modelname, config["MODEL"])
    mappings_file_path = cutils.get_mappings_filepath(cfg)

    model_lightning = XRFClassifier(
        model=model_arch,
        model_cfg=config["MODEL"],
        use_soft_labels=config["MODEL"]["USE_SOFT_LABELS"],
        num_classes_cfg=config["MODEL"]["NUM_CLASSES"],
        mappings_file=mappings_file_path,
        smoothing_eps=config["MODEL"].get("SMOOTHING_EPS", 0.1),
        class_counts=class_counts,
        use_focal=use_focal,
        automatic_uncertainty_weighting=automatic_uncertainty_weighting,
    )
    trainer_cfg = config["MODEL"]["LEARNING"]["TRAINER"]

    # Callbacks on macro-F1
    callbacks = cutils.get_callbacks_list(
        objective=2, cfg=config, cb_image=dm.mineral_image
    )

    trainer = cutils.build_trainer(
        cfg=config,
        run_path=run_path,
        callbacks=callbacks,
        gradient_clip_val=1.0,
    )

    try:
        _, _ = cutils.print_model_summary(model_lightning)
        trainer.fit(model_lightning, dm)
        return trainer.callback_metrics[trainer_cfg["MONITOR_METRIC"]].item()
    except Exception as e:
        print(f"Trial failed with exception: {e}")
        return 0.0


def fine_tune_model(
    real_parquet: str,
    label_mapper: LabelMapper,
    rng: np.random.Generator,
    cfg: dict,
    epochs: int = 10,
    lr: float = 1e-4,
    device: str = "auto",
    model_name: str = "baseline",
) -> str:
    """
    Fine-tune a pretrained synthetic model on real spectral data.

    Loads a model from a synthetic training checkpoint, freezes lower-level
    layers, and retrains on real data using provided transformations and metadata.
    Tracks results via MLflow and evaluates the model on validation and test splits.

    Parameters
    ----------
    real_parquet : str
        Path to real-world data in Parquet format.
    image_path : str
        Path to directory where segmentation maps will be saved.
    real_transforms : OrderedTransformPipeline
        Preprocessing and normalization transforms for real data.
    label_mapper : LabelMapper
        Mapper object for label-index translation and class names.
    synthetic_ckpt_path : str
        Path to the checkpoint file of the pretrained synthetic model.
    epochs : int, optional
        Number of fine-tuning epochs (default is 10).
    lr : float, optional
        Learning rate for fine-tuning (default is 1e-4).
    device : str, optional
        Device to use ("cuda", "cpu", or "auto") (default is "auto").
    model_name : str, optional
        Name of the architecture to fine-tune (default is "baseline").

    Returns
    -------
    None
    """
    # --- setup paths and configs ---
    os.makedirs(cfg["MODEL"]["LEARNING"]["TRAINER"]["RUN_PATH_FINETUNE"], exist_ok=True)
    ft_cfg = cfg["MODEL"]["LEARNING"]["FINE_TUNE"]
    synthetic_ckpt_path = ft_cfg.get("SYNTHETIC_CKPT_PATH")
    run_path = (
        f"{cfg['MODEL']['LEARNING']['TRAINER']['RUN_PATH_FINETUNE']}/{model_name}"
    )
    config["MODEL"]["LEARNING"]["TRAINER"]["RUN_PATH_FINETUNE"] = run_path
    assert (
        synthetic_ckpt_path
    ), "SYNTHETIC_CKPT_PATH must be set in the config for fine-tuning."

    # --- Setup ---
    batch_size = cfg["MODEL"]["LEARNING"]["TRAINER"]["BATCH_SIZE"]
    num_workers = cfg["MODEL"]["LEARNING"]["TRAINER"]["NUM_WORKERS"]
    use_focal = cfg["MODEL"]["LEARNING"]["LOSS"].get("USE_FOCAL", False)
    use_soft_labels = cfg["MODEL"].get("USE_SOFT_LABELS", True)
    smoothing_eps = cfg["MODEL"].get("SMOOTHING_EPS", 0.0)
    model_arch, _ = cutils.build_model(model_name, cfg["MODEL"])
    num_classes = cfg["MODEL"]["NUM_CLASSES"]
    image_path = ft_cfg.get("REAL_DATA_IMAGE_PATH", None)

    # --- DataModule ---
    fine_tune_dm = dl.FineTuneDataModule(
        parquet_path=real_parquet,
        label_mapper=label_mapper,
        config=cfg,
        rng=rng,
        batch_size=batch_size,
        num_workers=num_workers,
        image_path=image_path,
    )
    fine_tune_dm.setup()

    # --- load pretrained model
    logger.info(f"Loading pretrained model from {synthetic_ckpt_path}")
    class_counts = cutils.calculate_class_counts(fine_tune_dm, num_classes)
    finetune_model = XRFClassifier.load_from_checkpoint(
        synthetic_ckpt_path,
        model=model_arch,
        model_cfg=cfg["MODEL"],
        use_soft_labels=use_soft_labels,
        smoothing_eps=smoothing_eps,
        use_focal=use_focal,
        class_counts=class_counts,
        num_classes_cfg=num_classes,
        label_mapper=label_mapper,
    )

    callbacks = cutils.get_callbacks_list(
        objective=3, cfg=cfg, cb_image=fine_tune_dm.mineral_image
    )
    _, sig_hash = data_module.pipeline_signature()
    run_name, _, _, experiment_name = cutils.setup_mlflow_training(
        finetune_model, transforms_signature=sig_hash, cfg=cfg, is_finetuning=True
    )
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri="file:./mlruns",
        run_name=run_name,
    )
    acc, devs = cutils.trainer_accel(cfg=cfg)

    trainer = cutils.build_trainer(
        cfg=cfg,
        callbacks=callbacks,
        run_path=run_path,
        logger=mlf_logger,
        gradient_clip_val=1.0,
    )

    logger.info("--- Starting Fine-Tuning ---")
    trainer.fit(finetune_model, datamodule=fine_tune_dm)
    logger.info("--- Fine-Tuning Complete ---")

    # Get the path to the best model saved by the ModelCheckpoint callback
    best_fit_ckpt = trainer.checkpoint_callback.best_model_path
    mlflow.log_param("best_finetuned_checkpoint", best_fit_ckpt)
    logger.info(f"Best fine-tuned model saved at: {best_fit_ckpt}")

    return best_fit_ckpt


def train_model(
    label_mapper: LabelMapper,
    model: str = "baseline",
    config_file_path: str = None,
    epochs: int = None,
    data_module: dl.SimpleSpectraLDM = None,
    cfg: dict = None,
    testing_model: bool = False,
    find_learning_rate: bool = False,
) -> str:
    assert config_file_path is not None, "Config file path must be specified."
    assert cfg is not None, "Configuration dictionary must be provided."

    sig_str, sig_hash = data_module.pipeline_signature()
    # -- retrieve mlflow variables
    run_name, iteration, validation_mode, experiment_name = (
        cutils.setup_mlflow_training(model, transforms_signature=sig_hash, cfg=cfg)
    )
    train_summary, val_summary, real_summary = data_module.get_transform_summaries()
    run_path = f"{cfg['MODEL']['LEARNING']['TRAINER']['RUN_PATH_TRAIN']}/{model}"
    # update config for downstream use
    cfg["MODEL"]["ARCH"] = model.__class__.__name__
    cfg["MODEL"]["LEARNING"]["TRAINER"]["RUN_PATH_TRAIN"] = run_path

    with mlflow.start_run(run_name=run_name):
        print("\n")
        logger.info(f"===MLFlow Setup===")
        logger.info(
            f"Experiment: {experiment_name}, Validation mode: {validation_mode}, Run Name: {run_name}"
        )
        try:
            os.makedirs(run_path, exist_ok=True)
            shutil.copy(
                config_file_path,
                os.path.join(run_path, os.path.basename(config_file_path)),
            )
            logger.info(f"Config file {config_file_path} saved to results.")
        except shutil.SameFileError:
            logger.warning(f"Config file {config_file_path} already exists.")

        # build the CNN architecture
        logger.info(f"Training with model: {model}")
        cnn_arch, arch_kwargs = cutils.build_model(model, cfg["MODEL"])
        num_classes = cfg["MODEL"]["NUM_CLASSES"]
        acc, devs = cutils.trainer_accel(cfg=cfg)
        logger.info(f"Accelerator: {acc}, Device: {devs}")

        # get class counts
        class_counts = cutils.calculate_class_counts(data_module, num_classes)
        use_focal = cfg["MODEL"]["LEARNING"]["LOSS"].get("USE_FOCAL", False)
        learning_rate = cfg["MODEL"]["LEARNING"]["OPTIMIZER"].get("LEARNING_RATE", 1e-3)
        if not epochs:
            epochs = cfg["MODEL"]["LEARNING"]["TRAINER"]["MAX_EPOCHS"]
        num_training_steps = len(data_module.train_dataloader()) * epochs
        automatic_uncertainty_weighting = cfg["MODEL"].get(
            "AUTOMATIC_UNCERTAINTY_WEIGHTING", False
        )

        # instantiate the Lightning model
        XRF = XRFClassifier(
            model=cnn_arch,
            label_mapper=label_mapper,
            model_cfg=cfg["MODEL"],
            use_soft_labels=cfg["MODEL"].get("USE_SOFT_LABELS", True),
            use_focal=use_focal,
            smoothing_eps=cfg["MODEL"].get("SMOOTHING_EPS", 0.0),
            class_counts=class_counts,
            learning_rate=learning_rate,
            num_training_steps=num_training_steps,
            automatic_uncertainty_weighting=automatic_uncertainty_weighting,
            use_arcface_head=cfg["MODEL"].get("USE_ARCFACE_HEAD", False),
            warmup_epochs=cfg["MODEL"].get(
                "WARMUP_EPOCHS", 0
            ),  # 0 = no curriculum learning
        )

        # --- logging
        mlflow.log_params(arch_kwargs)
        key_params = cutils.get_key_run_params(cfg)
        mlflow.log_params(key_params)
        mlflow.log_param("iteration", iteration)
        mlflow.log_param("transforms_signature", sig_hash)
        mlflow.log_param(
            "TRAIN_TRANSFORMS", json.dumps({"train_summary": train_summary})
        )
        mlflow.log_param("VAL_TRANSFORMS", json.dumps({"val_summary": val_summary}))
        mlflow.log_param("REAL_TRANSFORMS", json.dumps({"real_summary": real_summary}))
        mlflow.log_dict(train_summary, "train_transforms.json")
        mlflow.log_dict(val_summary, "val_transforms.json")
        mlflow.log_dict(real_summary, "real_transforms.json")
        total_params, trainable_params = cutils.print_model_summary(XRF)
        mlflow.log_param("total_parameters", total_params)
        mlflow.log_param("trainable_parameters", trainable_params)
        mlflow.log_param("num_pure_train_samples", len(data_module.train_pure))
        mlflow.log_param("num_mixed_train_samples", len(data_module.train_mixed))
        mlflow.log_param("num_pure_val_samples", len(data_module.val_pure))
        mlflow.log_param("num_mixed_val_samples", len(data_module.val_mixed))
        mlflow.log_param("num_total_train_samples", len(data_module.dataset_train))
        mlflow.log_param("num_total_val_samples", len(data_module.dataset_val))
        mlflow.log_param("using_arcface", cfg["MODEL"].get("USE_ARCFACE_HEAD"))
        mlflow.log_param(
            "automatic_uncertainty_weighting", automatic_uncertainty_weighting
        )
        mlflow.log_param("warmup_epochs", cfg["MODEL"].get("WARMUP_EPOCHS", 0))

        # -- Callbacks -- #
        # cb_image = data_module.mineral_image # does not work
        cb_image = np.load(cfg["DATA"]["DATASET_KWARGS"][1]["ds_kwargs"]["image_path"])
        logger.info(f"Using image with shape {cb_image.shape} for callbacks.")
        callbacks = cutils.get_callbacks_list(objective=0, cfg=cfg, cb_image=cb_image)

        # -- MLflow logger
        mlf_logger = MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri="file:./mlruns",
            run_name=run_name,
        )

        # -- Pytorch lighning trainer
        if not epochs:
            epochs = cfg["MODEL"]["LEARNING"]["TRAINER"]["MAX_EPOCHS"]

        trainer = cutils.build_trainer(
            cfg=cfg,
            run_path=run_path,
            callbacks=callbacks,
            logger=mlf_logger,
            gradient_clip_val=1.0,
            precision="32-true",
            # accumulate_grad_batches=4,  # gradient accumulation to simulate larger batch sizes
            # profiler=profiler,
        )

        # -- train model
        if find_learning_rate:
            logger.info("Finding optimal learning rate...")
            tuner = pl.tuner.Tuner(trainer)
            lr_finder = tuner.lr_find(XRF, datamodule=data_module)
            suggestion = lr_finder.suggestion()
            logger.info(f"LR Finder suggestion: {lr_finder.suggestion()}")
            if isinstance(XRF.model, ParallelHybridModel):
                logger.info(
                    "Applying suggestion with Anchor and Scale logic for Hybrid Model."
                )
                # get original LRs from config to calculate ratios
                original_lrs = cfg["MODEL"]["LEARNING"]["OPTIMIZER"]["LEARNING_RATES"]
                original_cnn_lr = original_lrs["CNN_BRANCH"]

                # calculate ratios relative to CNN anchor
                transformer_ratio = original_lrs["TRANSFORMER_BRANCH"] / original_cnn_lr
                fusion_ratio = original_lrs["FUSION_HEAD"] / original_cnn_lr

                # calculate new scaled LRs
                new_cnn_lr = suggestion
                new_transformer_lr = suggestion * transformer_ratio
                new_fusion_lr = suggestion * fusion_ratio

                logger.info(
                    f"New differential LRs: CNN={new_cnn_lr:.2e}, Transformer={new_transformer_lr:.2e}, Fusion={new_fusion_lr:.2e}"
                )

                # 4. Update the model's hparams so configure_optimizers can see the new values
                XRF.hparams.learning_rate = new_cnn_lr
                XRF.hparams.scaled_learning_rates = {
                    "CNN_BRANCH": new_cnn_lr,
                    "TRANSFORMER_BRANCH": new_transformer_lr,
                    "FUSION_HEAD": new_fusion_lr,
                }
                mlflow.log_params(XRF.hparams.scaled_learning_rates)
            else:
                XRF.hparams.learning_rate = lr_finder.suggestion()
                mlflow.log_param("lr_finder_suggestion", suggestion)

        logger.info("Start training...\n\n")
        trainer.fit(model=XRF, datamodule=data_module)
        final_loss = trainer.callback_metrics[
            cfg["MODEL"]["LEARNING"]["TRAINER"]["MONITOR_METRIC"]
        ].item()

        best_ckpt = callbacks[
            0
        ].best_model_path  # assumes first callback is ModelCheckpoint
        logger.info(f"Best checkpoint saved at: {best_ckpt}")
        mlflow.log_param("best_checkpoint", best_ckpt)

        # --- TEST --- #
        if testing_model:
            print(f"\n{'-'*30}")
            logger.info("Starting test on best model...")
            data_module.mixture_setup(stage="test")
            trainer.test(ckpt_path="best", datamodule=data_module)
            # # -- log final metric
            mlflow.log_metric(
                cfg["MODEL"]["LEARNING"]["TRAINER"]["MONITOR_METRIC"], final_loss
            )

            # -- save to MLflow
            wrapped = SingleOutput(XRF)
            # create a float32 NumPy input example

            bsz = cfg["MODEL"]["LEARNING"]["TRAINER"]["BATCH_SIZE"]
            # print(f"TRAIN MODEL: BATCH SIZE: {bsz}")
            input_example = np.random.randn(bsz, 1, 4096).astype(np.float32)

            mlflow.pytorch.log_model(wrapped, "model", input_example=input_example)
            logger.info("Training complete.")
            return best_ckpt


def test_model(
    cfg: dict,
    label_mapper: LabelMapper,
    model_name: str,
    rng: np.random.Generator,
) -> None:
    """
    Loads a model and evaluates it on a holdout test set using settings from the config.
    This test run includes all necessary callbacks (e.g., for segmentation).
    """
    logger.info("--- Starting Evaluation Mode ---")

    # 1. Get paths and settings from the config
    test_cfg = cfg["MODEL"]["LEARNING"]["TESTING"]
    trainer_cfg = cfg["MODEL"]["LEARNING"]["TRAINER"]
    ckpt_path = test_cfg.get("CKPT_PATH_TO_TEST")
    parquet_base_path = test_cfg.get("HOLDOUT_TEST_PARQUET")
    test_parquet_path = f"{parquet_base_path}/{config['DATA']['SYNTHETIC_DATASET_NAME']}/dataframe.parquet"
    mappings_filepath = cutils.get_mappings_filepath(cfg)
    image_pathy = cfg["MODEL"]["LEARNING"]["TESTING"].get("IMAGE_PATH_SEGMAP_ORIG")

    assert ckpt_path, "CKPT_PATH_TO_TEST must be defined in the config."
    assert test_parquet_path, "HOLDOUT_TEST_PARQUET must be defined in the config."
    assert image_pathy, "IMAGE_PATH_SEGMAP_ORIG must be defined in the config."

    logger.info(f"Loading model from checkpoint: {ckpt_path}")
    logger.info(f"Using test data from: {test_parquet_path}")

    # configure logger to use experiment name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"Evaluation_{model_name}_{Path(ckpt_path).stem}_{timestamp}"
    mlf_logger = MLFlowLogger(
        experiment_name="_EVALUATION",  # A separate experiment for clarity
        run_name=run_name,
        tracking_uri="file:./mlruns",
    )

    # build test pipeline
    test_pipeline, _ = build_transforms_pipeline(
        rng=rng,
        stage="finetune_val",
        cfg=cfg,
        num_classes=label_mapper.get_num_classes(),
    )
    test_df = pd.read_parquet(test_parquet_path)
    image_path = cfg["MODEL"]["LEARNING"]["TESTING"].get("REAL_DATA_IMAGE_PATH")
    test_ds = dl.RealSpectraDataset(
        test_df, test_pipeline, label_mapper, image_path=image_pathy
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg["MODEL"]["LEARNING"]["TRAINER"]["BATCH_SIZE"],
        num_workers=cfg["MODEL"]["LEARNING"]["TRAINER"]["NUM_WORKERS"],
    )

    # model loading
    model_arch, _ = cutils.build_model(model_name, cfg["MODEL"])
    num_classes = config["MODEL"]["NUM_CLASSES"]
    dummy_class_counts = torch.ones(
        num_classes
    )  # for class counts (not used in testing, but required by the model)

    trained_model = XRFClassifier.load_from_checkpoint(
        ckpt_path,
        model=model_arch,
        model_cfg=cfg["MODEL"],
        mappings_file=mappings_filepath,
        use_soft_labels=cfg["MODEL"].get("USE_SOFT_LABELS", True),
        num_classes_cfg=num_classes,
        class_counts=dummy_class_counts,
        use_focal=False,
        smoothing_eps=cfg["MODEL"].get("SMOOTHING_EPS", 0.1),
        label_mapper=label_mapper,
    )

    # get testing callbacks
    test_callbacks = cutils.get_callbacks_list(
        objective=4, cfg=cfg, cb_image=test_ds.image
    )

    # create trainer
    trainer = cutils.build_trainer(
        cfg=cfg,
        callbacks=test_callbacks,
        logger=mlf_logger,
        run_path=f"{trainer_cfg['RUN_PATH_TRAIN']}/{model_name}_eval",
        precision="32-true",
    )
    logger.info("Running evaluation with all test-time callbacks...")
    trainer.test(model=trained_model, dataloaders=test_loader)
    logger.info("--- Evaluation Complete ---")


class SingleOutput(torch.nn.Module):
    def __init__(self, wrapped_model):
        super().__init__()
        self.wrapped = wrapped_model

    def forward(self, x):
        feats, logits_hard, logits_soft = self.wrapped(x)
        return logits_hard


def run_hpo(args, config):
    hpo_cfg = config["MODEL"]["LEARNING"]["HPO"]
    hpo_kwargs = {
        "direction": hpo_cfg.get("DIRECTION", "maximize"),
        "storage": hpo_cfg.get("STORAGE_URL", "sqlite:///hyperparam.db"),
        "load_if_exists": hpo_cfg.get("LOAD_IF_EXISTS", True),
    }
    hpo_kwargs["sampler"] = optuna.samplers.TPESampler()
    hpo_kwargs["pruner"] = optuna.pruners.MedianPruner()
    hpo_kwargs["study_name"] = f"{model}_aug_search_phase1"

    if not args.phase2:
        ### PHASE 1: Augmentation Parameters Optimization
        run_path = config["MODEL"]["LEARNING"]["TRAINER"]["RUN_PATH_HPO1"]
        logger.info(
            f"[Running Phase 1: Augmentation Parameters Optimization] Default path: {run_path}"
        )
        study1 = optuna.create_study(**hpo_kwargs)
        study1.optimize(
            lambda trial: objective_phase1(
                trial, modelname=model, cfg=config, run_path=run_path
            ),
            n_trials=hpo_cfg.get(
                "HYPERPARAMETER_N_TRIALS_OBJ1",
                config["MODEL"]["LEARNING"]["HPO"]["HYPERPARAMETER_N_TRIALS_OBJ1"],
            ),
        )
        best_aug = study1.best_params
        logger.info(f"Best Augmentation Parameters: {best_aug}")
    else:
        ### PHASE 2: Model Hyperparameters Optimization
        # TODO: Use the best augmentation parameters (in config.yml) from phase 1
        hpo_kwargs["study_name"] = f"{model}_model_search_phase2"
        run_path = config["MODEL"]["LEARNING"]["TRAINER"]["RUN_PATH_HPO2"]
        logger.info(
            f"[Running Phase 2: Model Hyperparameters Optimization] Default path: {run_path}"
        )
        study2 = optuna.create_study(**hpo_kwargs)
        study2.optimize(
            lambda trial: objective_phase2(trial, modelname=model, cfg=config),
            n_trials=hpo_cfg.get(
                "HYPERPARAMETER_N_TRIALS_OBJ2",
                config["MODEL"]["LEARNING"]["HPO"]["HYPERPARAMETER_N_TRIALS_OBJ2"],
            ),
        )


if __name__ == "__main__":
    model_choices = [
        "attn-cnn",
        "lstm",
        "baseline",
        "transformer",
        "sanity-check",
        "simple",
    ]
    parser = argparse.ArgumentParser(
        description="Run ML training.", epilog="Bruker Nano Analytics"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=model_choices,
        default="baseline",
        help="Select the model architecture.",
    )
    parser.add_argument(
        "--optional_transforms",
        type=int,
        default=0,
        help="Number of optional transformations, randomly drawn. Capped at available number of optioanl transformations (2).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to train the model.",
    )
    parser.add_argument(
        "--mix",
        action="store_true",
        help="Enable mix augmentation. Default is False.",
    )
    parser.add_argument(
        "--hyperparam",
        action="store_true",
        help="Run Optuna hyperparameter optimization. Default is False.",
    )
    parser.add_argument(
        "--phase2",
        action="store_true",
        help="Run Phase 2 of hyperparameter optimization (model hyperparameters). Default is False (Phase 1).",
    )
    parser.add_argument(
        "--fine_tune",
        action="store_true",
        help="Fine-tune the model on real data. Default is False (train from scratch).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode. Requires ckpt-path and test-parquet. Default is False.",
    )
    args = parser.parse_args()

    config_file_path = "../configs/config.yml"
    config = load_config(config_file_path)
    config = copy.deepcopy(config)  # truly non-global now

    model = args.model if args.model else "baseline"
    num_optional_transforms = (
        args.optional_transforms if args.optional_transforms else 0
    )
    epochs = args.epochs if args.epochs else None
    mix_enabled = args.mix if args.mix else False

    # mixing augmentation setup
    config["DATA"]["MIX_AUGMENT"]["AUGMENT_FRACTION"] = (
        config["DATA"]["MIX_AUGMENT"].get("AUGMENT_FRACTION", 1.0)
        if mix_enabled
        else 0.0
    )
    config["MODEL"]["LEARNING"]["TRAINER"]["MAX_EPOCHS"] = (
        args.epochs
        if args.epochs
        else config["MODEL"]["LEARNING"]["TRAINER"]["MAX_EPOCHS"]
    )
    # -- Set seed
    seed = config["MODEL"]["LEARNING"]["TRAINER"].get("TRAINING_SEED", 42)
    pl.seed_everything(seed, workers=True, verbose=False)
    logger.info(f"Global seed set to: {seed}.")

    # -- Label mapper wrapper dapper zapper
    label_mapper = cutils.get_label_mapper(cfg=config)
    num_classes = label_mapper.get_num_classes()
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Batch Size: {config['MODEL']['LEARNING']['TRAINER']['BATCH_SIZE']}")
    logger.info(
        f"Number of workers: {config['MODEL']['LEARNING']['TRAINER']['NUM_WORKERS']}"
    )

    if mix_enabled:
        logger.info("Mixture Augmentation enabled.")
        logger.info(
            f"Augmentation Fraction: {config['DATA']['MIX_AUGMENT']['AUGMENT_FRACTION']}, with weight step: {config['DATA']['MIX_AUGMENT'].get('WEIGHT_STEP')}"
        )
    else:
        logger.info("Mixture Augmentation disabled.")

    # -- update config with correct number of classes
    cutils.set_cfg_num_classes(cfg=config, num_classes=num_classes)

    # -- update conf with minimal transforms and ds class settings
    cutils.set_cfg_dataset(cfg=config)

    # -- set seed
    rng0 = np.random.default_rng(
        config["MODEL"]["LEARNING"]["TRAINER"]["TRAINING_SEED"]
    )

    # check if hyperparam mode or not
    if args.hyperparam:
        logger.info("--- Running in HPO mode. ---\n")
        run_hpo(args, config)

    elif args.fine_tune:
        logger.info("--- Running in Fine-Tuning mode. ---\n")
        ft_cfg = config["MODEL"]["LEARNING"]["FINE_TUNE"]

        # Fine-tuning phase
        tf_kwargs = {
            "real_parquet": ft_cfg.get("REAL_DATA_PARQUET"),
            "label_mapper": label_mapper,
            "rng": rng0,
            "cfg": config,
            "model_name": model,
            "epochs": config["MODEL"]["LEARNING"]["TRAINER"].get("MAX_EPOCHS"),
            "lr": ft_cfg.get("LEARNING_RATE_TF", 1e-4),
            "device": config["MODEL"]["LEARNING"]["TRAINER"].get("DEVICE", "cuda"),
        }
        fine_tune_model(**tf_kwargs)

    elif args.test:
        logger.info("--- Running in Test mode. ---")
        test_model(
            cfg=config,
            label_mapper=label_mapper,
            model_name=model,
            rng=rng0,
        )

    else:
        logger.info("--- Running in Training mode. ---\n")
        data_module = cutils.create_data_module(cfg=config, with_test_dataloader=True)
        # train with final pipeline
        best_synth_ckpt = train_model(
            model=model,
            epochs=epochs,
            data_module=data_module,
            config_file_path=config_file_path,
            cfg=config,
            testing_model=False,
            find_learning_rate=False,
            label_mapper=label_mapper,
        )
