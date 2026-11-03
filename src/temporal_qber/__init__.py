"""
Temporal-QBER Attack Detection for BB84 QKD using ML
"""

__version__ = "0.1.0"

# Core modules
from .config import (
    QAD_CLASSES,
    BB84_CLASSES,
    QAD_FEATURE_COLS,
    TEMPORAL_WINDOW_SIZE,
    MIN_SIFTED_BITS,
    AVAILABLE_MODELS,
    TrainConfig,
    PipelineConfig,
    BB84SimConfig,
    BB84SessionConfig,
    DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
)

from .dataset import (
    load_qad_dataset,
    prepare_features_targets,
    split_data,
    fit_scaler,
    transform_data,
    save_preprocessing,
    load_preprocessing,
    get_class_distribution,
    print_dataset_info,
)

from .train import (
    create_models,
    train_model,
    cross_validate_model,
    train_all_models,
    save_models,
    load_model,
    predict,
    predict_proba,
)

from .evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_per_class_f1,
    evaluate_baselines,
    cusum_detector,
    save_results,
    print_summary,
)

from .explain import (
    explain_model_shap,
    feature_group_ablation,
    define_qad_feature_groups,
    save_shap_results,
)

from .predict import (
    load_artifacts,
    preprocess_sample,
    predict_single,
    predict_batch,
    demo_prediction,
)

# BB84 Simulator modules
from .simulator import (
    Alice,
    Bob,
    Basis,
    Bit,
    BB84_STATES,
    STATE_LABELS,
    QuantumChannel,
    IdealChannel,
    DepolarizingChannel,
    PhaseFlipChannel,
    AmplitudeDampingChannel,
    create_channel,
    BB84Protocol,
    BB84Result,
    run_bb84_simulation,
    SessionConfig,
    SessionResult,
    run_session,
)

# Dataset modules
from .datasets import (
    generate_bb84_dataset,
    load_bb84_dataset,
    dataset_to_features_labels,
    create_bb84_datasets,
)

__all__ = [
    # Config
    "QAD_CLASSES",
    "BB84_CLASSES",
    "QAD_FEATURE_COLS",
    "TEMPORAL_WINDOW_SIZE",
    "MIN_SIFTED_BITS",
    "AVAILABLE_MODELS",
    "TrainConfig",
    "PipelineConfig",
    "BB84SimConfig",
    "BB84SessionConfig",
    "DATA_DIR",
    "MODELS_DIR",
    "RESULTS_DIR",
    # Dataset
    "load_qad_dataset",
    "prepare_features_targets",
    "split_data",
    "fit_scaler",
    "transform_data",
    "save_preprocessing",
    "load_preprocessing",
    "get_class_distribution",
    "print_dataset_info",
    # Train
    "create_models",
    "train_model",
    "cross_validate_model",
    "train_all_models",
    "save_models",
    "load_model",
    "predict",
    "predict_proba",
    # Evaluate
    "evaluate_model",
    "plot_confusion_matrix",
    "plot_per_class_f1",
    "evaluate_baselines",
    "cusum_detector",
    "save_results",
    "print_summary",
    # Explain
    "explain_model_shap",
    "feature_group_ablation",
    "define_qad_feature_groups",
    "save_shap_results",
    # Predict
    "load_artifacts",
    "preprocess_sample",
    "predict_single",
    "predict_batch",
    "demo_prediction",
    # BB84 Simulator
    "Alice",
    "Bob",
    "Basis",
    "Bit",
    "BB84_STATES",
    "STATE_LABELS",
    "QuantumChannel",
    "IdealChannel",
    "DepolarizingChannel",
    "PhaseFlipChannel",
    "AmplitudeDampingChannel",
    "create_channel",
    "BB84Protocol",
    "BB84Result",
    "run_bb84_simulation",
    "SessionConfig",
    "SessionResult",
    "run_session",
    # Datasets
    "generate_bb84_dataset",
    "load_bb84_dataset",
    "dataset_to_features_labels",
    "create_bb84_datasets",
]