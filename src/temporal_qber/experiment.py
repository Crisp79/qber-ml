"""
Experiment tracking for Temporal-QBER pipeline using JSON.
"""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    experiment_id: str
    timestamp: str
    git_commit: Optional[str] = None
    dataset_name: str = "QAD"
    dataset_variant: str = "balanced"
    n_samples: Optional[int] = None
    n_features: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    train_test_split: Dict[str, Any] = field(default_factory=dict)
    cv_config: Dict[str, Any] = field(default_factory=dict)
    random_seeds: Dict[str, int] = field(default_factory=dict)
    model_hyperparameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    models: List[str] = field(default_factory=list)


@dataclass
class ModelResult:
    """Results for a single model."""
    model_name: str
    train_time: float = 0.0
    eval_time: float = 0.0
    test_accuracy: float = 0.0
    cv_results: Dict[str, Any] = field(default_factory=dict)
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    model_path: str = ""
    plots: List[str] = field(default_factory=list)
    shap_results: Optional[Dict[str, Any]] = None
    ablation_results: Optional[Dict[str, Any]] = None
    status: str = "completed"
    error: Optional[str] = None


@dataclass
class ExperimentResult:
    """Complete experiment results."""
    config: ExperimentConfig
    models: Dict[str, ModelResult] = field(default_factory=dict)
    baselines: Dict[str, Any] = field(default_factory=dict)
    total_duration: float = 0.0
    status: str = "completed"
    error: Optional[str] = None
    completed_at: Optional[str] = None


class ExperimentTracker:
    """Track and save experiment metadata and results."""
    
    def __init__(
        self,
        experiment_id: Optional[str] = None,
        output_dir: str = "results/experiments",
        logger=None
    ):
        self.experiment_id = experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.start_time = time.time()
        self.result: Optional[ExperimentResult] = None
        
        # Get git commit if available
        self.git_commit = self._get_git_commit()
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except Exception:
            pass
        return None
    
    def start_experiment(self, config: ExperimentConfig):
        """Initialize experiment tracking."""
        config.experiment_id = self.experiment_id
        config.timestamp = datetime.now().isoformat()
        config.git_commit = self.git_commit
        
        self.result = ExperimentResult(config=config)
        
        if self.logger:
            self.logger.info(f"Experiment started: {self.experiment_id}")
    
    def record_model_start(self, model_name: str):
        """Record model training start."""
        if self.result:
            self.result.models[model_name] = ModelResult(model_name=model_name)
    
    def record_model_result(
        self,
        model_name: str,
        train_time: float,
        eval_time: float,
        test_accuracy: float,
        cv_results: Dict[str, Any],
        fold_metrics: List[Dict[str, float]],
        aggregate_metrics: Dict[str, float],
        model_path: str = "",
        plots: List[str] = None,
        shap_results: Dict[str, Any] = None,
        ablation_results: Dict[str, Any] = None,
        status: str = "completed",
        error: str = None
    ):
        """Record complete model results."""
        if self.result:
            # Create model result if not exists (e.g., if record_model_start wasn't called)
            if model_name not in self.result.models:
                self.result.models[model_name] = ModelResult(model_name=model_name)
            
            model_result = self.result.models[model_name]
            model_result.train_time = train_time
            model_result.eval_time = eval_time
            model_result.test_accuracy = test_accuracy
            model_result.cv_results = cv_results
            model_result.fold_metrics = fold_metrics
            model_result.aggregate_metrics = aggregate_metrics
            model_result.model_path = model_path
            model_result.plots = plots or []
            model_result.shap_results = shap_results
            model_result.ablation_results = ablation_results
            model_result.status = status
            model_result.error = error
            
            # Save incrementally
            self.save_incremental()
            
            if self.logger:
                self.logger.info(f"Model {model_name} results recorded")
    
    def record_baselines(self, baselines: Dict[str, Any]):
        """Record baseline results."""
        if self.result:
            self.result.baselines = baselines
    
    def finalize(self, status: str = "completed", error: str = None):
        """Finalize experiment and save results."""
        if self.result:
            self.result.total_duration = time.time() - self.start_time
            self.result.status = status
            self.result.error = error
            self.result.completed_at = datetime.now().isoformat()
            
            self.save_final()
            
            if self.logger:
                self.logger.info(f"Experiment {self.experiment_id} {status} in {self.result.total_duration:.1f}s")
    
    def save_incremental(self):
        """Save experiment results incrementally (after each model)."""
        if not self.result:
            return
        
        # Convert to serializable dict, skipping non-serializable fields
        data = self._to_dict_safe(self.result)
        
        filepath = self.output_dir / f"{self.experiment_id}_incremental.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=self._json_serializer)
    
    def save_final(self):
        """Save final experiment results."""
        if not self.result:
            return
        
        data = self._to_dict_safe(self.result)
        
        filepath = self.output_dir / f"{self.experiment_id}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=self._json_serializer)
        
        # Also save a latest symlink/copy
        latest_path = self.output_dir / "latest.json"
        with open(latest_path, 'w') as f:
            json.dump(data, f, indent=2, default=self._json_serializer)
    
    def _to_dict_safe(self, obj: Any) -> Any:
        """Convert dataclass to dict recursively, skipping non-serializable fields."""
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for field_name in obj.__dataclass_fields__:
                value = getattr(obj, field_name)
                # Skip known non-serializable fields
                if field_name in ('shap_results', 'ablation_results') and value is not None:
                    # Store only summary info for these
                    if field_name == 'shap_results' and isinstance(value, dict):
                        result[field_name] = {
                            'top_features': value.get('top_features', [])[:10],
                            'n_features': len(value.get('global_importance', [])),
                        }
                    elif field_name == 'ablation_results' and isinstance(value, dict):
                        result[field_name] = {
                            'baseline_accuracy': value.get('baseline_accuracy'),
                            'baseline_f1': value.get('baseline_f1'),
                            'groups': list(value.get('ablation_results', {}).keys()),
                        }
                    else:
                        result[field_name] = str(type(value))
                else:
                    result[field_name] = self._to_dict_safe(value)
            return result
        elif isinstance(obj, dict):
            return {k: self._to_dict_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._to_dict_safe(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _to_dict(self, obj: Any) -> Any:
        """Convert dataclass to dict recursively."""
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for field_name in obj.__dataclass_fields__:
                value = getattr(obj, field_name)
                result[field_name] = self._to_dict(value)
            return result
        elif isinstance(obj, dict):
            return {k: self._to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._to_dict(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dataclass_fields__'):
            return self._to_dict(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def get_experiment_path(self) -> Path:
        """Get path to experiment results file."""
        return self.output_dir / f"{self.experiment_id}.json"


def load_experiment(experiment_id: str, experiments_dir: str = "results/experiments") -> Optional[ExperimentResult]:
    """Load experiment results from file."""
    filepath = Path(experiments_dir) / f"{experiment_id}.json"
    if not filepath.exists():
        return None
    
    with open(filepath) as f:
        data = json.load(f)
    
    # Reconstruct objects (simplified - just return dict)
    return data


def list_experiments(experiments_dir: str = "results/experiments") -> List[Dict[str, Any]]:
    """List all experiments with summary info."""
    exp_dir = Path(experiments_dir)
    if not exp_dir.exists():
        return []
    
    experiments = []
    for f in exp_dir.glob("*.json"):
        if f.name == "latest.json":
            continue
        try:
            with open(f) as fp:
                data = json.load(fp)
            experiments.append({
                'experiment_id': data.get('config', {}).get('experiment_id', f.stem),
                'timestamp': data.get('config', {}).get('timestamp', ''),
                'status': data.get('status', 'unknown'),
                'duration': data.get('total_duration', 0),
                'models': list(data.get('models', {}).keys()),
            })
        except Exception:
            continue
    
    return sorted(experiments, key=lambda x: x['timestamp'], reverse=True)


def create_experiment_config_from_args(args, n_features: int, class_dist: Dict[int, int]) -> ExperimentConfig:
    """Create experiment config from CLI args and dataset info."""
    from temporal_qber.config import QAD_CLASSES
    
    return ExperimentConfig(
        experiment_id="",  # Will be set by tracker
        timestamp="",      # Will be set by tracker
        dataset_variant="balanced" if args.balanced else "unbalanced",
        n_samples=args.n_samples,
        n_features=n_features,
        class_distribution={
            QAD_CLASSES[k] if k < len(QAD_CLASSES) else str(k): v 
            for k, v in class_dist.items()
        },
        train_test_split={
            'test_size': args.test_size if hasattr(args, 'test_size') else 0.2,
            'stratified': True
        },
        cv_config={
            'folds': args.cv_folds if hasattr(args, 'cv_folds') else 5,
            'shuffle': True
        },
        random_seeds={
            'data_split': 42,
            'models': 42,
            'cv': 42
        },
        model_hyperparameters={}  # Will be filled by train module
    )