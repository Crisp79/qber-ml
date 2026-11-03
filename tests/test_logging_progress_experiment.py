"""
Tests for logging, progress bars, and experiment tracking modules.
"""

import sys
import json
import tempfile
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from temporal_qber.logging_config import (
    setup_logging, get_logger, log_experiment_config
)
from temporal_qber.progress import (
    ProgressManager, TrainingProgressTracker, create_xgb_callback, track_training_time
)
from temporal_qber.experiment import (
    ExperimentTracker, ExperimentConfig, ModelResult, ExperimentResult,
    list_experiments
)
from temporal_qber.config import TrainConfig


class TestLogging:
    """Test logging configuration."""
    
    def test_setup_logging_creates_file(self, tmp_path):
        """Test that setup_logging creates log file."""
        log_dir = tmp_path / "logs"
        logger = setup_logging(log_level=logging.DEBUG, log_dir=str(log_dir), experiment_id="test_exp")
        
        logger.info("Test message")
        logger.debug("Debug message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Check log file exists
        log_files = list(log_dir.glob("experiment_test_exp.log"))
        assert len(log_files) == 1
        
        # Check content
        content = log_files[0].read_text()
        assert "Test message" in content
        assert "Debug message" in content
        assert "Warning message" in content
        assert "Error message" in content
    
    def test_no_duplicate_handlers(self, tmp_path):
        """Test that calling setup_logging twice doesn't create duplicate handlers."""
        log_dir = tmp_path / "logs"
        
        logger1 = setup_logging(log_dir=str(log_dir), experiment_id="test1")
        handler_count_1 = len(logger1.handlers)
        
        logger2 = setup_logging(log_dir=str(log_dir), experiment_id="test2")
        handler_count_2 = len(logger2.handlers)
        
        assert handler_count_1 == handler_count_2
    
    def test_get_logger_returns_same_instance(self):
        """Test that get_logger returns the same logger instance."""
        logger1 = get_logger()
        logger2 = get_logger()
        assert logger1 is logger2
    
    def test_log_experiment_config(self, tmp_path):
        """Test logging experiment configuration."""
        log_dir = tmp_path / "logs"
        logger = setup_logging(log_level=logging.INFO, log_dir=str(log_dir), experiment_id="test_config")
        
        config = {
            'model': 'xgboost',
            'n_estimators': 200,
            'learning_rate': 0.1,
            'dataset': 'QAD'
        }
        log_experiment_config(logger, config)
        
        log_files = list(log_dir.glob("experiment_test_config.log"))
        content = log_files[0].read_text()
        assert "EXPERIMENT CONFIGURATION" in content
        assert "model: xgboost" in content
        assert "n_estimators: 200" in content


class TestProgressManager:
    """Test progress bar manager."""
    
    def test_progress_manager_disabled(self):
        """Test progress manager when disabled."""
        pm = ProgressManager(enabled=False)
        
        # Should not raise
        with pm.progress(total=10, desc="Test") as pbar:
            for i in range(10):
                pass
        
        # Log should work
        pm.log("Test message")
    
    def test_progress_manager_enabled(self):
        """Test progress manager when enabled."""
        pm = ProgressManager(enabled=True)
        
        with pm.progress(total=5, desc="Test", unit="step") as pbar:
            for i in range(5):
                pbar.update(1)
        
        pm.close_all()
    
    def test_stage_progress(self):
        """Test stage progress bar."""
        pm = ProgressManager(enabled=True)
        
        with pm.stage("Test Stage", total=3) as pbar:
            for i in range(3):
                pm.update_stage(pbar, 1, f"step {i}")
        
        pm.close_all()


class TestTrainingProgressTracker:
    """Test training progress tracker."""
    
    def test_tracker_records_fold_times(self, tmp_path):
        """Test that tracker records fold times."""
        log_dir = tmp_path / "logs"
        logger = setup_logging(log_level=logging.DEBUG, log_dir=str(log_dir), experiment_id="test_tracker", disable_console=True)
        
        pm = ProgressManager(enabled=False, logger=logger)
        tracker = TrainingProgressTracker("test_model", 3, logger=logger, progress_manager=pm)
        
        tracker.start_fold(1)
        import time
        time.sleep(0.01)
        tracker.end_fold({'accuracy': 0.9, 'macro_f1': 0.85})
        
        tracker.start_fold(2)
        time.sleep(0.01)
        tracker.end_fold({'accuracy': 0.92, 'macro_f1': 0.88})
        
        tracker.start_fold(3)
        time.sleep(0.01)
        tracker.end_fold({'accuracy': 0.91, 'macro_f1': 0.87})
        
        summary = tracker.finish()
        
        assert len(summary['fold_times']) == 3
        assert summary['total_time'] > 0
        assert summary['avg_time'] > 0


class TestExperimentTracker:
    """Test experiment tracking."""
    
    def test_experiment_tracker_creates_files(self, tmp_path):
        """Test that experiment tracker creates JSON files."""
        exp_dir = tmp_path / "experiments"
        tracker = ExperimentTracker(experiment_id="test_exp", output_dir=str(exp_dir))
        
        config = ExperimentConfig(
            experiment_id="test_exp",
            timestamp="2024-01-01T00:00:00",
            dataset_name="QAD",
            dataset_variant="balanced",
            n_samples=1000,
            n_features=45,
            class_distribution={"clean": 300, "swap_injection": 100},
            train_test_split={"test_size": 0.2},
            cv_config={"folds": 5},
            random_seeds={"data_split": 42},
            model_hyperparameters={"xgboost": {"n_estimators": 200}}
        )
        
        tracker.start_experiment(config)
        
        tracker.record_model_result(
            model_name="xgboost",
            train_time=10.5,
            eval_time=0.5,
            test_accuracy=0.95,
            cv_results={"cv_accuracy_mean": 0.94, "cv_accuracy_std": 0.01},
            fold_metrics=[{"fold": 1, "accuracy": 0.94}, {"fold": 2, "accuracy": 0.95}],
            aggregate_metrics={"test_accuracy": 0.95, "macro_f1": 0.94},
            model_path="models/xgboost.joblib",
            plots=["results/xgboost_cm.png"],
            status="completed"
        )
        
        tracker.finalize(status="completed")
        
        # Check files created
        exp_file = exp_dir / "test_exp.json"
        assert exp_file.exists()
        
        latest_file = exp_dir / "latest.json"
        assert latest_file.exists()
        
        # Check content
        with open(exp_file) as f:
            data = json.load(f)
        
        assert data['config']['experiment_id'] == "test_exp"
        assert data['config']['dataset_name'] == "QAD"
        assert 'xgboost' in data['models']
        assert data['models']['xgboost']['test_accuracy'] == 0.95
        assert data['status'] == "completed"
        assert data['total_duration'] > 0
    
    def test_incremental_save(self, tmp_path):
        """Test incremental saving after each model."""
        exp_dir = tmp_path / "experiments"
        tracker = ExperimentTracker(experiment_id="test_incremental", output_dir=str(exp_dir))
        
        config = ExperimentConfig(
            experiment_id="test_incremental",
            timestamp="2024-01-01T00:00:00",
            dataset_name="QAD",
            dataset_variant="balanced",
            n_samples=1000,
            n_features=45,
            class_distribution={},
            train_test_split={},
            cv_config={},
            random_seeds={},
            model_hyperparameters={}
        )
        
        tracker.start_experiment(config)
        
        # Record first model
        tracker.record_model_result(
            model_name="rf",
            train_time=5.0,
            eval_time=0.3,
            test_accuracy=0.93,
            cv_results={},
            fold_metrics=[],
            aggregate_metrics={},
            model_path="models/rf.joblib",
            plots=[],
            status="completed"
        )
        
        # Check incremental file exists
        inc_file = exp_dir / "test_incremental_incremental.json"
        assert inc_file.exists()
        
        with open(inc_file) as f:
            data = json.load(f)
        
        assert 'rf' in data['models']
        assert data['models']['rf']['test_accuracy'] == 0.93
    
    def test_list_experiments(self, tmp_path):
        """Test listing experiments."""
        exp_dir = tmp_path / "experiments"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a couple of experiment files
        exp1 = {
            'config': {'experiment_id': 'exp1', 'timestamp': '2024-01-01T00:00:00'},
            'status': 'completed',
            'total_duration': 100,
            'models': {'rf': {}, 'xgb': {}}
        }
        exp2 = {
            'config': {'experiment_id': 'exp2', 'timestamp': '2024-01-02T00:00:00'},
            'status': 'failed',
            'total_duration': 50,
            'models': {'svm': {}}
        }
        
        with open(exp_dir / "exp1.json", 'w') as f:
            json.dump(exp1, f)
        with open(exp_dir / "exp2.json", 'w') as f:
            json.dump(exp2, f)
        
        experiments = list_experiments(str(exp_dir))
        
        assert len(experiments) == 2
        # Should be sorted by timestamp descending
        assert experiments[0]['experiment_id'] == 'exp2'
        assert experiments[1]['experiment_id'] == 'exp1'


class TestConfig:
    """Test configuration updates."""
    
    def test_train_config_to_dict(self):
        """Test TrainConfig.to_dict() method."""
        config = TrainConfig(
            test_size=0.2,
            cv_folds=5,
            rf_n_estimators=200,
            xgb_n_estimators=200,
            xgb_learning_rate=0.1
        )
        
        d = config.to_dict()
        
        assert d['test_size'] == 0.2
        assert d['cv_folds'] == 5
        assert d['rf_n_estimators'] == 200
        assert d['xgb_n_estimators'] == 200
        assert d['xgb_learning_rate'] == 0.1
        assert d['show_progress'] == True
        assert d['log_level'] == 'INFO'
    
    def test_pipeline_config_to_dict(self):
        """Test PipelineConfig.to_dict() method."""
        from temporal_qber.config import PipelineConfig, TrainConfig
        
        pipeline_config = PipelineConfig(
            train=TrainConfig(),
            models=['random_forest', 'xgboost'],
            do_shap=True,
            do_ablation=False
        )
        
        d = pipeline_config.to_dict()
        
        assert 'train' in d
        assert d['models'] == ['random_forest', 'xgboost']
        assert d['do_shap'] == True
        assert d['do_ablation'] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])