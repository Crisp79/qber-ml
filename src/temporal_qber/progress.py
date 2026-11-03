"""
Progress bar utilities using tqdm for Temporal-QBER pipeline.
"""

import sys
import time
from contextlib import contextmanager
from typing import Optional, Iterator, Any
from tqdm import tqdm


class ProgressManager:
    """Manages progress bars for the pipeline."""
    
    def __init__(self, enabled: bool = True, logger=None):
        self.enabled = enabled
        self.logger = logger
        self._active_bars = []
    
    @contextmanager
    def progress(
        self,
        iterable=None,
        total: Optional[int] = None,
        desc: str = "Processing",
        unit: str = "it",
        leave: bool = True,
        **kwargs
    ) -> Iterator[tqdm]:
        """Create a progress bar context manager."""
        if not self.enabled:
            # Return a dummy iterator that yields from iterable
            if iterable is not None:
                yield iterable
            else:
                yield range(total or 0)
            return
        
        bar = tqdm(
            iterable=iterable,
            total=total,
            desc=desc,
            unit=unit,
            leave=leave,
            dynamic_ncols=True,
            file=sys.stdout,
            **kwargs
        )
        self._active_bars.append(bar)
        try:
            yield bar
        finally:
            bar.close()
            if bar in self._active_bars:
                self._active_bars.remove(bar)
    
    @contextmanager
    def stage(self, name: str, total: int = 1, unit: str = "stage") -> Iterator[tqdm]:
        """Create a stage-level progress bar (indeterminate if total=1)."""
        # For single-stage operations, use a spinner-style bar without percentage
        if total <= 1:
            with self.progress(total=total, desc=name, unit=unit, leave=True, 
                             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as bar:
                yield bar
        else:
            with self.progress(total=total, desc=name, unit=unit, leave=True) as bar:
                yield bar
    
    def update_stage(self, bar: tqdm, increment: int = 1, msg: str = ""):
        """Update a stage progress bar."""
        if self.enabled and bar:
            bar.update(increment)
            if msg:
                bar.set_postfix_str(msg)
    
    def log(self, msg: str):
        """Log a message that works with tqdm."""
        if self.logger:
            self.logger.info(msg)
        elif self.enabled:
            tqdm.write(msg)
        else:
            print(msg)
    
    def close_all(self):
        """Close all active progress bars."""
        for bar in self._active_bars:
            bar.close()
        self._active_bars.clear()


def create_progress_manager(enabled: bool = True, logger=None) -> ProgressManager:
    """Create a progress manager instance."""
    return ProgressManager(enabled=enabled, logger=logger)


class XGBoostProgressCallback:
    """XGBoost training callback for progress reporting (compatible with XGBoost >= 3.0)."""
    
    def __init__(self, progress_bar: tqdm, total_estimators: int):
        self.progress_bar = progress_bar
        self.total_estimators = total_estimators
        self.current_iteration = 0
        self.start_time = time.time()
    
    def __call__(self, env):
        """XGBoost callback function (for xgb.train API)."""
        if self.progress_bar is None:
            return
        
        self.current_iteration = env.iteration + 1
        elapsed = time.time() - self.start_time
        
        # Calculate ETA
        if self.current_iteration > 0:
            rate = self.current_iteration / elapsed
            remaining = (self.total_estimators - self.current_iteration) / rate if rate > 0 else 0
            eta_str = f"{remaining:.0f}s" if remaining > 0 else "done"
        else:
            eta_str = "calculating..."
        
        self.progress_bar.n = self.current_iteration
        self.progress_bar.set_postfix_str(f"iter {self.current_iteration}/{self.total_estimators} ETA: {eta_str}")
        self.progress_bar.refresh()
    
    def after_iteration(self, model, epoch, evals_log):
        """Callback for XGBoost sklearn API (after each iteration)."""
        if self.progress_bar is None:
            return False
        
        self.current_iteration = epoch + 1
        elapsed = time.time() - self.start_time
        
        # Calculate ETA
        if self.current_iteration > 0:
            rate = self.current_iteration / elapsed
            remaining = (self.total_estimators - self.current_iteration) / rate if rate > 0 else 0
            eta_str = f"{remaining:.0f}s" if remaining > 0 else "done"
        else:
            eta_str = "calculating..."
        
        self.progress_bar.n = self.current_iteration
        self.progress_bar.set_postfix_str(f"iter {self.current_iteration}/{self.total_estimators} ETA: {eta_str}")
        self.progress_bar.refresh()
        return False  # Continue training


def create_xgb_callback(progress_bar: tqdm, n_estimators: int) -> XGBoostProgressCallback:
    """Create XGBoost progress callback."""
    return XGBoostProgressCallback(progress_bar, n_estimators)


class TrainingProgressTracker:
    """Track training progress with timing and ETA."""
    
    def __init__(self, model_name: str, total_folds: int, logger=None, progress_manager=None):
        self.model_name = model_name
        self.total_folds = total_folds
        self.current_fold = 0
        self.fold_start_time = None
        self.logger = logger
        self.progress_manager = progress_manager
        self.fold_times = []
        self.fold_bar = None
        self._bar_context = None
        
        if progress_manager and progress_manager.enabled:
            self._bar_context = progress_manager.progress(
                total=total_folds,
                desc=f"{model_name} CV",
                unit="fold"
            )
            self.fold_bar = self._bar_context.__enter__()
    
    def start_fold(self, fold_num: int):
        """Mark start of a fold."""
        self.current_fold = fold_num
        self.fold_start_time = time.time()
        
        msg = f"[{self.model_name}] Fold {fold_num}/{self.total_folds} started"
        if self.logger:
            self.logger.info(msg)
        elif self.progress_manager:
            self.progress_manager.log(msg)
    
    def end_fold(self, metrics: dict = None):
        """Mark end of a fold and report metrics."""
        if self.fold_start_time is None:
            return
        
        fold_time = time.time() - self.fold_start_time
        self.fold_times.append(fold_time)
        
        msg = f"[{self.model_name}] Fold {self.current_fold}/{self.total_folds} completed in {fold_time:.1f}s"
        if metrics:
            acc = metrics.get('accuracy', 0)
            f1 = metrics.get('macro_f1', 0)
            msg += f" | Acc: {acc:.4f} | Macro-F1: {f1:.4f}"
        
        if self.logger:
            self.logger.info(msg)
        elif self.progress_manager:
            self.progress_manager.log(msg)
        
        if self.fold_bar and hasattr(self.fold_bar, 'update'):
            self.fold_bar.update(1)
            if metrics:
                self.fold_bar.set_postfix_str(f"Acc: {metrics.get('accuracy', 0):.4f}")
    
    def finish(self):
        """Finish tracking and report summary."""
        if self._bar_context:
            self._bar_context.__exit__(None, None, None)
        
        total_time = sum(self.fold_times)
        avg_time = total_time / len(self.fold_times) if self.fold_times else 0
        
        msg = f"[{self.model_name}] CV complete: {len(self.fold_times)} folds, "
        msg += f"total={total_time:.1f}s, avg={avg_time:.1f}s/fold"
        
        if self.logger:
            self.logger.info(msg)
        elif self.progress_manager:
            self.progress_manager.log(msg)
        
        return {
            'fold_times': self.fold_times,
            'total_time': total_time,
            'avg_time': avg_time
        }


def track_training_time(model_name: str, logger=None, progress_manager=None):
    """Context manager to track training time."""
    class TimeTracker:
        def __init__(self):
            self.start_time = None
            self.elapsed = 0
        
        def __enter__(self):
            self.start_time = time.time()
            msg = f"Training {model_name}..."
            if logger:
                logger.info(msg)
            elif progress_manager:
                progress_manager.log(msg)
            return self
        
        def __exit__(self, *args):
            self.elapsed = time.time() - self.start_time
            msg = f"  Completed in {self.elapsed:.1f}s"
            if logger:
                logger.info(msg)
            elif progress_manager:
                progress_manager.log(msg)
    
    return TimeTracker()