"""
TraceNormalizer

Normalizes reasoning trace data for consistent formatting and downstream processing.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import warnings


class TraceNormalizer:
    """
    Normalizes and standardizes reasoning trace data from V.V.A.L.T.

    Ensures trace data is consistently formatted, scaled, and ready for
    analysis, storage, or comparison across different runs. Provides
    deterministic normalization for reproducible trace processing.
    """

    def __init__(
        self,
        scale_method: str = "standard",
        handle_missing: str = "zero",
        precision: int = 6,
        seed: int = 42
    ):
        """
        Initialize TraceNormalizer.

        Args:
            scale_method: Method for scaling numerical values
                - "standard": Zero mean, unit variance (z-score)
                - "minmax": Scale to [0, 1] range
                - "robust": Use median and IQR for outlier resistance
                - "none": No scaling applied
            handle_missing: How to handle missing/None values
                - "zero": Replace with 0.0
                - "mean": Replace with mean of available values
                - "median": Replace with median of available values
                - "skip": Skip normalization for fields with missing values
            precision: Number of decimal places for rounding (for consistency)
            seed: Random seed for deterministic behavior
        """
        self.scale_method = scale_method
        self.handle_missing = handle_missing
        self.precision = precision
        self.seed = seed

        # Validation
        valid_scale_methods = ["standard", "minmax", "robust", "none"]
        if scale_method not in valid_scale_methods:
            raise ValueError(
                f"Invalid scale_method '{scale_method}'. "
                f"Must be one of {valid_scale_methods}"
            )

        valid_missing_handlers = ["zero", "mean", "median", "skip"]
        if handle_missing not in valid_missing_handlers:
            raise ValueError(
                f"Invalid handle_missing '{handle_missing}'. "
                f"Must be one of {valid_missing_handlers}"
            )

        np.random.seed(seed)

        # Statistics cache for consistent normalization across batches
        self.stats_cache = {}
        self.cache_enabled = False

    def enable_cache(self, cache: Optional[Dict] = None):
        """
        Enable statistics caching for consistent normalization.

        Args:
            cache: Optional pre-computed statistics cache
        """
        self.cache_enabled = True
        if cache is not None:
            self.stats_cache = cache

    def disable_cache(self):
        """Disable caching and clear cache."""
        self.cache_enabled = False
        self.stats_cache = {}

    def get_cache(self) -> Dict:
        """
        Get current statistics cache.

        Returns:
            Dictionary of cached statistics
        """
        return self.stats_cache.copy()

    def normalize_trace(self, trace: Dict) -> Dict:
        """
        Normalize complete reasoning trace.

        Args:
            trace: Raw reasoning trace from V.V.A.L.T

        Returns:
            Normalized trace dictionary
        """
        if not isinstance(trace, dict):
            raise TypeError(f"Trace must be a dictionary, got {type(trace)}")

        normalized = {}

        # Normalize each top-level component
        for key, value in trace.items():
            if isinstance(value, dict):
                normalized[key] = self._normalize_dict(value, prefix=key)
            elif isinstance(value, (list, np.ndarray)):
                normalized[key] = self._normalize_array(value, name=key)
            elif isinstance(value, (int, float, np.number)):
                normalized[key] = self._normalize_scalar(value, name=key)
            else:
                # Keep non-numeric data as-is
                normalized[key] = value

        return normalized

    def _normalize_dict(self, data: Dict, prefix: str = "") -> Dict:
        """
        Recursively normalize dictionary data.

        Args:
            data: Dictionary to normalize
            prefix: Prefix for cache keys

        Returns:
            Normalized dictionary
        """
        normalized = {}

        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                normalized[key] = self._normalize_dict(value, prefix=full_key)
            elif isinstance(value, (list, np.ndarray)):
                normalized[key] = self._normalize_array(value, name=full_key)
            elif isinstance(value, (int, float, np.number)):
                normalized[key] = self._normalize_scalar(value, name=full_key)
            elif value is None:
                normalized[key] = self._handle_missing_value(full_key)
            else:
                # Keep non-numeric data as-is
                normalized[key] = value

        return normalized

    def _normalize_array(
        self,
        arr: np.ndarray,
        name: str = "array"
    ) -> np.ndarray:
        """
        Normalize array data.

        Args:
            arr: Array to normalize
            name: Name for cache key

        Returns:
            Normalized array
        """
        # Convert to numpy array if needed
        if isinstance(arr, list):
            arr = np.array(arr)

        # Handle missing values
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            arr = self._sanitize_array(arr)

        # Apply scaling
        if self.scale_method != "none":
            arr = self._scale_array(arr, name)

        # Round for consistency
        arr = np.round(arr, decimals=self.precision)

        return arr

    def _normalize_scalar(
        self,
        value: float,
        name: str = "scalar"
    ) -> float:
        """
        Normalize scalar value.

        Args:
            value: Scalar to normalize
            name: Name for cache key

        Returns:
            Normalized scalar
        """
        # Handle invalid values
        if np.isnan(value) or np.isinf(value):
            value = self._handle_missing_value(name)

        # Apply scaling (treat as single-element array)
        if self.scale_method != "none":
            arr = np.array([value])
            scaled = self._scale_array(arr, name)
            value = float(scaled[0])

        # Round for consistency
        value = round(value, self.precision)

        return value

    def _scale_array(self, arr: np.ndarray, name: str) -> np.ndarray:
        """
        Scale array based on configured method.

        Args:
            arr: Array to scale
            name: Name for statistics caching

        Returns:
            Scaled array
        """
        # Use cached statistics if available
        if self.cache_enabled and name in self.stats_cache:
            stats = self.stats_cache[name]
        else:
            stats = self._compute_stats(arr)
            if self.cache_enabled:
                self.stats_cache[name] = stats

        if self.scale_method == "standard":
            # Z-score normalization
            mean = stats["mean"]
            std = stats["std"]
            if std > 1e-8:
                return (arr - mean) / std
            else:
                return arr - mean

        elif self.scale_method == "minmax":
            # Min-max scaling to [0, 1]
            min_val = stats["min"]
            max_val = stats["max"]
            if max_val - min_val > 1e-8:
                return (arr - min_val) / (max_val - min_val)
            else:
                return np.zeros_like(arr)

        elif self.scale_method == "robust":
            # Robust scaling using median and IQR
            median = stats["median"]
            q1 = stats["q1"]
            q3 = stats["q3"]
            iqr = q3 - q1
            if iqr > 1e-8:
                return (arr - median) / iqr
            else:
                return arr - median

        return arr

    def _compute_stats(self, arr: np.ndarray) -> Dict:
        """
        Compute statistics for array.

        Args:
            arr: Array to analyze

        Returns:
            Dictionary of statistics
        """
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "q1": float(np.percentile(arr, 25)),
            "q3": float(np.percentile(arr, 75)),
        }

    def _sanitize_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Sanitize array by handling NaN and Inf values.

        Args:
            arr: Array to sanitize

        Returns:
            Sanitized array
        """
        # Make a copy to avoid modifying original
        arr = arr.copy()

        # Replace infinities with large finite values
        arr = np.nan_to_num(arr, nan=np.nan, posinf=1e10, neginf=-1e10)

        # Handle remaining NaN values
        if np.any(np.isnan(arr)):
            if self.handle_missing == "zero":
                arr = np.nan_to_num(arr, nan=0.0)
            elif self.handle_missing == "mean":
                mean_val = np.nanmean(arr)
                arr = np.where(np.isnan(arr), mean_val, arr)
            elif self.handle_missing == "median":
                median_val = np.nanmedian(arr)
                arr = np.where(np.isnan(arr), median_val, arr)
            else:  # skip
                warnings.warn(
                    f"Array contains NaN values and handle_missing='skip'. "
                    f"Values will be replaced with 0.",
                    RuntimeWarning
                )
                arr = np.nan_to_num(arr, nan=0.0)

        return arr

    def _handle_missing_value(self, name: str) -> float:
        """
        Handle missing scalar value.

        Args:
            name: Name of the value

        Returns:
            Replacement value
        """
        if self.handle_missing == "zero":
            return 0.0
        else:
            # For scalar, can't compute mean/median, default to 0
            warnings.warn(
                f"Missing value for '{name}'. Replacing with 0.0",
                RuntimeWarning
            )
            return 0.0

    def normalize_batch_traces(self, traces: List[Dict]) -> List[Dict]:
        """
        Normalize batch of traces with consistent statistics.

        Computes statistics from first trace and applies to all traces
        in batch for consistency.

        Args:
            traces: List of trace dictionaries

        Returns:
            List of normalized trace dictionaries
        """
        if not traces:
            return []

        # Enable caching for consistent normalization
        original_cache_state = self.cache_enabled
        self.enable_cache()

        try:
            # Normalize all traces (first trace will populate cache)
            normalized_traces = [self.normalize_trace(trace) for trace in traces]
        finally:
            # Restore original cache state
            if not original_cache_state:
                self.disable_cache()

        return normalized_traces

    def compare_traces(
        self,
        trace1: Dict,
        trace2: Dict,
        normalize: bool = True
    ) -> Dict:
        """
        Compare two traces and compute difference metrics.

        Args:
            trace1: First trace
            trace2: Second trace
            normalize: If True, normalize traces before comparison

        Returns:
            Dictionary of comparison metrics
        """
        if normalize:
            trace1 = self.normalize_trace(trace1)
            trace2 = self.normalize_trace(trace2)

        differences = {}
        self._compute_trace_differences(trace1, trace2, differences, prefix="")

        return {
            "differences": differences,
            "summary": self._summarize_differences(differences)
        }

    def _compute_trace_differences(
        self,
        t1: Any,
        t2: Any,
        result: Dict,
        prefix: str
    ):
        """
        Recursively compute differences between trace structures.

        Args:
            t1: First trace element
            t2: Second trace element
            result: Dictionary to store results
            prefix: Key prefix for nested structures
        """
        if isinstance(t1, dict) and isinstance(t2, dict):
            for key in set(list(t1.keys()) + list(t2.keys())):
                new_prefix = f"{prefix}.{key}" if prefix else key
                if key in t1 and key in t2:
                    self._compute_trace_differences(
                        t1[key], t2[key], result, new_prefix
                    )
                else:
                    result[new_prefix] = "missing_in_one_trace"

        elif isinstance(t1, (np.ndarray, list)) and isinstance(t2, (np.ndarray, list)):
            arr1 = np.array(t1) if isinstance(t1, list) else t1
            arr2 = np.array(t2) if isinstance(t2, list) else t2

            if arr1.shape == arr2.shape:
                diff = np.abs(arr1 - arr2)
                result[prefix] = {
                    "max_diff": float(np.max(diff)),
                    "mean_diff": float(np.mean(diff)),
                    "std_diff": float(np.std(diff))
                }
            else:
                result[prefix] = f"shape_mismatch_{arr1.shape}_vs_{arr2.shape}"

        elif isinstance(t1, (int, float, np.number)) and isinstance(t2, (int, float, np.number)):
            result[prefix] = float(abs(t1 - t2))

    def _summarize_differences(self, differences: Dict) -> Dict:
        """
        Summarize difference metrics.

        Args:
            differences: Dictionary of differences

        Returns:
            Summary statistics
        """
        numeric_diffs = []

        def extract_numeric(d):
            for key, value in d.items():
                if isinstance(value, dict) and "mean_diff" in value:
                    numeric_diffs.append(value["mean_diff"])
                elif isinstance(value, (int, float)):
                    numeric_diffs.append(value)
                elif isinstance(value, dict):
                    extract_numeric(value)

        extract_numeric(differences)

        if numeric_diffs:
            return {
                "num_differences": len(numeric_diffs),
                "max_difference": float(np.max(numeric_diffs)),
                "mean_difference": float(np.mean(numeric_diffs)),
                "median_difference": float(np.median(numeric_diffs)),
            }
        else:
            return {
                "num_differences": 0,
                "max_difference": 0.0,
                "mean_difference": 0.0,
                "median_difference": 0.0,
            }
