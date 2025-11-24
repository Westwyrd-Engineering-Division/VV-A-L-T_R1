"""
ConsistencyVerifier

Safety validation and consistency checking for V.V.A.L.T outputs.
"""

import numpy as np
from typing import Tuple, Dict
import warnings


class ConsistencyVerifier:
    """
    Validates safety and consistency of vector outputs.

    Detects NaN and Inf values, clips outputs to safe ranges, and provides
    fail-safe behavior on invalid inputs. Ensures system safety guarantees.
    """

    def __init__(self, safe_range: Tuple[float, float] = (-10.0, 10.0)):
        """
        Initialize ConsistencyVerifier.

        Args:
            safe_range: Tuple of (min, max) for safe output range
        """
        self.safe_min, self.safe_max = safe_range

    def check_validity(self, x: np.ndarray) -> Dict[str, bool]:
        """
        Check if vector contains invalid values.

        Args:
            x: Vector to check

        Returns:
            Dictionary with validity flags
        """
        return {
            "has_nan": bool(np.any(np.isnan(x))),
            "has_inf": bool(np.any(np.isinf(x))),
            "has_invalid": bool(np.any(np.isnan(x)) or np.any(np.isinf(x))),
            "in_range": bool(np.all((x >= self.safe_min) & (x <= self.safe_max)))
        }

    def verify(self, x: np.ndarray, strict: bool = False) -> np.ndarray:
        """
        Verify and sanitize vector output.

        Args:
            x: Input vector to verify
            strict: If True, raise exception on invalid values. If False, sanitize.

        Returns:
            Verified and sanitized vector

        Raises:
            ValueError: If strict=True and invalid values detected
        """
        validity = self.check_validity(x)

        # Check for invalid values
        if validity["has_invalid"]:
            if strict:
                raise ValueError(
                    f"Invalid values detected: "
                    f"NaN={validity['has_nan']}, Inf={validity['has_inf']}"
                )
            else:
                warnings.warn(
                    "Invalid values detected. Applying fail-safe sanitization.",
                    RuntimeWarning
                )
                # Replace NaN with 0, Inf with safe bounds
                x = np.nan_to_num(
                    x,
                    nan=0.0,
                    posinf=self.safe_max,
                    neginf=self.safe_min
                )

        # Clip to safe range
        x_safe = np.clip(x, self.safe_min, self.safe_max)

        return x_safe

    def verify_deterministic(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        tolerance: float = 1e-9
    ) -> bool:
        """
        Verify that two outputs are identical (determinism check).

        Args:
            x1: First vector
            x2: Second vector
            tolerance: Maximum allowed difference

        Returns:
            True if vectors are identical within tolerance
        """
        if x1.shape != x2.shape:
            return False

        diff = np.abs(x1 - x2)
        max_diff = np.max(diff)

        return bool(max_diff <= tolerance)

    def get_safety_report(self, x: np.ndarray) -> Dict:
        """
        Generate comprehensive safety report for vector.

        Args:
            x: Vector to analyze

        Returns:
            Dictionary containing safety metrics
        """
        validity = self.check_validity(x)

        return {
            "validity": validity,
            "statistics": {
                "mean": float(np.mean(x)) if not validity["has_invalid"] else None,
                "std": float(np.std(x)) if not validity["has_invalid"] else None,
                "min": float(np.min(x)) if not validity["has_invalid"] else None,
                "max": float(np.max(x)) if not validity["has_invalid"] else None,
                "norm": float(np.linalg.norm(x)) if not validity["has_invalid"] else None,
            },
            "safety": {
                "is_safe": validity["in_range"] and not validity["has_invalid"],
                "safe_range": (self.safe_min, self.safe_max),
                "needs_clipping": not validity["in_range"],
                "needs_sanitization": validity["has_invalid"],
            }
        }

    def verify_bounds(self, x: np.ndarray) -> bool:
        """
        Verify that vector is within safe bounds.

        Args:
            x: Vector to check

        Returns:
            True if all values are within safe range
        """
        validity = self.check_validity(x)
        return validity["in_range"] and not validity["has_invalid"]
