#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lqsoliveira

normalizer.py
=============
Modular structure for vectorized normalization with accumulated state.

Hierarchy:
    AlgorithmConfig
        └── NormalizerConfig   (static user parameters)
    Normalizer                 (dynamic state + operational logic)
        └── uses NormalizerConfig internally
"""

#from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────
# 1. CONFIGURATIONS (dataclasses — immutable by contract)
# ─────────────────────────────────────────────

@dataclass
class NormalizerConfig:
    """
    Static configuration parameters for the normalizer.
    Defined by the user before execution; do not change during processing.

    Attributes
    ----------
    feature_range : tuple[float, float]
        Target normalization interval. Default (0.0, 1.0).
    clip : bool
        If True, applies np.clip after normalizing, ensuring [low, high]
        even with floating point imprecision. Default True.
    handle_constant : str
        Strategy for constant columns/features (max == min):
          "zero"  → normalized = 0.0  (default)
          "half"  → normalized = 0.5
          "raise" → raises ValueError
    min_vals : array-like | float | None
        User-predefined minima. None = automatically calculated
        at the first fit/update from the data.
    max_vals : array-like | float | None
        User-predefined maxima. Same semantics as min_vals.
    dtype : np.dtype
        Numerical type used internally. Default np.float64.
    """
    feature_range:    tuple             = (0.0, 1.0)
    clip:             bool              = True
    handle_constant:  str               = "zero"
    min_vals:         Optional[object]  = None   # array-like, scalar or None
    max_vals:         Optional[object]  = None   # array-like, scalar or None
    dtype:            np.dtype          = np.float64

    def __post_init__(self) -> None:
        low, high = self.feature_range
        if low >= high:
            raise ValueError(
                f"Invalid feature_range: low={low} must be < high={high}."
            )
        if self.handle_constant not in {"zero", "half", "raise"}:
            raise ValueError(
                f"Invalid handle_constant='{self.handle_constant}'. "
                "Use 'zero', 'half' or 'raise'."
            )

# ─────────────────────────────────────────────
# 2. MODULAR FUNCTIONS (stateless — reusable independently)
# ─────────────────────────────────────────────

def normalize_array(
    arr:              np.ndarray,
    min_vals:         np.ndarray,
    max_vals:         np.ndarray,
    feature_range:    tuple = (0.0, 1.0),
    clip:             bool  = True,
    handle_constant:  str   = "zero",
) -> np.ndarray:
    """
    Normalizes ``arr`` to ``feature_range`` using provided minima and maxima.

    This function is *stateless*: it does not accumulate or modify min/max.
    For incremental use (streaming, batches), use the ``Normalizer`` class.

    Parameters
    ----------
    arr : np.ndarray
        Array to normalize. Any shape compatible with min_vals/max_vals.
    min_vals : np.ndarray | scalar
        Reference minima. Must be broadcastable with arr.
    max_vals : np.ndarray | scalar
        Reference maxima. Must be broadcastable with arr.
    feature_range : tuple[float, float]
        Target interval [low, high]. Default (0.0, 1.0).
    clip : bool
        Applies clipping to the result to guarantee [low, high]. Default True.
    handle_constant : str
        Strategy for constant columns — "zero", "half" or "raise".

    Returns
    -------
    np.ndarray
        Array normalized to the feature_range interval.

    Raises
    ------
    ValueError
        If shapes are incompatible, feature_range is invalid,
        or handle_constant="raise" and constant columns exist.
    """
    arr       = np.asarray(arr)
    min_vals  = np.asarray(min_vals)
    max_vals  = np.asarray(max_vals)
    low, high = feature_range

    # Validates shape compatibility via broadcasting
    try:
        np.broadcast_shapes(arr.shape, min_vals.shape, max_vals.shape)
    except ValueError:
        raise ValueError(
            f"Incompatible shapes for broadcasting: "
            f"arr={arr.shape}, min_vals={min_vals.shape}, max_vals={max_vals.shape}."
        )

    # Detects constant columns/features
    constant_mask = max_vals == min_vals
    if np.any(constant_mask):
        if handle_constant == "raise":
            raise ValueError(
                "Constant columns detected (max == min). "
                "Set handle_constant to 'zero' or 'half' to handle them."
            )

    # Safe denominator: replaces 0 with 1.0 to avoid division by zero
    range_vals = np.where(constant_mask, 1.0, max_vals - min_vals)

    # Vectorized normalization to [0, 1]
    normalized = (arr - min_vals) / range_vals

    # Applies fixed value to constant features
    if np.any(constant_mask):
        fill_value = 0.5 if handle_constant == "half" else 0.0
        normalized = np.where(constant_mask, fill_value, normalized)

    # Scales to arbitrary feature_range
    if (low, high) != (0.0, 1.0):
        normalized = normalized * (high - low) + low

    # Final clipping against float imprecision
    if clip:
        normalized = np.clip(normalized, low, high)

    return normalized


def compute_min_max(arr: np.ndarray, axis: Optional[int] = None):
    """
    Calculates minima and maxima of an array along an axis.

    Parameters
    ----------
    arr : np.ndarray
        Input data.
    axis : int | None
        Axis along which to calculate. None = global.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (min_vals, max_vals)
    """
    return np.min(arr, axis=axis), np.max(arr, axis=axis)


def update_min_max(
    current_min: np.ndarray,
    current_max: np.ndarray,
    new_arr:     np.ndarray,
    axis:        Optional[int] = None,
):
    """
    Updates accumulated minima and maxima with new data (incremental use).

    Parameters
    ----------
    current_min, current_max : np.ndarray
        Accumulated minima/maxima up to the current moment.
    new_arr : np.ndarray
        New batch of data.
    axis : int | None
        Axis along which to calculate on the new data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (updated_min, updated_max)
    """
    batch_min, batch_max = compute_min_max(new_arr, axis=axis)
    return np.minimum(current_min, batch_min), np.maximum(current_max, batch_max)


# ─────────────────────────────────────────────
# 3. NORMALIZER CLASS (stateful — incremental / streaming use)
# ─────────────────────────────────────────────

class Normalizer:
    """
    Normalizer with accumulated state.
    Follows the fit / transform / fit_transform pattern from scikit-learn.

    Basic usage
    ----------
    >>> config = NormalizerConfig(feature_range=(0.0, 1.0), clip=True)
    >>> norm = Normalizer(config)
    >>> norm.fit(data)
    >>> output = norm.transform(data)

    Incremental usage (streaming / batches)
    --------------------------------------
    >>> norm = Normalizer(NormalizerConfig())
    >>> for batch in stream:
    ...     norm.partial_fit(batch)
    >>> output = norm.transform(full_data)
    """

    def __init__(self, config: NormalizerConfig) -> None:
        self.config = config
        # Dynamic state — filled during fit/partial_fit
        self._fitted_min: Optional[np.ndarray] = None
        self._fitted_max: Optional[np.ndarray] = None
        self._is_fitted:  bool                 = False

    # ── read-only properties ──────────────────

    @property
    def fitted_min(self) -> Optional[np.ndarray]:
        return self._fitted_min

    @property
    def fitted_max(self) -> Optional[np.ndarray]:
        return self._fitted_max

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    # ── main methods ───────────────────────

    def fit(self, arr: np.ndarray, axis: Optional[int] = 0) -> "Normalizer":
        """
        Calculates and stores min/max from ``arr``.
        Replaces any previous state.

        Parameters
        ----------
        arr : np.ndarray
            Reference data for fitting.
        axis : int | None
            Axis along which to calculate min/max.
            0 = per column/feature (default); None = global.

        Returns
        -------
        self  (allows chaining: norm.fit(data).transform(data))
        """
        arr = np.asarray(arr, dtype=self.config.dtype)

        # If the user predefined min/max in the config, uses those values
        if self.config.min_vals is not None and self.config.max_vals is not None:
            self._fitted_min = np.asarray(self.config.min_vals, dtype=self.config.dtype)
            self._fitted_max = np.asarray(self.config.max_vals, dtype=self.config.dtype)
        else:
            self._fitted_min, self._fitted_max = compute_min_max(arr, axis=axis)

        self._is_fitted = True
        return self

    def partial_fit(self, arr: np.ndarray, axis: Optional[int] = 0) -> "Normalizer":
        """
        Updates accumulated min/max with new data (batches / streaming).
        If not yet fitted, equivalent to ``fit``.

        Parameters
        ----------
        arr : np.ndarray
            New batch of data.
        axis : int | None
            Axis along which to calculate min/max on the batch.

        Returns
        -------
        self
        """
        arr = np.asarray(arr, dtype=self.config.dtype)

        if not self._is_fitted:
            return self.fit(arr, axis=axis)

        self._fitted_min, self._fitted_max = update_min_max(
            self._fitted_min, self._fitted_max, arr, axis=axis
        )
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        """
        Normalizes ``arr`` using stored min/max.

        Parameters
        ----------
        arr : np.ndarray
            Data to normalize.

        Returns
        -------
        np.ndarray
            Normalized array.

        Raises
        ------
        RuntimeError
            If called before ``fit`` or ``partial_fit``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Normalizer has not been fitted. Call fit() or partial_fit() before transform()."
            )

        arr = np.asarray(arr, dtype=self.config.dtype)

        return normalize_array(
            arr             = arr,
            min_vals        = self._fitted_min,
            max_vals        = self._fitted_max,
            feature_range   = self.config.feature_range,
            clip            = self.config.clip,
            handle_constant = self.config.handle_constant,
        )

    def fit_transform(self, arr: np.ndarray, axis: Optional[int] = 0) -> np.ndarray:
        """
        Shortcut for fit(arr) followed by transform(arr).

        Parameters
        ----------
        arr : np.ndarray
            Reference data and data to normalize.
        axis : int | None
            Axis for fitting.

        Returns
        -------
        np.ndarray
            Normalized array.
        """
        return self.fit(arr, axis=axis).transform(arr)

    def reset(self) -> "Normalizer":
        """Discards the accumulated state. Useful for re-fitting without creating a new instance."""
        self._fitted_min = None
        self._fitted_max = None
        self._is_fitted  = False
        return self

    def __repr__(self) -> str:
        status = (
            f"fitted (min={self._fitted_min}, max={self._fitted_max})"
            if self._is_fitted else "not fitted"
        )
        return f"Normalizer(config={self.config}, status={status})"


# ─────────────────────────────────────────────
# 4. USAGE EXAMPLE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    # ── Global algorithm configuration ──
    algo_cfg = AlgorithmConfig(
        normalizer=NormalizerConfig(
            feature_range   = (0.0, 1.0),
            clip            = True,
            handle_constant = "zero",
        ),
        seed  = 42,
        dtype = np.float64,
    )

    # ── Usage via class (stateful) ──
    data = np.random.randn(100, 4)               # 100 samples, 4 features

    norm = Normalizer(algo_cfg.normalizer)
    result = norm.fit_transform(data)

    print("=== Normalizer (stateful) ===")
    print(f"  Input shape : {data.shape}")
    print(f"  Output shape   : {result.shape}")
    print(f"  Min per column: {result.min(axis=0).round(4)}")
    print(f"  Max per column: {result.max(axis=0).round(4)}")

    # ── Incremental usage (partial_fit) ──
    norm_inc = Normalizer(algo_cfg.normalizer)
    for batch in np.array_split(data, 5):
        norm_inc.partial_fit(batch)
    result_inc = norm_inc.transform(data)

    print("\n=== Incremental Normalizer (partial_fit x5) ===")
    print(f"  Accumulated min : {norm_inc.fitted_min.round(4)}")
    print(f"  Accumulated max : {norm_inc.fitted_max.round(4)}")
    print(f"  Results equal to complete fit: {np.allclose(result, result_inc)}")

    # ── Usage via modular function (stateless) ──
    mn, mx = compute_min_max(data, axis=0)
    result_fn = normalize_array(data, mn, mx)

    print("\n=== normalize_array (stateless) ===")
    print(f"  Min per column: {result_fn.min(axis=0).round(4)}")
    print(f"  Max per column: {result_fn.max(axis=0).round(4)}")

    # ── Constant column test ──
    data_const = data.copy()
    data_const[:, 2] = 5.0                      # constant column 2

    norm_c = Normalizer(algo_cfg.normalizer)
    result_c = norm_c.fit_transform(data_const)

    print("\n=== Constant column (handle_constant='zero') ===")
    print(f"  Column 2 after normalization: {np.unique(result_c[:, 2])}")