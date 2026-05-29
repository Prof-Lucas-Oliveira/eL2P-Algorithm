#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:29:40 2026

@author: lqsoliveira


fuzzy_rule.py
=============
Module for calculating the activation degree of multiple fuzzy rules with:
  - Gaussian antecedents with a single global sigma
  - Activation degree computed directly via multivariate Gaussian
  - Rule base as matrix (n_rules × n_antecedents)
  - Consequent coefficients: [a_i, b_i] — dimension 2 per rule
  - Local output: y_i = a_i·φ_i + b_i
  - Automatic initialization by the first data point when modal is not defined

compute() return values:
    activations : np.ndarray shape (n_rules,)  — phi of each rule
    memberships: None                          — not calculated
    best_idx  : int — index of the most activated rule
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

@dataclass
class FuzzyRuleConfig:
    """
    Static configuration parameters for the fuzzy rule base.

    Attributes
    ----------
    sigma : float
        Global standard deviation of the Gaussians. Single value for all
        rules and antecedents. Default 0.3.
    modal_values : array-like shape (n_rules, n_antecedents) | None
        Modal values (centers) of the Gaussians.
        - None  → first data point automatically initializes rule 1.
        - 2D    → complete rule base defined by user.
        - 1D    → interpreted as single rule (n_rules=1).
    n_antecedents : int | None
        Number of antecedents per rule. Inferred from modal_values if
        provided; required when modal_values=None.
    coeff_init : float
        Initialization value for the functional consequent coefficients.
        Default 0.0.

    Notes on coefficients
    -------------------
    Each rule has an affine functional model in the consequents:

        y_i^k = a_i · φ_i + b_i

    Coefficients are stored as vector [a_i, b_i]
    of size 2 per rule. The complete matrix has shape:

        (n_rules, 2)

    Coefficient initialization (rank-1):
        At the first step, y^k initializes the coefficients via:

            x̃  = [0, 1]
            a_init = (y_k / (x̃ @ x̃)) * x̃  →  [0, y_k]

        Guarantees y_i(φ=1) = y_k exactly at the first iteration.
        If y_init is provided in the config, initialization occurs
        in __init__; otherwise it occurs at the first update_rule().
    """
    sigma:          float            = 0.3
    modal_values:   Optional[object] = None
    n_antecedents:  Optional[int]    = None
    coeff_init:     float            = 0.0
    alpha:          float            = 0.5    # 1st order filter learning rate ∈ [0,1]
    eta:            float            = 0.1    # NLMS base step ∈ (0,1]
    eps:            float            = 1e-8   # NLMS numerical stability
    y_init:         Optional[float]  = None   # desired output for 1st data point for rank-1 init

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive. Received: {self.sigma}.")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0,1]. Received: {self.alpha}.")
        if not 0.0 < self.eta <= 1.0:
            raise ValueError(f"eta must be in (0,1]. Received: {self.eta}.")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive. Received: {self.eps}.")

        if self.modal_values is not None:
            mv = np.asarray(self.modal_values, dtype=float)

            # Normalizes 1D → 2D (single rule)
            if mv.ndim == 1:
                mv = mv[np.newaxis, :]          # (1, n_antecedents)
            elif mv.ndim != 2:
                raise ValueError(
                    "modal_values must be 1D (single rule) or "
                    "2D shape (n_rules, n_antecedents)."
                )

            # Updates the field to always be 2D
            object.__setattr__(self, 'modal_values', mv)

            n_ant = mv.shape[1]
            if self.n_antecedents is None:
                object.__setattr__(self, 'n_antecedents', n_ant)
            elif self.n_antecedents != n_ant:
                raise ValueError(
                    f"n_antecedents={self.n_antecedents} incompatible with "
                    f"modal_values.shape[1]={n_ant}."
                )
        else:
            if self.n_antecedents is not None and self.n_antecedents < 1:
                raise ValueError(
                    f"n_antecedents must be >= 1. Received: {self.n_antecedents}."
                )


# ─────────────────────────────────────────────
# 2. MODULAR FUNCTIONS (stateless)
# ─────────────────────────────────────────────

def compute_activations(
    x:      np.ndarray,
    modals: np.ndarray,
    sigma:  float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Calculates the activation degree of each rule directly via
    multivariate Gaussian on the vector x.

    With a single global sigma equal in all dimensions, the
    product t-norm of univariate Gaussians is identical to the
    multivariate Gaussian:

        phi[r] = exp( -0.5 · ||x - c[r]||² / sigma² )

    The operation is performed in a vectorized manner over all
    rules simultaneously, without the need to calculate the
    membership matrix per coordinate nor explicitly apply the
    t-norm.

    Parameters
    ----------
    x      : shape (n_antecedents,)
    modals : shape (n_rules, n_antecedents)
    sigma  : float — global standard deviation

    Returns
    -------
    tuple (activations, memberships, best_idx)
        activations   : shape (n_rules,)  — phi of each rule
        memberships : None               — not calculated (compatibility)
        best_idx    : int — index of the rule with highest phi
    """
    diff      = x - modals                                    # (n_rules, p)
    sq_dist   = np.sum(diff ** 2, axis=1)                    # (n_rules,)
    activations = np.exp(-0.5 * sq_dist / (sigma ** 2))        # (n_rules,)
    best_idx  = int(np.argmax(activations))
    return activations, None, best_idx


def defuzzify(
    phi:    np.ndarray,
    modals: np.ndarray,
    coeffs: np.ndarray,
    eps:    float = 1e-8,
) -> tuple[float, np.ndarray]:
    """
    Calculates the global output of the fuzzy system (defuzzification).

        y_i^k = a_i · φ_i + b_i     (local output evaluated at the activation degree)

        y_out = Σ( φ_i · y_i^k ) / Σ( φ_i )

    Parameters
    ----------
    phi    : np.ndarray shape (n_rules,)
        Activation degrees of the rules (output of compute_activations).
    modals : np.ndarray shape (n_rules, n_antecedents)
        Modal values of the rules (not used in calculation, kept for compatibility).
    coeffs : np.ndarray shape (n_rules, 2)
        Consequent coefficients — [a_i, b_i] per rule.
    eps    : float
        Numerical stability to avoid division by zero.

    Returns
    -------
    tuple (y_out, f_local)
        y_out   : float — global defuzzified output
        f_local : np.ndarray shape (n_rules,) — local output y_i^k
    """
    # Local output of each rule: y_i = a_i·φ_i + b_i
    f_local = coeffs[:, 0] * phi + coeffs[:, 1]
    # Weighted average by activation degrees
    y_out   = float(np.dot(phi, f_local) / (np.sum(phi) + eps))
    return y_out, f_local


def compatibility(
    x:      np.ndarray,
    modals: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Compatibility degree between x and each rule (modal value).

        rho[r] = 1 - ||x - c[r]|| / sqrt(p)

    where p is the dimension of the input vector (n_antecedents) and
    || . || is the Euclidean distance.

    Since data are normalized in [0, 1]^p, the maximum possible
    distance is sqrt(p), guaranteeing rho ∈ [0, 1] by construction.

    Parameters
    ----------
    x      : np.ndarray shape (p,)
        Normalized input vector.
    modals : np.ndarray shape (n_rules, p)
        Matrix of modal values — one row per rule.

    Returns
    -------
    tuple (rho, best_idx)
        rho      : np.ndarray shape (n_rules,) — compatibility degree
        best_idx : int — index of the most compatible rule
    """
    p         = x.shape[0]
    dist      = np.linalg.norm(x - modals, axis=1)   # shape (n_rules,)
    rho       = 1.0 - dist / np.sqrt(p)
    best_idx  = int(np.argmax(rho))
    return rho, best_idx


# ─────────────────────────────────────────────
# 3. FUZZYRULE CLASS (stateful)
# ─────────────────────────────────────────────

class FuzzyRule:
    """
    Fuzzy rule base with Gaussian antecedents and product t-norm.

    The base is a matrix (n_rules × n_antecedents). Each row is the vector
    of modal values of a rule.

    Automatic initialization
    ------------------------
    If modal_values=None, the first data point initializes rule 1.
    New rules can be added via add_rule() or set_rules().

    Usage — user-defined base
    ---------------------------------
    >>> config = FuzzyRuleConfig(
    ...     sigma        = 0.3,
    ...     modal_values = np.array([[0.3, 0.4, 0.6],
    ...                              [0.1, 0.25, 0.75],
    ...                              [0.9, 0.5,  0.1]]),
    ... )
    >>> rule = FuzzyRule(config)
    >>> activations, memberships, best_idx = rule.compute(x_normalized)

    Usage — automatic modal (first data point)
    ---------------------------------------
    >>> config = FuzzyRuleConfig(sigma=0.3, n_antecedents=3)
    >>> rule = FuzzyRule(config)
    >>> activations, memberships, best_idx = rule.compute(x_normalized)
    """

    def __init__(self, config: FuzzyRuleConfig) -> None:
        self.config  = config
        self._modals:            Optional[np.ndarray] = None   # (n_rules, n_antecedents)
        self._coeffs:            Optional[np.ndarray] = None   # (n_rules, 2)
        self._n_updates:         Optional[np.ndarray] = None   # (n_rules,) counter per rule
        self._coeff_initialized: bool                 = False  # True after rank-1 init
        self._is_initialized:    bool                 = False

        if config.modal_values is not None:
            self._modals = np.array(config.modal_values, dtype=float)
            n_rules      = self._modals.shape[0]
            # Coefficients: [a_i, b_i] — dimension 2 per rule
            self._coeffs    = np.full((n_rules, 2), config.coeff_init)
            self._n_updates = np.zeros(n_rules, dtype=int)
            self._is_initialized = True
            # If y_init provided in config, pre-initializes via rank-1
            # using x̃ = [0, 1] as per algorithm definition
            if config.y_init is not None:
                x_aug = np.array([0.0, 1.0])
                a_init = (config.y_init / (x_aug @ x_aug)) * x_aug
                self._coeffs[:] = a_init
                self._coeff_initialized = True

    # ── properties ────────────────────────────

    @property
    def modals(self) -> Optional[np.ndarray]:
        """Matrix of modal values shape (n_rules, n_antecedents)."""
        return self._modals

    @property
    def n_rules(self) -> int:
        """Current number of rules."""
        return 0 if self._modals is None else self._modals.shape[0]

    @property
    def n_antecedents(self) -> Optional[int]:
        """Number of antecedents per rule."""
        return None if self._modals is None else self._modals.shape[1]

    @property
    def coeffs(self) -> Optional[np.ndarray]:
        """
        Matrix of consequent coefficients, shape (n_rules, 2).
        Each row: [a_i, b_i] — angular coefficient and bias of rule i.
        """
        return self._coeffs

    @property
    def n_updates(self) -> Optional[np.ndarray]:
        """Vector of update counters per rule, shape (n_rules,)."""
        return self._n_updates

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    # ── main methods ───────────────────────

    def compute(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Calculates the activation degree of all rules for vector x.

        If not yet initialized, x automatically defines the modal
        of rule 1 (phi=1.0 guaranteed at the first step).

        Parameters
        ----------
        x : array-like shape (n_antecedents,)
            Normalized input vector.

        Returns
        -------
        tuple (activations, memberships, best_idx)
            activations   : np.ndarray shape (n_rules,)
            memberships : None
            best_idx    : int
        """
        x = np.asarray(x, dtype=float).ravel()

        # Validates dimension
        if self.config.n_antecedents is not None:
            if len(x) != self.config.n_antecedents:
                raise ValueError(
                    f"Dimension of x ({len(x)}) incompatible with "
                    f"n_antecedents={self.config.n_antecedents}."
                )

        # Automatic initialization: x becomes rule 1
        if not self._is_initialized:
            self._modals    = x[np.newaxis, :].copy()
            self._coeffs    = np.full((1, 2), self.config.coeff_init)
            self._n_updates = np.zeros(1, dtype=int)
            self._is_initialized = True

        return compute_activations(x, self._modals, self.config.sigma)

    def compatibility(self, x: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Compatibility degree between x and each rule in the base.

            rho[r] = 1 - ||x - c[r]|| / sqrt(p)

        Parameters
        ----------
        x : array-like shape (p,)
            Normalized input vector.

        Returns
        -------
        tuple (rho, best_idx)
            rho      : np.ndarray shape (n_rules,)
            best_idx : int — index of the most compatible rule

        Raises
        ------
        RuntimeError
            If called before any initialization of the rule base.
        """
        if not self._is_initialized:
            raise RuntimeError(
                "Rule base not initialized. "
                "Call compute() or add_rule() before compatibility()."
            )
        x = np.asarray(x, dtype=float).ravel()
        return compatibility(x, self._modals)

    def eL2P(
        self,
        x:    np.ndarray,
        y:    float,
        norm: object,
    ) -> float:
        """
        Evolving Level-2 Participatory Learning (eL2P) — complete step.

        Sequentially executes the 5 stages of the incremental learning
        algorithm for a single input/output pair (x^k, y^k):

            1. Normalization      — norm.partial_fit(x) + norm.transform(x)
            2. Activation          — compute(x_norm)
            3. Compatibility   — compatibility(x_norm) + update_rule(x_norm, y, best)
            4. Redundancy       — check_redundancy()
            5. Defuzzification    — defuzzify(phi)  →  y_out

        Parameters
        ----------
        x : array-like shape (n_antecedents,) or (1, n_antecedents)
            Raw input data (not normalized).
        y : float
            Desired output at instant k.
        norm : Normalizer
            Normalizer instance already configured and fitted.

        Returns
        -------
        float
            Defuzzified output y_out of the fuzzy system at instant k.
        """
        # ── 1. Normalization ──────────────────────────────────────
        x_raw = np.asarray(x, dtype=float)
        if x_raw.ndim == 1:
            x_raw = x_raw[np.newaxis, :]
        norm.partial_fit(x_raw)
        x_norm = norm.transform(x_raw).ravel()

        # ── 2. Activation ──────────────────────────────────────────
        phi, _, _ = self.compute(x_norm)

        # ── 3. Compatibility + update decision ──────────────────────────
        _, best_idx = self.compatibility(x_norm)
        self.update_rule(x_norm, float(y), best_idx)

        # ── 4. Redundancy check ────────────────────────────────────────
        self.check_redundancy()

        # ── 5. Defuzzification ────────────────────────────────────
        # Recalculates phi with possibly updated rule base
        phi_final, _, _ = self.compute(x_norm)
        y_out, _        = self.defuzzify(phi_final)

        return y_out

    def defuzzify(self, phi: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Calculates the global output of the fuzzy system for activation degrees phi.

            y_i^k = a_i · φ_i + b_i
            y_out  = Σ( φ_i · y_i^k ) / Σ( φ_i )

        Parameters
        ----------
        phi : np.ndarray shape (n_rules,)
            Activation degrees of the rules (output of compute()).

        Returns
        -------
        tuple (y_out, f_local)
            y_out   : float — global defuzzified output
            f_local : np.ndarray shape (n_rules,) — local output y_i^k

        Raises
        ------
        RuntimeError
            If called before initialization of the rule base.
        """
        if not self._is_initialized:
            raise RuntimeError(
                "Rule base not initialized. "
                "Call compute() before defuzzify()."
            )
        phi = np.asarray(phi, dtype=float).ravel()
        if len(phi) != self.n_rules:
            raise ValueError(
                f"Dimension of phi ({len(phi)}) incompatible with "
                f"n_rules={self.n_rules}."
            )
        return defuzzify(phi, self._modals, self._coeffs, self.config.eps)

    def check_redundancy(self) -> dict:
        """
        Checks and resolves redundancy between all pairs of clusters (i,j).

        Criterion:
            dist(c_i, c_j) < 3·sigma  →  redundant clusters

        Action (first occurrence found, index i < j):
            - modal[j]  ← arithmetic mean (modal[i]  + modal[j])  / 2
            - coeffs[j] ← arithmetic mean (coeffs[i] + coeffs[j]) / 2
            - n_updates[j] ← max(n_updates[i], n_updates[j])
            - removes cluster i

        Single merge per call. To eliminate all redundancies,
        call repeatedly until action='none'.

        Returns
        -------
        dict with keys:
            'action'  : str   — 'merged' | 'none'
            'pair'    : tuple — (i, j) merged, or None
            'dist'    : float — distance of the merged pair, or None
            'n_rules' : int   — number of rules after the operation
        """
        if not self._is_initialized or self.n_rules < 2:
            return {'action': 'none', 'pair': None, 'dist': None,
                    'n_rules': self.n_rules}

        threshold = 3.0 * self.config.sigma

        # Scans all pairs (i,j) with i < j — lexicographic order
        for i in range(self.n_rules):
            for j in range(i + 1, self.n_rules):
                dist = float(np.linalg.norm(self._modals[i] - self._modals[j]))

                if dist < threshold:
                    # ── Merge: arithmetic mean ──
                    self._modals[j]    = (self._modals[i]  + self._modals[j])  / 2.0
                    self._coeffs[j]    = (self._coeffs[i]  + self._coeffs[j])  / 2.0
                    self._n_updates[j] = max(self._n_updates[i], self._n_updates[j])

                    # ── Removes cluster i ──
                    self._modals    = np.delete(self._modals,    i, axis=0)
                    self._coeffs    = np.delete(self._coeffs,    i, axis=0)
                    self._n_updates = np.delete(self._n_updates, i)

                    if self.n_rules == 0:
                        self._is_initialized = False

                    return {'action': 'merged', 'pair': (i, j),
                            'dist': dist, 'n_rules': self.n_rules}

        return {'action': 'none', 'pair': None, 'dist': None,
                'n_rules': self.n_rules}

    def update_rule(
        self,
        x:        np.ndarray,
        y:        float,
        best_idx: int,
    ) -> dict:
        """
        Applies the update decision structure for data point x^k.

        Decision structure (cluster with highest compatibility = best_idx):

            dist ≤ sigma
                → modal unchanged; NLMS updates functional

            sigma < dist ≤ 3·sigma
                → 1st order filter updates modal;
                  NLMS updates functional

            dist > 3·sigma
                → new cluster created with v = x^k;
                  coefficients inherited from best_idx (Option 1)

        In all cases NLMS uses:
            x̃      = [μ_i, 1]
            error    = y^k − y_i^k = y^k − (a_i·μ_i + b_i)
            eta_ad  = eta / (eps + ||x̃||² · (n_upd + 1))

        Parameters
        ----------
        x        : np.ndarray shape (n_antecedents,)
            Normalized input data at instant k.
        y        : float
            Desired output provided externally at instant k.
        best_idx : int
            Index of the rule with highest compatibility (output of compatibility()).

        Returns
        -------
        dict with keys:
            'action'   : str   — 'no_change' | 'update' | 'new_rule'
            'dist'     : float — Euclidean distance
            'error'     : float — functional error before update
            'best_idx' : int   — index of the affected rule (or new rule)
        """
        if not self._is_initialized:
            raise RuntimeError(
                "Rule base not initialized. "
                "Call compute() before update_rule()."
            )

        x = np.asarray(x, dtype=float).ravel()

        cfg   = self.config
        c     = self._modals[best_idx]
        dist  = float(np.linalg.norm(x - c))

        # ── Activation degree of rule best_idx (vectorized form) ──
        sq_dist_best = float(np.sum((x - c) ** 2))
        mu_best      = float(np.exp(-0.5 * sq_dist_best / (cfg.sigma ** 2)))

        # ── Rank-1 initialization at first data point ──
        # x̃ = [0, 1] as per algorithm definition (zero level)
        if not self._coeff_initialized:
            x_aug  = np.array([0.0, 1.0])
            a_init = (float(y) / (x_aug @ x_aug)) * x_aug
            self._coeffs[:] = a_init
            self._coeff_initialized = True

        # ── Local output: y_i = a_i·μ_i + b_i ──
        x_aug = np.array([mu_best, 1.0])
        f_c   = float(self._coeffs[best_idx] @ x_aug)
        error  = float(y) - f_c

        # ── Adaptive eta: step shrinks with n_updates ──
        n_upd  = int(self._n_updates[best_idx])
        eta_ad = cfg.eta / (cfg.eps + float(x_aug @ x_aug) * (n_upd + 1))

        if dist <= cfg.sigma:
            # ── Case 1: inside cluster — fixed modal, updates functional ──
            self._coeffs[best_idx]    += eta_ad * error * x_aug
            self._n_updates[best_idx] += 1
            action = 'no_change'

        elif dist <= 3.0 * cfg.sigma:
            # ── Case 2: update zone ──
            # 1. 1st order digital filter on modal
            self._modals[best_idx] = (cfg.alpha * c
                                      + (1.0 - cfg.alpha) * x)
            # 2. NLMS on coefficients using x̃ = [μ_i, 1]
            self._coeffs[best_idx]    += eta_ad * error * x_aug
            self._n_updates[best_idx] += 1
            action = 'update'

        else:
            # ── Case 3: dist > 3σ — creates new cluster ──
            new_modal  = x.copy()
            new_coeffs = self._coeffs[best_idx].copy()   # inherits from best_idx
            self._modals    = np.vstack([self._modals,    new_modal])
            self._coeffs    = np.vstack([self._coeffs,    new_coeffs])
            self._n_updates = np.append(self._n_updates,  0)
            action    = 'new_rule'
            best_idx  = self.n_rules - 1   # index of the new rule

        return {
            'action':   action,
            'dist':     dist,
            'error':     error,
            'best_idx': best_idx,
        }

    def add_rule(self, modal: np.ndarray) -> "FuzzyRule":
        """
        Adds a new rule to the base.

        Parameters
        ----------
        modal : array-like shape (n_antecedents,)
            Vector of modal values for the new rule.

        Returns
        -------
        self
        """
        modal = np.asarray(modal, dtype=float).ravel()

        if self._is_initialized and modal.shape[0] != self.n_antecedents:
            raise ValueError(
                f"Dimension of modal ({modal.shape[0]}) incompatible with "
                f"n_antecedents={self.n_antecedents}."
            )

        n_ant         = modal.shape[0]
        new_coeff_row = np.full((1, 2), self.config.coeff_init)

        if self._modals is None:
            self._modals    = modal[np.newaxis, :]
            self._coeffs    = new_coeff_row
            self._n_updates = np.zeros(1, dtype=int)
            self._is_initialized = True
        else:
            self._modals    = np.vstack([self._modals,   modal])
            self._coeffs    = np.vstack([self._coeffs,   new_coeff_row])
            self._n_updates = np.append(self._n_updates, 0)

        return self

    def remove_rule(self, idx: int) -> "FuzzyRule":
        """
        Removes the rule at index idx from the base.

        Parameters
        ----------
        idx : int
            Index of the rule to remove (0-based).

        Returns
        -------
        self
        """
        if self._modals is None or idx >= self.n_rules or idx < 0:
            raise IndexError(
                f"Index {idx} invalid for base with {self.n_rules} rule(s)."
            )
        self._modals    = np.delete(self._modals,    idx, axis=0)
        self._coeffs    = np.delete(self._coeffs,    idx, axis=0)
        self._n_updates = np.delete(self._n_updates, idx)
        if self.n_rules == 0:
            self._modals         = None
            self._coeffs         = None
            self._n_updates      = None
            self._is_initialized = False
        return self

    def set_rules(self, modals: np.ndarray) -> "FuzzyRule":
        """
        Replaces the entire rule base.

        Parameters
        ----------
        modals : array-like shape (n_rules, n_antecedents)

        Returns
        -------
        self
        """
        modals = np.asarray(modals, dtype=float)
        if modals.ndim == 1:
            modals = modals[np.newaxis, :]
        if modals.ndim != 2:
            raise ValueError("modals must be 2D shape (n_rules, n_antecedents).")
        n_rules         = modals.shape[0]
        self._modals    = modals
        self._coeffs    = np.full((n_rules, 2), self.config.coeff_init)
        self._n_updates = np.zeros(n_rules, dtype=int)
        self._is_initialized = True
        return self

    def reset(self) -> "FuzzyRule":
        """Discards the entire learned rule base and resets coefficients."""
        if self.config.modal_values is not None:
            self._modals    = np.array(self.config.modal_values, dtype=float)
            n_rules         = self._modals.shape[0]
            self._coeffs    = np.full((n_rules, 2), self.config.coeff_init)
            self._n_updates = np.zeros(n_rules, dtype=int)
            self._is_initialized = True
        else:
            self._modals         = None
            self._coeffs         = None
            self._n_updates      = None
            self._is_initialized = False
        return self

    def __repr__(self) -> str:
        if self._is_initialized:
            status = (f"{self.n_rules} rule(s), "
                      f"n_antecedents={self.n_antecedents}, "
                      f"coeffs={self._coeffs.shape}, "
                      f"n_updates={self._n_updates.tolist()}")
        else:
            status = "not initialized"
        return f"FuzzyRule(sigma={self.config.sigma}, {status})"


# ─────────────────────────────────────────────
# 4. USAGE EXAMPLE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("CASE A — Modal NOT defined: first data point initializes")
    print("=" * 60)

    config_a = FuzzyRuleConfig(sigma=0.3, n_antecedents=3)
    rule_a   = FuzzyRule(config_a)

    x_k1 = np.array([0.5, 0.5, 0.5])
    act1, mu1, best1 = rule_a.compute(x_k1)
    print(f"\n  k=1  x={x_k1}  →  phi={act1.round(4)}  best=rule[{best1}]")
    print(f"       automatic modal: {rule_a.modals}")

    x_k2 = np.array([0.3, 0.7, 0.5])
    act2, mu2, best2 = rule_a.compute(x_k2)
    print(f"\n  k=2  x={x_k2}  →  phi={act2.round(4)}  best=rule[{best2}]")

    print("\n" + "=" * 60)
    print("CASE B — Base of 3 rules defined by user")
    print("=" * 60)

    config_b = FuzzyRuleConfig(
        sigma        = 0.3,
        modal_values = np.array([[0.3, 0.4, 0.6],
                                 [0.1, 0.25, 0.75],
                                 [0.9, 0.5,  0.1]]),
    )
    rule_b = FuzzyRule(config_b)
    print(f"\n  {rule_b}")

    x = np.array([0.5, 0.5, 0.5])
    act, mu, best = rule_b.compute(x)

    print(f"\n  x = {x}")
    for r in range(rule_b.n_rules):
        marker = " ◄ most activated" if r == best else ""
        print(f"  rule[{r}]  modal={rule_b.modals[r]}  "
              f"phi={act[r]:.6f}{marker}")

    print("\n" + "=" * 60)
    print("CASE C — add_rule() and remove_rule()")
    print("=" * 60)

    rule_b.add_rule(np.array([0.5, 0.5, 0.5]))
    print(f"\n  After add_rule:    {rule_b}")

    rule_b.remove_rule(1)
    print(f"  After remove_rule(1): {rule_b}")

    print("\n" + "=" * 60)
    print("CASE D — compatibility()")
    print("=" * 60)

    config_d = FuzzyRuleConfig(
        sigma        = 0.3,
        modal_values = np.array([[0.3, 0.4, 0.6],
                                 [0.1, 0.25, 0.75],
                                 [0.9, 0.5,  0.1]]),
    )
    rule_d = FuzzyRule(config_d)

    x = np.array([0.5, 0.5, 0.5])
    rho, best = rule_d.compatibility(x)

    print(f"\n  x = {x}  (p={len(x)},  sqrt(p)={np.sqrt(len(x)):.4f})")
    for r in range(rule_d.n_rules):
        dist   = np.linalg.norm(x - rule_d.modals[r])
        marker = " ◄ most compatible" if r == best else ""
        print(f"  rule[{r}]  modal={rule_d.modals[r]}"
              f"  dist={dist:.4f}  rho={rho[r]:.6f}{marker}")

    # Verification of extreme cases
    print("\n  Verification of extremes:")
    x_equal   = rule_d.modals[0].copy()
    x_opposite  = 1.0 - rule_d.modals[0]
    rho_eq, _ = compatibility(x_equal,  rule_d.modals)
    rho_op, _ = compatibility(x_opposite, rule_d.modals)
    print(f"  x == c[0]       →  rho[0] = {rho_eq[0]:.6f}  (expected 1.0)")
    print(f"  x = 1-c[0]      →  rho[0] = {rho_op[0]:.6f}  (expected < 1.0)")

    print("\n" + "=" * 60)
    print("CASE E — Consequent coefficients")
    print("=" * 60)

    config_e = FuzzyRuleConfig(
        sigma        = 0.3,
        modal_values = np.array([[0.3, 0.4, 0.6],
                                 [0.1, 0.25, 0.75],
                                 [0.9, 0.5,  0.1]]),
        coeff_init   = 0.0,
    )
    rule_e = FuzzyRule(config_e)

    print(f"\n  {rule_e}")
    print(f"\n  Coefficient matrix shape {rule_e.coeffs.shape}:")
    print(f"  (n_rules={rule_e.n_rules}, [a_i, b_i])")
    for r in range(rule_e.n_rules):
        a, b = rule_e.coeffs[r, 0], rule_e.coeffs[r, 1]
        print(f"  rule[{r}]  a={a:.1f}  b={b:.1f}   →  y_i = a·φ_i + b")

    # Simulates manual coefficient update per rule
    print("\n  Simulating coefficient update (rule 0):")
    rule_e.coeffs[0] = np.array([0.5, 0.1])   # [a, b]
    print(f"  rule[0]  a={rule_e.coeffs[0, 0]}  b={rule_e.coeffs[0, 1]}")

    # add_rule: verifies that new rule receives zeroed coefficients
    rule_e.add_rule(np.array([0.5, 0.5, 0.5]))
    print(f"\n  After add_rule:  coeffs shape={rule_e.coeffs.shape}")
    print(f"  New rule[3]  a={rule_e.coeffs[3, 0]}  b={rule_e.coeffs[3, 1]}  (zeros)")

    # remove_rule: verifies synchronization
    rule_e.remove_rule(1)
    print(f"\n  After remove_rule(1):  coeffs shape={rule_e.coeffs.shape}")
    print(f"  rule[0] preserved: a={rule_e.coeffs[0, 0]}  b={rule_e.coeffs[0, 1]}")

    print("\n" + "=" * 60)
    print("CASE F — update_rule(): complete decision structure")
    print("=" * 60)

    config_f = FuzzyRuleConfig(
        sigma        = 0.3,
        alpha        = 0.5,
        eta          = 0.1,
        modal_values = np.array([[0.3, 0.4, 0.6]]),
        coeff_init   = 0.0,
    )
    rule_f = FuzzyRule(config_f)
    # Sets non-zero initial coefficients for visualization of update
    rule_f.coeffs[0] = np.array([0.5, 0.1])   # [a, b]

    scenarios = [
        ("dist ≤ sigma     (case 1)", np.array([0.32, 0.42, 0.58]), 0.7),
        ("sigma<dist≤3σ   (case 2)", np.array([0.55, 0.60, 0.40]), 0.7),
        ("dist > 3sigma    (case 3)", np.array([0.95, 0.05, 0.90]), 0.7),
    ]

    for label, x_test, y_test in scenarios:
        rho, best = rule_f.compatibility(x_test)
        coeff_before = rule_f.coeffs[best].copy()
        modal_before = rule_f.modals[best].copy()

        result = rule_f.update_rule(x_test, y_test, best)

        print(f"\n  [{label}]")
        print(f"    x={x_test}   dist={result['dist']:.4f}")
        print(f"    action='{result['action']}'   error={result['error']:.4f}")
        print(f"    modal:  {modal_before.round(4)}  →  {rule_f.modals[result['best_idx']].round(4)}")
        print(f"    coeffs: {coeff_before.round(4)}  →  {rule_f.coeffs[result['best_idx']].round(4)}")
        print(f"    n_rules={rule_f.n_rules}  n_updates={rule_f.n_updates}")

    print("\n" + "=" * 60)
    print("CASE G — check_redundancy()")
    print("=" * 60)

    config_g = FuzzyRuleConfig(
        sigma        = 0.3,
        modal_values = np.array([[0.3, 0.4, 0.6],
                                 [0.35, 0.45, 0.55],   # close to [0] → redundant
                                 [0.9,  0.1,  0.8]]),  # distant
        coeff_init   = 0.0,
    )
    rule_g = FuzzyRule(config_g)
    rule_g.coeffs[0] = np.array([0.5, 0.1])
    rule_g.coeffs[1] = np.array([0.4, 0.2])
    rule_g.coeffs[2] = np.array([0.9, 0.5])

    print(f"\n  Before:  {rule_g}")
    print(f"  modals:\n{rule_g.modals}")

    result_g = rule_g.check_redundancy()

    print(f"\n  Result: {result_g}")
    print(f"  After: {rule_g}")
    print(f"  modals:\n{rule_g.modals}")
    print(f"  coeffs:\n{rule_g.coeffs}")

    print("\n" + "=" * 60)
    print("CASE H — defuzzify(): global output of the fuzzy system")
    print("=" * 60)

    config_h = FuzzyRuleConfig(
        sigma        = 0.3,
        modal_values = np.array([[0.325, 0.425, 0.575],
                                 [0.9,   0.1,   0.8  ]]),
    )
    rule_h = FuzzyRule(config_h)
    rule_h.coeffs[0] = np.array([0.45, 0.15])
    rule_h.coeffs[1] = np.array([0.90, 0.50])

    x_h = np.array([0.5, 0.4, 0.6])
    phi_h, _, best_h = rule_h.compute(x_h)
    y_out, f_local = rule_h.defuzzify(phi_h)

    print(f"\n  x^k      = {x_h}")
    print(f"  phi      = {phi_h.round(6)}  (best_idx={best_h})")
    print(f"  y_i^k    = {f_local.round(6)}  (y_i = a_i·φ_i + b_i)")
    print(f"  sum_phi  = {phi_h.sum():.6f}")
    print(f"  y_out    = {y_out:.6f}")
    print()
    print("  Manual verification:")
    for r in range(rule_h.n_rules):
        a_r, b_r = rule_h.coeffs[r]
        y_r = a_r * phi_h[r] + b_r
        print(f"    y_{r} = {a_r}·{phi_h[r]:.6f} + {b_r} = {y_r:.6f}")
    print(f"  y_out = ({phi_h[0]:.4f}·{f_local[0]:.4f} + "
          f"{phi_h[1]:.4f}·{f_local[1]:.4f}) / {phi_h.sum():.4f} = {y_out:.6f}")

