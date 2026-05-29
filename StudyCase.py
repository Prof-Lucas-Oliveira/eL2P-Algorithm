#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lqsoliveira

===================================================================
Full comparison: TS-4 vs eL2P — Fault Condition

Training:
    TS-4 : 250 steps of the reference signal (Regime 1)
    eL2P : 250 steps reference (Regime 1) + 250 steps chirp

Validation: k = 1..2000  |  Plant change at k=500
    k ≤ 500 : original plant   y(k+1) =   y*u/(1+y²) - tan(u)   γ=1
    k > 500 : modified plant   y(k+1) = 2*y*u/(1+y²) - tan(u)   γ=2

Performance indices per interval (RMSE, IAE, IVU):
    I1:   0 < k ≤  500  — γ=1
    I2: 500 < k ≤ 1000  — γ=2 transient
    I3:1000 < k ≤ 2000  — γ=2 adapted

Plot 2x1:
    Top    : reference (black dotted), TS-4 (red solid),
             eL2P (blue dashed)
    Bottom : control signals u(k) — same colour/line styles

Dependencies: normalizer.py  fuzzy_rule.py  config.py
              numpy  scipy  matplotlib  scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from sklearn.cluster import KMeans
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config     import AlgorithmConfig
from normalizer import NormalizerConfig, Normalizer
from fuzzy_rule import FuzzyRuleConfig, FuzzyRule


# ═══════════════════════════════════════════════════════════════
# 1. PLANT, INVERSE AND REFERENCE
# ═══════════════════════════════════════════════════════════════

def plant_orig(y: float, u: float) -> float:
    """Original plant γ=1."""
    u = np.clip(u, -1.39, 1.39)
    return float(y * u / (1.0 + y**2) - np.tan(u))


def plant_mod(y: float, u: float) -> float:
    """Modified plant γ=2."""
    u = np.clip(u, -1.39, 1.39)
    return float(2.0 * y * u / (1.0 + y**2) - np.tan(u))


def plant_k(y: float, u: float, k: int) -> float:
    """Active plant at time step k."""
    return plant_orig(y, u) if k <= 500 else plant_mod(y, u)


def inv_orig(y: float, yd: float) -> float:
    """Exact inverse of the original plant."""
    try:
        f = lambda u: plant_orig(y, u) - yd
        if f(-1.39) * f(1.39) > 0: return 0.0
        return float(brentq(f, -1.39, 1.39, xtol=1e-9))
    except Exception: return 0.0


def inv_mod(y: float, yd: float) -> float:
    """Exact inverse of the modified plant."""
    try:
        f = lambda u: plant_mod(y, u) - yd
        if f(-1.39) * f(1.39) > 0: return 0.0
        return float(brentq(f, -1.39, 1.39, xtol=1e-9))
    except Exception: return 0.0


def inv_k(y: float, yd: float, k: int) -> float:
    """Exact inverse of the active plant — supervision signal."""
    return inv_orig(y, yd) if k <= 500 else inv_mod(y, yd)


def reference(k: int) -> float:
    """Desired output y_d(k+1) — continuous beyond k=1000."""
    if k <= 500:
        return 0.6*np.sin(2*np.pi*k/250) + 0.2*np.sin(2*np.pi*k/50)
    return 0.3*np.sin(2*np.pi*k/55) + 0.1*np.sin(2*np.pi*k/105)


def chirp(k: int, N: int = 250,
          f0: float = 1/250, f1: float = 1/30,
          A: float = 0.75) -> float:
    """Chirp signal for eL2P training diversification."""
    phase = 2 * np.pi * k * (f0 + (f1 - f0) * k / (2 * N))
    return float(np.clip(A * np.sin(phase), -0.84, 0.84))


# ═══════════════════════════════════════════════════════════════
# 2. NORMALISATION
# ═══════════════════════════════════════════════════════════════

MIN_B = np.array([-0.85, -1.50, -1.0])
MAX_B = np.array([ 0.85,  1.50,  1.0])


def normalize(x: np.ndarray) -> np.ndarray:
    return np.clip((x - MIN_B) / (MAX_B - MIN_B), 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════
# 3. TRAINING AND SIMULATION — TS-4
# ═══════════════════════════════════════════════════════════════

def treino_ts4(n_train: int = 250,
               verbose: bool = True) -> tuple:
    """
    Offline training of TS-4.
    K-Means (4 clusters) + LS per cluster.
    Regressors: z = [y_d(k+1), y(k), 1]
    """
    y, u = 0.0, 0.0
    X_norm, X_raw, U = [], [], []

    for k in range(1, n_train + 1):
        yd  = reference(k)
        u_s = inv_orig(y, yd)
        xr  = np.array([yd, u_s, y])
        X_norm.append(normalize(xr))
        X_raw.append(xr)
        U.append(u_s)
        y = plant_orig(y, u_s)
        u = u_s

    X_norm = np.array(X_norm)
    X_raw  = np.array(X_raw)
    U      = np.array(U)

    km      = KMeans(n_clusters=4, random_state=42, n_init=20)
    km.fit(X_norm)
    centers = km.cluster_centers_

    coeffs = np.zeros((4, 3))
    for r in range(4):
        mask = km.labels_ == r
        Zr   = np.hstack([X_raw[mask][:, [0, 2]],
                          np.ones((mask.sum(), 1))])
        if mask.sum() >= 3:
            coeffs[r] = np.linalg.lstsq(Zr, U[mask], rcond=None)[0]

    if verbose:
        print(f"  TS-4 training: {n_train} steps  →  "
              f"4 rules  12 params")
        print(f"  Samples/cluster: "
              f"{[(km.labels_==r).sum() for r in range(4)]}")

    return centers, coeffs


def ts4_predict(x_norm: np.ndarray,
                x_raw:  np.ndarray,
                centers: np.ndarray,
                coeffs:  np.ndarray,
                sigma:   float = 0.050) -> float:
    """TS-4 inference: Gaussians + t-norm + defuzzification."""
    phi = np.prod(
        np.exp(-0.5 * ((x_norm - centers) / sigma)**2), axis=1)
    f_r = coeffs @ np.array([x_raw[0], x_raw[2], 1.0])
    s   = phi.sum()
    return float((phi @ f_r) / s) if s > 1e-12 \
           else float(f_r[np.argmax(phi)])


def simular_ts4(centers: np.ndarray,
                coeffs:  np.ndarray,
                n_sim:   int = 2000,
                sigma:   float = 0.050,
                verbose: bool = True) -> tuple:
    """Closed-loop simulation with static TS-4 (no update)."""
    y, u = 0.0, 0.0
    Y  = np.zeros(n_sim + 1)
    YD = np.zeros(n_sim + 1)
    U  = np.zeros(n_sim + 1)

    Y[0]  = y
    YD[0] = reference(0)
    U[0]  = u

    for k in range(1, n_sim + 1):
        yd_k1 = reference(k)
        xr    = np.array([yd_k1, u, y])
        xn    = normalize(xr)
        u_hat = float(np.clip(
            ts4_predict(xn, xr, centers, coeffs, sigma),
            -1.39, 1.39))
        y_k1  = plant_k(y, u_hat, k)

        Y[k]  = y_k1
        YD[k] = yd_k1
        U[k]  = u_hat

        y = y_k1
        u = u_hat

    if verbose:
        e = Y[1:] - YD[1:]
        print(f"\n  TS-4 simulation: {n_sim} steps")
        print(f"  RMSE I1={np.sqrt(np.mean(e[:500]**2)):.4f}  "
              f"I2={np.sqrt(np.mean(e[500:1000]**2)):.4f}  "
              f"I3={np.sqrt(np.mean(e[1000:]**2)):.4f}")

    return Y, YD, U


# ═══════════════════════════════════════════════════════════════
# 4. TRAINING AND SIMULATION — eL2P
# ═══════════════════════════════════════════════════════════════

cfg = AlgorithmConfig(
    normalizer = NormalizerConfig(
        min_vals=MIN_B, max_vals=MAX_B,
        clip=True, handle_constant="zero"),
    fuzzy_rule = FuzzyRuleConfig(
        sigma=0.010, alpha=0.9, eta=0.9,
        n_antecedents=3, coeff_init=0.0),
)


def make_norm() -> Normalizer:
    n = Normalizer(cfg.normalizer)
    n.fit(np.zeros((1, 3)))
    return n


def clone_rule(src: FuzzyRule) -> FuzzyRule:
    dst = FuzzyRule(cfg.fuzzy_rule)
    dst._modals            = src.modals.copy()
    dst._coeffs            = src.coeffs.copy()
    dst._n_updates         = src.n_updates.copy()
    dst._is_initialized    = True
    dst._coeff_initialized = True
    return dst


def treino_el2p(n_train: int = 250,
                verbose: bool = True) -> tuple:
    """
    Offline training of eL2P — Strategy A+B (n_train + n_train steps).
    Part A: n_train steps of the reference signal (Regime 1)
    Part B: n_train steps of the chirp signal
    """
    norm = make_norm()
    rule = FuzzyRule(cfg.fuzzy_rule)
    y, u = 0.0, 0.0

    # ── Part A: reference signal ──
    for k in range(1, n_train + 1):
        yd    = reference(k)
        u_s   = inv_orig(y, yd)
        x_raw = np.array([yd, u_s, y]).reshape(1, -1)
        rule.eL2P(x_raw, u_s, norm)
        y = plant_orig(y, u_s)
        u = u_s
    n_after_A = rule.n_rules

    # ── Part B: chirp signal ──
    for k in range(1, n_train + 1):
        yd    = chirp(k, N=n_train)
        u_s   = inv_orig(y, yd)
        x_raw = np.array([yd, u_s, y]).reshape(1, -1)
        rule.eL2P(x_raw, u_s, norm)
        y = plant_orig(y, u_s)
        u = u_s
    n_after_B = rule.n_rules

    if verbose:
        print(f"\n  eL2P training: {n_train} ref + {n_train} chirp")
        print(f"  Rules after A: {n_after_A}  →  after A+B: {n_after_B}"
              f"  ({n_after_B*4} params)")

    return rule, norm, n_after_A, n_after_B


def simular_el2p(rule_trained: FuzzyRule,
                 norm_trained: Normalizer,
                 n_sim:   int = 2000,
                 verbose: bool = True) -> tuple:
    """Closed-loop simulation with adaptive eL2P (continuous learning)."""
    rule_cl = clone_rule(rule_trained)
    y, u = 0.0, 0.0

    Y  = np.zeros(n_sim + 1)
    YD = np.zeros(n_sim + 1)
    U  = np.zeros(n_sim + 1)

    Y[0]  = y
    YD[0] = reference(0)
    U[0]  = u

    for k in range(1, n_sim + 1):
        yd_k1  = reference(k)
        u_star = inv_k(y, yd_k1, k)

        # prediction
        x_norm = np.clip(
            (np.array([yd_k1, u, y]) - MIN_B) / (MAX_B - MIN_B),
            0.0, 1.0)
        phi, _, _ = rule_cl.compute(x_norm)
        u_hat, _  = rule_cl.defuzzify(phi)
        u_hat     = float(np.clip(u_hat, -1.39, 1.39))

        y_k1 = plant_k(y, u_hat, k)

        # continuous learning
        x_train = np.array([yd_k1, u_hat, y]).reshape(1, -1)
        rule_cl.eL2P(x_train, u_star, norm_trained)

        Y[k]  = y_k1
        YD[k] = yd_k1
        U[k]  = u_hat

        y = y_k1
        u = u_hat

    n_final = rule_cl.n_rules

    if verbose:
        e = Y[1:] - YD[1:]
        print(f"\n  eL2P simulation: {n_sim} steps")
        print(f"  Final rules: {n_final}")
        print(f"  RMSE I1={np.sqrt(np.mean(e[:500]**2)):.4f}  "
              f"I2={np.sqrt(np.mean(e[500:1000]**2)):.4f}  "
              f"I3={np.sqrt(np.mean(e[1000:]**2)):.4f}")

    return Y, YD, U, n_final


# ═══════════════════════════════════════════════════════════════
# 5. PERFORMANCE INDICES
# ═══════════════════════════════════════════════════════════════

def calcular_indices(Y: np.ndarray,
                     YD: np.ndarray,
                     U:  np.ndarray) -> dict:
    """Computes RMSE, IAE and IVU over the three evaluation intervals."""
    e = Y[1:] - YD[1:]
    m = {}
    for lbl, a, b in [('I1',0,500),('I2',500,1000),('I3',1000,2000)]:
        m[f'RMSE_{lbl}'] = float(np.sqrt(np.mean(e[a:b]**2)))
        m[f'IAE_{lbl}']  = float(np.sum(np.abs(e[a:b])))
        m[f'IVU_{lbl}']  = float(np.sum(np.abs(np.diff(U[a:b+1]))))
    return m


def imprimir_tabela(m_ts: dict, m_el: dict,
                    n_rules_ts: int, n_rules_el: int,
                    n_params_ts: int, n_params_el: int) -> None:
    """Prints the comparative performance index table."""
    sep = "=" * 76
    print(f"\n{sep}")
    print("  PERFORMANCE INDICES — Training: TS-4=250 ref | "
          "eL2P=250 ref + 250 chirp")
    print(f"  Validation: k=1..2000  |  Fault: γ: 1→2 at k=500")
    print(sep)
    print(f"  {'Controller':<22}  {'Interval':<14}  {'Plant':<11}  "
          f"{'RMSE':>8}  {'IAE':>8}  {'IVU':>8}")
    print("  " + "-"*72)

    intervals = [
        ('I1', '0<k≤500',    'γ=1       '),
        ('I2', '500<k≤1000', 'γ=2 trans.'),
        ('I3', '1000<k≤2000','γ=2 adapt.'),
    ]

    for ctrl, m, rules, params in [
            ("TS-4 (static)",    m_ts, n_rules_ts, n_params_ts),
            ("eL2P (adaptive)",  m_el, n_rules_el, n_params_el)]:
        for key, intv, planta in intervals:
            print(f"  {ctrl:<22}  {intv:<14}  {planta:<11}  "
                  f"{m[f'RMSE_{key}']:>8.4f}  "
                  f"{m[f'IAE_{key}']:>8.3f}  "
                  f"{m[f'IVU_{key}']:>8.3f}")
        print(f"  {'':22}  {'Rules:':>14}  {str(rules):<11}  "
              f"{'Params:':>8}  {params}")
        print("  " + "-"*72)
    print(sep)


# ═══════════════════════════════════════════════════════════════
# 6. PLOT
# ═══════════════════════════════════════════════════════════════

def plotar(YD: np.ndarray,
           Y_ts: np.ndarray, U_ts: np.ndarray,
           Y_el: np.ndarray, U_el: np.ndarray,
           m_ts: dict, m_el: dict,
           n_rules_ts: int, n_rules_el_train: int,
           n_rules_el_final: int) -> None:
    """
    2x1 figure:
        Top    : reference (k--), TS-4 (r-), eL2P (b--)
        Bottom : control signals — same colour/line styles
    """
    k_axis = np.arange(len(YD))
    shade  = [(0, 500, 'green'), (500, 1000, 'red'), (1000, 2001, 'orange')]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # ── Top subplot: plant output ─────────────────────────────
    ax1 = axes[0]
    for a, b, c in shade:
        ax1.axvspan(a, b, alpha=0.05, color=c)

    ax1.plot(k_axis, YD, 'k:',  lw=1.6,
             label=r'$y_d^{k+1}$ (reference)')
    ax1.plot(k_axis, Y_ts, 'r-',  lw=1.0, alpha=0.90,
             label=f'TS-4  (4 rules  12 params)')
    ax1.plot(k_axis, Y_el, 'b--', lw=1.0, alpha=0.90,
             label=f'eL2P  '
                   f'({n_rules_el_train}→{n_rules_el_final} rules  '
                   f'{n_rules_el_final*4} params)')

    ax1.axvline(500,  color='red',  ls='--', lw=1.1,
                label=r'fault $\gamma: 1 \to 2$')
    ax1.axvline(1000, color='gray', ls=':',  lw=0.9)

    ax1.set_ylabel(r'$y^{k+1}$', fontsize=11)

    ax1.grid(True, alpha=0.22)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_xlim(0, 2000)

    # RMSE annotations per interval
    for xm, key in [(250, 'I1'), (750, 'I2'), (1500, 'I3')]:
        ax1.text(xm, -0.97, f"RMSE\n"
                 f"TS4: {m_ts[f'RMSE_{key}']:.4f}\n"
                 f"eL2P: {m_el[f'RMSE_{key}']:.4f}",
                 ha='center', fontsize=7,
                 bbox=dict(boxstyle='round,pad=0.25',
                           fc='white', alpha=0.75))

    # ── Bottom subplot: control signal ────────────────────────
    ax2 = axes[1]
    for a, b, c in shade:
        ax2.axvspan(a, b, alpha=0.05, color=c)

    ax2.plot(k_axis, U_ts, 'r-',  lw=1.0, alpha=0.90,
             label=r'$\hat{u}^k$ TS-4')
    ax2.plot(k_axis, U_el, 'b--', lw=1.0, alpha=0.90,
             label=r'$\hat{u}^k$ eL2P')

    ax2.axvline(500,  color='red',  ls='--', lw=1.1)
    ax2.axvline(1000, color='gray', ls=':',  lw=0.9)
    ax2.set_xlim(0, 2000)
    ax2.set_ylabel(r'$u^k$', fontsize=11)
    ax2.set_xlabel('$k$', fontsize=11)

    ax2.grid(True, alpha=0.22)

    # Interval legend patches
    from matplotlib.patches import Patch
    leg_patches = [
        Patch(color='green',  alpha=0.25, label=r'$\mathcal{I}_1$: γ=1'),
        Patch(color='red',    alpha=0.25, label=r'$\mathcal{I}_2$: γ=2 trans.'),
        Patch(color='orange', alpha=0.25, label=r'$\mathcal{I}_3$: γ=2 adapt.'),
    ]

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    N_TRAIN = 250
    N_SIM   = 2000
    SEP     = "=" * 60

    print(SEP)
    print("ILSC COMPARISON — TS-4 vs eL2P")
    print(SEP)
    print(f"  Training:   {N_TRAIN} steps ref (TS-4) | "
          f"{N_TRAIN} ref + {N_TRAIN} chirp (eL2P)")
    print(f"  Validation: k=1..{N_SIM}  |  Fault: γ: 1→2 at k=500\n")

    # ── Phase 1: Training ────────────────────────────────────
    print("PHASE 1 — OFFLINE TRAINING")
    print("-" * 40)

    centers, coeffs = treino_ts4(n_train=N_TRAIN, verbose=True)

    rule_trained, norm_trained, n_after_A, n_after_B = \
        treino_el2p(n_train=N_TRAIN, verbose=True)

    # ── Phase 2: Simulation ──────────────────────────────────
    print(f"\nPHASE 2 — SIMULATION  (k=1..{N_SIM})")
    print("-" * 40)

    Y_ts, YD_ts, U_ts = simular_ts4(
        centers, coeffs, n_sim=N_SIM, verbose=True)

    Y_el, YD_el, U_el, n_final_el = simular_el2p(
        rule_trained, norm_trained, n_sim=N_SIM, verbose=True)

    # ── Phase 3: Performance indices ─────────────────────────
    YD = np.array([reference(k) for k in range(N_SIM + 1)])

    m_ts = calcular_indices(Y_ts, YD, U_ts)
    m_el = calcular_indices(Y_el, YD, U_el)

    imprimir_tabela(
        m_ts, m_el,
        n_rules_ts=4,          n_rules_el=n_after_B,
        n_params_ts=12,        n_params_el=n_final_el * 4)

    # ── Phase 4: Plot ────────────────────────────────────────
    plotar(YD, Y_ts, U_ts, Y_el, U_el,
           m_ts, m_el,
           n_rules_ts=4,
           n_rules_el_train=n_after_B,
           n_rules_el_final=n_final_el)