#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:51:29 2026

@author: lqsoliveira


=========
Configuração global do algoritmo.

Este arquivo é o único ponto de agregação de todas as sub-configs.
Nenhum módulo operacional (normalizer, fuzzy_rule, ...) importa outro —
todos importam apenas deste arquivo, evitando importações circulares.

Estrutura de dependências:
    config.py
        ↑               ↑
    normalizer.py   fuzzy_rule.py   (importam só config.py)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from normalizer  import NormalizerConfig
from fuzzy_rule  import FuzzyRuleConfig


@dataclass
class AlgorithmConfig:
    """
    Configuração global do algoritmo.
    Agrega todas as sub-configs de cada módulo.

    Atributos
    ----------
    normalizer : NormalizerConfig
        Configuração do módulo de normalização.
    fuzzy_rule : FuzzyRuleConfig
        Configuração do módulo de ativação de regra fuzzy.
    seed : int
        Semente global para reprodutibilidade.
    dtype : np.dtype
        Tipo numérico padrão propagado para todos os módulos.
        Pode ser sobrescrito individualmente em cada sub-config.
    """
    normalizer : NormalizerConfig = field(default_factory=NormalizerConfig)
    fuzzy_rule : FuzzyRuleConfig  = field(default_factory=FuzzyRuleConfig)
    seed       : int              = 42
    dtype      : np.dtype         = np.float64

    def __post_init__(self) -> None:
        # Propaga dtype global para sub-configs que não o definiram explicitamente
        if self.normalizer.dtype == np.float64 and self.dtype != np.float64:
            self.normalizer.dtype = self.dtype


# ─────────────────────────────────────────────
# EXEMPLO DE USO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    from normalizer import Normalizer
    from fuzzy_rule import FuzzyRule

    # ── Configuração única e centralizada ──
    cfg = AlgorithmConfig(
        normalizer = NormalizerConfig(
            min_vals = np.array([0.0, 0.0,   0.0]),
            max_vals = np.array([5.0, 6.0, 200.0]),
        ),
        fuzzy_rule = FuzzyRuleConfig(
            sigma        = 0.3,
            modal_values = None,   # primeiro dado define o modal
            n_antecedents = 3,
        ),
        seed  = 42,
        dtype = np.float64,
    )

    # ── Instancia os módulos a partir da config central ──
    norm = Normalizer(cfg.normalizer)
    norm.fit(np.array([[0.0, 0.0, 0.0]]))

    rule = FuzzyRule(cfg.fuzzy_rule)

    # ── Loop iterativo ──
    dados = [
        np.array([[2.5,  3.0, 100.0]]),   # k=1: dentro dos limites
        np.array([[2.5,  6.0, 250.0]]),   # k=2: x[2] fora do limite
    ]

    print(f"{'k':>3}  {'x_bruto':>22}  {'x_norm':>18}  {'phi':>8}")
    print("-" * 60)
    for k, x_bruto in enumerate(dados, start=1):
        norm.partial_fit(x_bruto)
        x_norm     = norm.transform(x_bruto).ravel()
        phi, mu    = rule.compute(x_norm)
        print(f"{k:>3}  {str(x_bruto[0]):>22}  {str(x_norm.round(3)):>18}  {phi:>8.6f}")

