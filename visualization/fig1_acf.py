"""
Figura 1 — Autocorrelação dos retornos logarítmicos mensais (ACF).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from config import CORES


def fig1_acf(retornos: pd.DataFrame, n_lags: int = 24, savepath: str = None):
    """
    Painel de ACF por commodity (lags 1–24 meses).

    A banda de confiança de 95% (±1.96/√N) serve como referência visual
    para ruído branco. Barras fora da banda sugerem autocorrelação
    estatisticamente diferente de zero — mas não são teste formal de
    memória longa (papel do R/S e DFA).

    Lags até 24 meses (2 anos) são suficientes para revelar padrões
    de curto e médio prazo sem perda de resolução visual.
    """
    cols  = list(retornos.columns)
    ncols = 2
    nrows = -(-len(cols) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 3.2))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        r  = retornos[col].dropna()
        n  = len(r)

        acf_vals, confint = acf(r, nlags=n_lags, alpha=0.05, fft=True)

        lags     = np.arange(1, n_lags + 1)
        acf_plot = acf_vals[1:]          # exclui lag 0 (sempre 1)
        lb       = confint[1:, 0] - acf_vals[1:]
        ub       = confint[1:, 1] - acf_vals[1:]

        # Banda de confiança simplificada ±1.96/√N
        banda = 1.96 / np.sqrt(n)

        # Barras coloridas pela commodity, com linha de referência em zero
        cores_barras = [
            CORES[col] if abs(v) > banda else "#CCCCCC"
            for v in acf_plot
        ]
        ax.bar(lags, acf_plot, color=cores_barras, alpha=0.85, width=0.6)
        ax.axhline(0,      color="black",     linewidth=0.8)
        ax.axhline( banda, color="steelblue", linewidth=0.9,
                   linestyle="--", alpha=0.7, label="IC 95% (±1.96/√N)")
        ax.axhline(-banda, color="steelblue", linewidth=0.9,
                   linestyle="--", alpha=0.7)

        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("Lag (meses)")
        ax.set_ylabel("Autocorrelação")
        ax.set_xlim(0.5, n_lags + 0.5)
        ax.set_ylim(-0.35, 0.35)
        ax.set_xticks(range(2, n_lags + 1, 2))
        ax.legend(fontsize=8, loc="upper right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=200)
    return fig