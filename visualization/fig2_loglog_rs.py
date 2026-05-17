"""
Figura 2 — Log-log plot R/S: evidência visual da inclinação H.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import CORES, MIN_OBS_HURST


def fig2_loglog_rs(retornos: pd.DataFrame, savepath: str = None):
    """
    Cada painel mostra os pontos (log n, log R/S) e a reta ajustada,
    cuja inclinação é o estimador H_RS.
    A reta pontilhada azul indica H = 0.5 (random walk) como referência.

    Função diagnóstica: permite verificar se a relação log-linear é
    razoável e identificar desvios nas extremidades da escala.
    Não substitui o bootstrap para quantificação de incerteza.
    """
    cols  = list(retornos.columns)
    ncols = 2
    nrows = -(-len(cols) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 3.8))
    axes      = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        r  = retornos[col].dropna().values
        n  = len(r)

        lags = np.unique(
            np.floor(np.logspace(np.log10(4), np.log10(n // 2), 20)).astype(int)
        )

        rs_vals, lag_vals = [], []
        for lag in lags:
            rs_list = []
            for inicio in range(0, n - lag, lag):
                sub = r[inicio:inicio + lag]
                std = sub.std(ddof=1)
                if std == 0:
                    continue
                d = np.cumsum(sub - sub.mean())
                rs_list.append((d.max() - d.min()) / std)
            if rs_list:
                rs_vals.append(np.mean(rs_list))
                lag_vals.append(lag)

        if len(lag_vals) < 3:
            axes[i].set_visible(False)
            continue

        log_lag = np.log(lag_vals)
        log_rs  = np.log(rs_vals)
        slope, intercept = np.polyfit(log_lag, log_rs, 1)

        ax.scatter(log_lag, log_rs, color=CORES[col], s=28, zorder=3, alpha=0.85)
        ax.plot(log_lag, np.polyval([slope, intercept], log_lag),
                color="black", linewidth=1.2, linestyle="--",
                label=f"H = {slope:.3f}")
        # Referência H = 0.5 passando pelo primeiro ponto ajustado
        ref_b = intercept + (slope - 0.5) * log_lag[0]
        ax.plot(log_lag, 0.5 * log_lag + ref_b,
                color="steelblue", linewidth=0.9, linestyle=":",
                alpha=0.7, label="H = 0.5 (RW)")

        ax.set_xlabel("log(n)")
        ax.set_ylabel("log(R/S)")
        ax.set_title(col, fontweight="bold")
        ax.legend(fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=200)
    return fig