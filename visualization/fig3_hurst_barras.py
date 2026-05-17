"""
Figura 3 — Barras: H_DFA por commodity com IC 95% (bootstrap).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import CORES


def fig3_hurst_global(tabela: pd.DataFrame, savepath: str = None):
    """
    Barras com H_DFA e barras de erro (IC 95% bootstrap).

    A linha tracejada marca H = 0.5 (random walk / EMH fraca).
    A faixa cinza [0.45, 0.55] é a zona de indeterminação — valores
    nessa faixa não permitem distinguir persistência de eficiência dada
    a incerteza típica do DFA em amostras mensais de 35 anos.

    Os IC 95% são extraídos da coluna 'IC 95% DFA' da tabela.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    nomes = tabela.index.tolist()
    h_dfa = tabela["H (DFA)"].values
    cores = [CORES[n] for n in nomes]

    ic_lb = np.array([
        float(tabela.loc[n, "IC 95% DFA"].strip("[]").split(",")[0])
        for n in nomes
    ])
    ic_ub = np.array([
        float(tabela.loc[n, "IC 95% DFA"].strip("[]").split(",")[1])
        for n in nomes
    ])
    yerr = np.array([h_dfa - ic_lb, ic_ub - h_dfa])

    barras = ax.bar(nomes, h_dfa, color=cores, alpha=0.85,
                    edgecolor="white", linewidth=0.5)
    ax.errorbar(nomes, h_dfa, yerr=yerr, fmt="none",
                ecolor="black", elinewidth=1.2, capsize=4, capthick=1.2)

    ax.axhline(0.5, color="black", linewidth=1.2, linestyle="--",
               label="H = 0.5 (random walk)")
    ax.axhspan(0.45, 0.55, alpha=0.08, color="gray",
               label="Zona de indeterminação [0.45, 0.55]")

    for barra, h in zip(barras, h_dfa):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 0.012,
            f"{h:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_ylim(0.35, 0.85)
    ax.set_ylabel("Expoente de Hurst H (DFA)")
    ax.tick_params(axis="x", labelrotation=15)
    ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=200)
    return fig