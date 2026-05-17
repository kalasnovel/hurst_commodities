"""
Figura 4 — Mapa de regimes de persistência ao longo do tempo.

Exibe áreas coloridas por commodity indicando o regime corrente em cada
janela de JANELA_ROLL meses: persistente (H > 0.55), neutro ou anti-persistente.
Estimativas com H fora de [0, 1] são tratadas como inválidas e exibidas
em branco — não como persistência forte.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from config import CORES, EVENTOS, JANELA_ROLL
from analysis.rolling import hurst_rolling

LIMIAR_PERSISTENTE     = 0.55
LIMIAR_ANTIPERSISTENTE = 0.45

COR_PERSISTENTE     = "#2C7BB6"
COR_NEUTRO          = "#DDDDDD"
COR_ANTIPERSISTENTE = "#D7191C"
COR_INVALIDO        = "#FFFFFF"   # branco — estimativa inválida, não regime


def _classificar(h: float) -> str:
    if np.isnan(h) or h <= 0.0 or h >= 1.0:
        return "invalido"
    if h > LIMIAR_PERSISTENTE:
        return "persistente"
    if h < LIMIAR_ANTIPERSISTENTE:
        return "anti"
    return "neutro"


def fig4_mapa_regimes(retornos: pd.DataFrame, savepath: str = None):
    """
    Mapa de regimes por commodity ao longo do tempo.

    Azul:    persistente (H > 0.55)
    Cinza:   neutro (0.45 ≤ H ≤ 0.55)
    Vermelho: anti-persistente (H < 0.45)
    Branco:  estimativa inválida (H fora de [0,1] ou dados insuficientes)

    Nota metodológica: cada ponto no tempo reflete a composição dos
    retornos dos JANELA_ROLL meses anteriores — não o comportamento
    instantâneo do mercado. A coincidência com eventos históricos
    marcados deve ser interpretada como descritiva, não causal.
    """
    print(f"  Calculando H rolling para mapa de regimes "
          f"(janela {JANELA_ROLL} meses)...")

    cols = list(retornos.columns)
    n    = len(cols)

    fig, ax = plt.subplots(figsize=(14, n * 1.1 + 1.5))

    for i, col in enumerate(cols):
        h_roll = hurst_rolling(retornos[col].dropna(), janela=JANELA_ROLL)
        h_roll = h_roll.dropna()

        datas  = h_roll.index
        h_vals = h_roll.values

        for j in range(len(datas) - 1):
            regime = _classificar(h_vals[j])
            cor = {
                "persistente" : COR_PERSISTENTE,
                "neutro"      : COR_NEUTRO,
                "anti"        : COR_ANTIPERSISTENTE,
                "invalido"    : COR_INVALIDO,
            }[regime]

            x0 = mdates.date2num(datas[j])
            x1 = mdates.date2num(datas[j + 1])
            ax.barh(i, x1 - x0, left=x0, height=0.75,
                    color=cor, alpha=0.88, linewidth=0)

        ax.text(-0.01, i, col, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=9, fontweight="bold",
                color=CORES[col])

    for label, data_ev in EVENTOS.items():
        try:
            xv = mdates.date2num(pd.Timestamp(data_ev))
            ax.axvline(xv, color="black", linewidth=1.0,
                       linestyle=":", alpha=0.6)
            pass  # rótulos removidos — eventos descritos no texto do artigo
        except Exception:
            pass

    legenda = [
        mpatches.Patch(color=COR_PERSISTENTE,
                       label=f"Persistente (H > {LIMIAR_PERSISTENTE})"),
        mpatches.Patch(color=COR_NEUTRO, edgecolor="gray", linewidth=0.5,
                       label=f"Neutro ({LIMIAR_ANTIPERSISTENTE} ≤ H ≤ {LIMIAR_PERSISTENTE})"),
        mpatches.Patch(color=COR_ANTIPERSISTENTE,
                       label=f"Anti-persistente (H < {LIMIAR_ANTIPERSISTENTE})"),
        mpatches.Patch(color=COR_INVALIDO, edgecolor="lightgray", linewidth=0.8,
                       label="Estimativa inválida (H ∉ [0,1])"),
    ]
    ax.legend(
        handles=legenda,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.08),
        ncol=2,
        fontsize=8,
        framealpha=0.9,
        borderaxespad=0,
    )

    ax.set_yticks([])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.tick_params(axis="x", rotation=30)

    try:
        ax.set_xlim(
            mdates.date2num(retornos.index[JANELA_ROLL]),
            mdates.date2num(retornos.index[-1]),
        )
    except Exception:
        pass

    ax.set_ylim(-0.5, n - 0.25)
    ax.set_xlabel("Data")
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=200)
    return fig