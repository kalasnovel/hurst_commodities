"""
Figura 5 — Heatmap de H (DFA) por período quinquenal × commodity.

Escala divergente centrada em 0.5 via TwoSlopeNorm:
  - vmin = 0.30 (anti-persistente)
  - vcenter = 0.50 (random walk — branco)
  - vmax = 1.00 (acomoda valores reais até ~0.96)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.cm as mcm
from config import EVENTOS
from estimators.dfa import hurst_dfa

MIN_OBS_HEATMAP = 48


def fig5_heatmap_regimes(retornos: pd.DataFrame, savepath: str = None):
    """
    Heatmap H (DFA) com commodities no eixo X e tempo no eixo Y.

    Escala divergente TwoSlopeNorm centrada em H = 0.5:
      Azul escuro  → H = 1.0 (persistente)
      Branco       → H = 0.5 (random walk)
      Vermelho     → H = 0.3 (anti-persistente)

    Células sem preenchimento: hachura cinza — distinto do branco (H = 0.5).
    """
    periodos = pd.date_range("1995-01-01", "2024-12-31", freq="5YS")
    labels   = [
        f"{periodos[j].year}–{periodos[j + 1].year - 1}"
        for j in range(len(periodos) - 1)
    ]
    cols   = list(retornos.columns)
    matriz = pd.DataFrame(index=labels, columns=cols, dtype=float)

    for col in cols:
        r = retornos[col].dropna()
        for j in range(len(periodos) - 1):
            sub = r.loc[periodos[j]:periodos[j + 1]]
            if len(sub) >= MIN_OBS_HEATMAP:
                h = hurst_dfa(sub.values)
                if 0.0 < h < 1.0:
                    matriz.loc[labels[j], col] = h

    matriz = matriz.iloc[::-1]   # mais recente no topo

    # Escala divergente centrada exatamente em 0.5
    norm = mcolors.TwoSlopeNorm(vmin=0.30, vcenter=0.50, vmax=1.00)
    cmap = mcm.RdBu   # vermelho < 0.5, branco = 0.5, azul > 0.5

    fig, ax = plt.subplots(figsize=(10, len(labels) * 0.9 + 2.5))

    for i in range(len(matriz.index)):
        for j in range(len(cols)):
            val = matriz.iloc[i, j]
            if np.isnan(val):
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=True, facecolor="#E8E8E8", edgecolor="white",
                    linewidth=1, hatch="////", zorder=1,
                ))
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=9, color="#AAAAAA", zorder=2)
            else:
                cor = cmap(norm(val))
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor=cor, edgecolor="white", linewidth=1, zorder=1,
                ))
                # Texto branco se afastado do centro, preto se próximo de 0.5
                dist = abs(val - 0.5)
                cor_txt = "white" if dist > 0.15 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=cor_txt, fontweight="bold", zorder=2)

    # Anotações de eventos
    evento_por_periodo = {}
    for ev_label, ev_data in EVENTOS.items():
        ev_ano = pd.Timestamp(ev_data).year
        for row_idx, periodo_label in enumerate(matriz.index):
            anos = periodo_label.split("–")
            if int(anos[0]) <= ev_ano <= int(anos[1]):
                evento_por_periodo.setdefault(row_idx, []).append(
                    ev_label.replace("\n", " ")
                )

    for row_idx, ev_list in evento_por_periodo.items():
        ax.annotate(
            " · ".join(ev_list),
            xy=(len(cols) - 0.5, row_idx),
            xytext=(len(cols) - 0.3, row_idx),
            fontsize=7, va="center", color="#555555",
            arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.8),
        )

    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(-0.5, len(matriz.index) - 0.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(matriz.index)))
    ax.set_yticklabels(matriz.index, fontsize=9)

    # Legenda para NA
    na_patch = mpatches.Patch(
        facecolor="#E8E8E8", hatch="////", edgecolor="gray",
        label="Dados insuficientes ou estimativa inválida"
    )
    ax.legend(handles=[na_patch], loc="upper left", fontsize=8, framealpha=0.9)

    # Colorbar com TwoSlopeNorm — eixo explícito
    cax = fig.add_axes([0.92, 0.12, 0.02, 0.76])
    cb  = mcolorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_label("H (DFA)", fontsize=10)
    cb.ax.axhline(0.5, color="black", linewidth=1.5, linestyle="--")
    # Ticks explícitos para cobrir a escala completa
    cb.set_ticks([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])

    fig.tight_layout(rect=[0, 0, 0.91, 1])
    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=200)
    return fig