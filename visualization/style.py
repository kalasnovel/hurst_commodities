"""
Estilo global para figuras.
"""

import matplotlib.pyplot as plt


def aplicar_estilo():
    plt.rcParams.update({
        "figure.dpi"       : 150,
        "figure.facecolor" : "white",
        "axes.facecolor"   : "white",
        "axes.grid"        : True,
        "grid.alpha"       : 0.3,
        "grid.linestyle"   : "--",
        "font.family"      : "serif",
        "font.size"        : 11,
        "axes.titlesize"   : 12,
        "axes.labelsize"   : 11,
        "legend.fontsize"  : 10,
        "xtick.labelsize"  : 9,
        "ytick.labelsize"  : 9,
    })
