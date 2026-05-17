"""
Pré-processamento das séries de preços.

Nota metodológica:
    R/S e DFA são aplicados sobre RETORNOS logarítmicos mensais —
    O objeto de interesse é a memória da série de variações de preço,
    não o nível ou o caminho cumulativo dos preços.
    Retornos são estacionários em primeiro momento (ADF rejeita raiz
    unitária), condição relevante para a validade do R/S clássico.
"""

import numpy as np
import pandas as pd


def preprocessar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retorna:
      precos   — preços nominais em USD (forward-fill de até 2 meses ausentes)
      retornos — retornos logarítmicos mensais: r_t = ln(P_t / P_{t-1})
    """
    precos   = df.copy().ffill(limit=2)
    retornos = np.log(precos / precos.shift(1)).dropna()
    return precos, retornos
