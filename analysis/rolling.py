"""
Análise rolling do Expoente de Hurst em janela deslizante.
"""

import numpy as np
import pandas as pd
from config import JANELA_ROLL
from estimators.dfa import hurst_dfa
from estimators.rs  import hurst_rs

# Janelas de robustez — 60 meses é o padrão; 84 e 120 meses são
# alternativas mais conservadoras com estimativas mais estáveis
# e menos sujeitas a extrapolação fora do intervalo [0, 1].
JANELAS_ROBUSTEZ = [60, 84, 120]


def hurst_rolling(
    serie: pd.Series,
    janela: int = JANELA_ROLL,
    metodo: str = "dfa",
) -> pd.Series:
    """
    Calcula H em janela deslizante de `janela` meses.

    Estimativas com H fora de [0, 1] são descartadas (substituídas por NaN)
    — valores nesse intervalo indicam instabilidade da estimação por
    escalas insuficientes, não persistência ou anti-persistência genuína.

    As primeiras `janela` posições são NaN (aquecimento da janela).
    """
    fn     = hurst_rs if metodo == "rs" else hurst_dfa
    h_vals = [np.nan] * len(serie)

    for i in range(janela, len(serie) + 1):
        sub = serie.iloc[i - janela:i].values
        h   = fn(sub)
        # Descartar estimativas fora do intervalo teórico
        h_vals[i - 1] = h if (not np.isnan(h) and 0.0 < h < 1.0) else np.nan

    return pd.Series(h_vals, index=serie.index)


def hurst_rolling_all(
    retornos: pd.DataFrame,
    janela: int = JANELA_ROLL,
    metodo: str = "dfa",
) -> pd.DataFrame:
    """Aplica hurst_rolling a todas as commodities para uma janela."""
    resultado = {}
    for col in retornos.columns:
        resultado[col] = hurst_rolling(retornos[col].dropna(), janela, metodo)
    return pd.DataFrame(resultado)


def hurst_rolling_robustez(
    retornos: pd.DataFrame,
    janelas: list = JANELAS_ROBUSTEZ,
    metodo: str   = "dfa",
) -> dict[int, pd.DataFrame]:
    """
    Calcula H rolling para múltiplas janelas — análise de robustez.

    Retorna dicionário {janela: DataFrame} para comparação visual
    da estabilidade das estimativas em função da janela escolhida.

    Janelas maiores produzem estimativas mais estáveis mas menos
    sensíveis a mudanças de regime recentes.
    """
    return {j: hurst_rolling_all(retornos, j, metodo) for j in janelas}
