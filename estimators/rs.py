"""
Estimador R/S (Rescaled Range) do Expoente de Hurst.

Referências:
    Hurst, H. E. (1951). Long-term storage capacity of reservoirs.
    Transactions of the American Society of Civil Engineers, 116, 770–799.

    Mandelbrot, B. B.; Wallis, J. R. (1969). Computer experiments with
    fractional Gaussian noises. Water Resources Research, 5(1), 228–267.

Limitação conhecida:
    O R/S clássico não controla autocorrelações de curto prazo, o que
    pode inflar H. Lo (1991) propôs o R/S modificado com correção de
    Newey-West — não implementado aqui. O DFA é o estimador principal
    por ser mais robusto a esse problema.
"""

import numpy as np
from config import MIN_OBS_HURST


def hurst_rs(serie: np.ndarray, min_lags: int = 4) -> float:
    """
    Estima H pelo método R/S clássico.

    H < 0.5 → anti-persistência (reversão à média mais rápida que RW)
    H = 0.5 → random walk (EMH fraca não rejeitada)
    H > 0.5 → persistência (memória longa)

    Parâmetros:
      serie    — array 1D de retornos logarítmicos
      min_lags — lag mínimo para o ajuste log-log (padrão: 4)

    Retorna:
      H estimado, ou np.nan se observações insuficientes.
    """
    n = len(serie)
    if n < MIN_OBS_HURST:
        return np.nan

    lags = np.unique(
        np.floor(np.logspace(np.log10(min_lags), np.log10(n // 2), 20)).astype(int)
    )

    rs_vals  = []
    lag_vals = []

    for lag in lags:
        rs_list = []
        for inicio in range(0, n - lag, lag):
            sub     = serie[inicio:inicio + lag]
            media   = sub.mean()
            desvpad = sub.std(ddof=1)
            if desvpad == 0:
                continue
            desvios = np.cumsum(sub - media)
            rs_list.append((desvios.max() - desvios.min()) / desvpad)
        if rs_list:
            rs_vals.append(np.mean(rs_list))
            lag_vals.append(lag)

    if len(lag_vals) < 3:
        return np.nan

    slope, *_ = np.polyfit(np.log(lag_vals), np.log(rs_vals), 1)
    return float(slope)
