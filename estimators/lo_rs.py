"""
Estimador R/S Modificado de Lo (1991).

Referência:
    Lo, A. W. (1991). Long-term memory in stock market prices.
    Econometrica, 59(5), 1279–1313.

Diferença em relação ao R/S clássico:
    O R/S clássico divide o range pelos desvio-padrão simples da amostra,
    o que infla H na presença de autocorrelações de curto prazo.
    O R/S modificado substitui o desvio-padrão por um estimador de
    longo prazo que incorpora as autocovariâncias até o lag q (correção
    de Newey-West), tornando-o robusto a dependências de curto prazo.

Escolha de q:
    A regra padrão é q = floor(T^(1/3)), o mesmo usado no bootstrap.
    Alternativas comuns: q = 4, q = 8, q = 12 (mensal).
    O estimador é sensível a q — variações devem ser reportadas
    como análise de robustez.

Interpretação:
    O resultado não é diretamente comparável ao H do R/S clássico.
    Lo (1991) propõe estatística de teste normalizada (Vq) com
    distribuição assintótica conhecida sob H0: sem memória longa.
    Este módulo retorna tanto H_Lo (inclinação log-log) quanto
    a estatística Vq para inferência.
"""

import numpy as np
from config import MIN_OBS_HURST


def _sigma_lo(serie: np.ndarray, q: int) -> float:
    """
    Estimador de longo prazo de Lo (1991) com correção de Newey-West.
    σ²(q) = s² + 2 * Σ_{j=1}^{q} w_j * γ_j
    onde w_j = 1 − j/(q+1) são os pesos de Bartlett.
    """
    n    = len(serie)
    s2   = serie.var(ddof=1)
    soma = 0.0
    for j in range(1, q + 1):
        w_j   = 1.0 - j / (q + 1)
        gamma = np.mean((serie[:n - j] - serie.mean()) *
                        (serie[j:]   - serie.mean()))
        soma += w_j * gamma
    return np.sqrt(max(s2 + 2 * soma, 1e-12))


def hurst_lo_rs(serie: np.ndarray, q: int = None,
                min_lags: int = 4) -> dict:
    """
    Expoente de Hurst pelo R/S Modificado de Lo (1991).

    Parâmetros:
      serie    — array 1D de retornos logarítmicos
      q        — lags para correção Newey-West.
                 Se None, usa q = floor(N^(1/3)), mínimo 4.
      min_lags — lag mínimo para o ajuste log-log (padrão: 4)

    Retorna dicionário com:
      H_lo   — inclinação do ajuste log-log (análogo ao H do R/S clássico)
      Vq     — estatística de teste de Lo (1991)
      q_used — valor de q utilizado
      rejeita_H0 — bool: Vq fora do IC 95% assintótico [0.809, 1.862]
                   (H0: sem memória longa)
    """
    n = len(serie)
    if n < MIN_OBS_HURST:
        return {"H_lo": np.nan, "Vq": np.nan,
                "q_used": None, "rejeita_H0": None}

    if q is None:
        q = max(int(n ** (1 / 3)), 4)

    sigma = _sigma_lo(serie, q)

    lags = np.unique(
        np.floor(np.logspace(np.log10(min_lags),
                             np.log10(n // 2), 20)).astype(int)
    )

    rs_vals  = []
    lag_vals = []

    for lag in lags:
        rs_list = []
        for inicio in range(0, n - lag, lag):
            sub    = serie[inicio:inicio + lag]
            sigma_sub = _sigma_lo(sub, min(q, len(sub) // 4))
            if sigma_sub == 0:
                continue
            desvios = np.cumsum(sub - sub.mean())
            rs_list.append((desvios.max() - desvios.min()) / sigma_sub)
        if rs_list:
            rs_vals.append(np.mean(rs_list))
            lag_vals.append(lag)

    if len(lag_vals) < 3:
        return {"H_lo": np.nan, "Vq": np.nan,
                "q_used": q, "rejeita_H0": None}

    slope, *_ = np.polyfit(np.log(lag_vals), np.log(rs_vals), 1)

    # Estatística Vq de Lo (1991): RS / (σ_Lo * √N)
    serie_cumdev = np.cumsum(serie - serie.mean())
    rs_full      = (serie_cumdev.max() - serie_cumdev.min()) / sigma
    Vq           = rs_full / np.sqrt(n)

    # IC 95% assintótico sob H0 (Lo, 1991, Tabela II)
    rejeita = not (0.809 <= Vq <= 1.862)

    return {
        "H_lo"       : float(slope),
        "Vq"         : float(Vq),
        "q_used"     : q,
        "rejeita_H0" : rejeita,
    }
