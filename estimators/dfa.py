"""
Estimador DFA (Detrended Fluctuation Analysis) do Expoente de Hurst.

Referência:
    Peng, C.-K. et al. (1994). Mosaic organization of DNA nucleotides.
    Physical Review E, 49(2), 1685–1689.

Vantagem sobre o R/S:
    Robusto à não-estacionariedade e a tendências determinísticas.
"""

import numpy as np
from config import MIN_OBS_HURST


def hurst_dfa(serie: np.ndarray, ordem: int = 1) -> float:
    """
    Estima H pelo método DFA de ordem `ordem`.

    Algoritmo:
      1. Integra a série (perfil): Y(k) = sum_{i=1}^{k} [r_i - r_bar]
      2. Divide em segmentos não-sobrepostos de escala s
      3. Remove tendência polinomial local de grau `ordem` em cada segmento
      4. Calcula F(s) = sqrt(média dos resíduos quadrados)
      5. H é a inclinação de log F(s) × log s

    Escalas: 4 a N/4, com 20 pontos em espaçamento logarítmico.
    Segmentos com menos de 4 janelas são descartados.

    Padrão: ordem=1 (DFA-1, remoção de tendência linear).
    Robustez com ordem=2 (DFA-2) reportada como análise de sensibilidade.

    Retorna H estimado, ou np.nan se observações insuficientes.
    """
    n = len(serie)
    if n < MIN_OBS_HURST:
        return np.nan

    perfil  = np.cumsum(serie - serie.mean())
    escalas = np.unique(
        np.floor(np.logspace(np.log10(4), np.log10(n // 4), 20)).astype(int)
    )

    flutuacoes  = []
    escala_vals = []

    for s in escalas:
        n_seg = n // s
        if n_seg < 4:
            continue
        f2 = []
        for seg_idx in range(n_seg):
            seg     = perfil[seg_idx * s:(seg_idx + 1) * s]
            x       = np.arange(s)
            coef    = np.polyfit(x, seg, ordem)
            residuo = seg - np.polyval(coef, x)
            f2.append(np.mean(residuo ** 2))
        flutuacoes.append(np.sqrt(np.mean(f2)))
        escala_vals.append(s)

    if len(escala_vals) < 3:
        return np.nan

    slope, *_ = np.polyfit(np.log(escala_vals), np.log(flutuacoes), 1)
    return float(slope)
