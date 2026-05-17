"""
Bootstrap em bloco para intervalos de confiança do Expoente de Hurst.

Referência:
    Künsch, H. R. (1989). The jackknife and the bootstrap for general
    stationary observations. The Annals of Statistics, 17(3), 1217–1241.

Tamanho do bloco padrão: b = floor(N^{1/3}), mínimo 4.
Seed fixa (padrão: 42) garante reprodutibilidade entre execuções.
"""

import numpy as np
from config import N_BOOT
from estimators.rs  import hurst_rs
from estimators.dfa import hurst_dfa


def _resample(serie: np.ndarray, bloco: int, rng: np.random.Generator) -> np.ndarray:
    n        = len(serie)
    n_blocos = n // bloco + 1
    inicios  = rng.integers(0, n - bloco + 1, n_blocos)
    return np.concatenate([serie[i:i + bloco] for i in inicios])[:n]


def bootstrap_ic(
    serie: np.ndarray,
    metodo: str  = "dfa",
    n_boot: int  = N_BOOT,
    alpha: float = 0.05,
    seed: int    = 42,
) -> tuple[float, float, float]:
    """
    IC para H via bootstrap com bloco deslizante.

    Parâmetros:
      seed — semente aleatória para reprodutibilidade (padrão: 42).
             Manter o mesmo seed em todas as execuções garante que abstract,
             tabelas, figuras e texto referenciem os mesmos valores.

    Retorna (H_obs, limite_inferior, limite_superior).
    """
    fn    = hurst_rs if metodo == "rs" else hurst_dfa
    h_obs = fn(serie)

    if np.isnan(h_obs):
        return (np.nan, np.nan, np.nan)

    n     = len(serie)
    bloco = max(int(n ** (1 / 3)), 4)
    rng   = np.random.default_rng(seed)

    h_boot = []
    for _ in range(n_boot):
        h_b = fn(_resample(serie, bloco, rng))
        if not np.isnan(h_b):
            h_boot.append(h_b)

    if not h_boot:
        return (h_obs, np.nan, np.nan)

    lb = float(np.percentile(h_boot, 100 * alpha / 2))
    ub = float(np.percentile(h_boot, 100 * (1 - alpha / 2)))
    return (h_obs, lb, ub)


def bootstrap_sensibilidade_bloco(
    serie: np.ndarray,
    metodo: str  = "dfa",
    n_boot: int  = N_BOOT,
    alpha: float = 0.05,
    seed: int    = 42,
) -> dict:
    """
    Análise de robustez: IC com três tamanhos de bloco distintos.

    Tamanhos:
      b_min     = max(floor(N^(1/4)), 3)
      b_default = max(floor(N^(1/3)), 4)  — padrão (Künsch, 1989)
      b_max     = max(floor(N^(1/2)), 8)

    Seed fixa garante reprodutibilidade. Cada tamanho de bloco usa
    uma sub-seed derivada do seed principal para resultados independentes.
    """
    fn    = hurst_rs if metodo == "rs" else hurst_dfa
    h_obs = fn(serie)
    n     = len(serie)

    blocos = {
        "b_min"     : max(int(n ** (1 / 4)), 3),
        "b_default" : max(int(n ** (1 / 3)), 4),
        "b_max"     : max(int(n ** (1 / 2)), 8),
    }

    resultado = {"H_obs": h_obs}

    for i, (nome, bloco) in enumerate(blocos.items()):
        rng    = np.random.default_rng(seed + i)   # sub-seed por bloco
        h_boot = []
        for _ in range(n_boot):
            h_b = fn(_resample(serie, bloco, rng))
            if not np.isnan(h_b):
                h_boot.append(h_b)

        if h_boot:
            lb = float(np.percentile(h_boot, 100 * alpha / 2))
            ub = float(np.percentile(h_boot, 100 * (1 - alpha / 2)))
        else:
            lb, ub = np.nan, np.nan

        resultado[nome] = {"bloco": bloco, "IC_lb": lb, "IC_ub": ub}

    return resultado