"""
Tabelas de resultados globais: estatísticas descritivas e Expoente de Hurst.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from config import N_BOOT
from estimators.rs        import hurst_rs
from estimators.dfa       import hurst_dfa
from estimators.lo_rs     import hurst_lo_rs
from estimators.bootstrap import bootstrap_ic, bootstrap_sensibilidade_bloco


def tabela1_estatisticas(retornos: pd.DataFrame) -> pd.DataFrame:
    """
    Tabela 1 — Estatísticas descritivas dos retornos logarítmicos mensais.
    """
    rows = []
    for col in retornos.columns:
        r       = retornos[col].dropna()
        adf_p   = adfuller(r, autolag="AIC")[1]
        _, jb_p = stats.jarque_bera(r)

        rows.append({
            "Commodity"         : col,
            "N"                 : len(r),
            "Média (%)"         : round(r.mean() * 100, 4),
            "Desvio-padrão (%)": round(r.std()  * 100, 4),
            "Mínimo (%)"        : round(r.min()  * 100, 4),
            "Máximo (%)"        : round(r.max()  * 100, 4),
            "Assimetria"        : round(stats.skew(r),     4),
            "Curtose"           : round(stats.kurtosis(r), 4),
            "Jarque-Bera (p)"   : round(jb_p,  4),
            "ADF (p-valor)"     : round(adf_p, 4),
        })

    return pd.DataFrame(rows).set_index("Commodity")


def tabela2_hurst(retornos: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    """
    Tabela 2 — H_RS e H_DFA com IC 95% por bootstrap (seed=42) + teste Lo.

    Lo (1991) é reportado como TESTE DE ROBUSTEZ, não como estimador alternativo:
      - Vq: estatística de teste (4 casas decimais para distinguir valores próximos)
      - Decisão Lo: "Não rejeita H0" ou "Rejeita H0" ao nível de 5%
      - H0: ausência de memória longa

    A coluna "H (Lo)" foi removida para evitar ambiguidade entre estimador e teste.

    Regime baseado no IC DFA (critério principal):
      lb > 0.5  → Persistente
      ub < 0.5  → Anti-persistente
      contrário → Indeterminado
    """
    rows = []
    for col in retornos.columns:
        r = retornos[col].dropna().values
        print(f"  Calculando Hurst para {col}...")

        h_rs,  lb_rs,  ub_rs  = bootstrap_ic(r, "rs",  n_boot, seed=42)
        h_dfa, lb_dfa, ub_dfa = bootstrap_ic(r, "dfa", n_boot, seed=42)
        lo                    = hurst_lo_rs(r)

        if not np.isnan(h_dfa):
            if lb_dfa > 0.5:
                regime = "Persistente"
            elif ub_dfa < 0.5:
                regime = "Anti-persistente"
            else:
                regime = "Indeterminado"
        else:
            regime = "—"

        decisao_lo = (
            "Rejeita H0 (5%)" if lo["rejeita_H0"]
            else "Não rejeita H0"
        ) if lo["rejeita_H0"] is not None else "—"

        rows.append({
            "Commodity"      : col,
            "H (R/S)"        : round(h_rs,       4),
            "IC 95% R/S"     : f"[{lb_rs:.3f}, {ub_rs:.3f}]",
            "H (DFA)"        : round(h_dfa,      4),
            "IC 95% DFA"     : f"[{lb_dfa:.3f}, {ub_dfa:.3f}]",
            "Regime (DFA)"   : regime,
            "Vq (Lo)"        : round(lo["Vq"],   4),   # 4 casas para distinguir valores próximos
            "Decisão Lo (5%)": decisao_lo,
        })

    return pd.DataFrame(rows).set_index("Commodity")


def tabela_sensibilidade_bloco(retornos: pd.DataFrame,
                               n_boot: int = N_BOOT) -> pd.DataFrame:
    """
    Tabela de robustez: IC 95% do H_DFA para três tamanhos de bloco.
    Seed fixa (42, 43, 44) por tamanho de bloco — mesma lógica do bootstrap_ic.
    """
    rows = []
    for col in retornos.columns:
        r    = retornos[col].dropna().values
        sens = bootstrap_sensibilidade_bloco(r, "dfa", n_boot, seed=42)

        rows.append({
            "Commodity"    : col,
            "H (DFA)"      : round(sens["H_obs"], 4),
            "IC b_min"     : f"[{sens['b_min']['IC_lb']:.3f}, {sens['b_min']['IC_ub']:.3f}]  (b={sens['b_min']['bloco']})",
            "IC b_default" : f"[{sens['b_default']['IC_lb']:.3f}, {sens['b_default']['IC_ub']:.3f}]  (b={sens['b_default']['bloco']})",
            "IC b_max"     : f"[{sens['b_max']['IC_lb']:.3f}, {sens['b_max']['IC_ub']:.3f}]  (b={sens['b_max']['bloco']})",
        })

    return pd.DataFrame(rows).set_index("Commodity")