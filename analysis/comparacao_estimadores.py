"""
Comparação entre estimadores H_RS e H_DFA por commodity.
O R/S clássico tende a inflar H em séries com autocorrelação de curto prazo.
"""

import numpy as np
import pandas as pd
from estimators.rs  import hurst_rs
from estimators.dfa import hurst_dfa


def tabela_comparacao_estimadores(retornos: pd.DataFrame) -> pd.DataFrame:
    """
    Tabela auxiliar: H_RS, H_DFA e discrepância absoluta por commodity.

    Discrepância > 0.05 deve ser mencionada nas limitações.
    Discrepância > 0.10 é sinal de possível viés do R/S por autocorrelação
    de curto prazo — reforça o DFA como estimador principal.
    
    """
    rows = []
    for col in retornos.columns:
        r     = retornos[col].dropna().values
        h_rs  = hurst_rs(r)
        h_dfa = hurst_dfa(r)
        disc  = abs(h_rs - h_dfa) if not (np.isnan(h_rs) or np.isnan(h_dfa)) else np.nan

        rows.append({
            "Commodity"          : col,
            "H (R/S)"            : round(h_rs,  4),
            "H (DFA)"            : round(h_dfa, 4),
            "Discrepância |RS−DFA|": round(disc, 4),
            "Alerta"             : (
                "Verificar — discrepância > 0.10" if disc > 0.10
                else "Mencionar nas limitações"   if disc > 0.05
                else "OK"
            ) if not np.isnan(disc) else "—",
        })

    return pd.DataFrame(rows).set_index("Commodity")
