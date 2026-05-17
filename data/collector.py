"""
Coleta de dados do FRED e gerador de dados simulados para fallback.
"""

import numpy as np
import pandas as pd
from config import COMMODITIES, DATA_INICIO, DATA_FIM, FRED_API_KEY


def coletar_dados(api_key: str) -> pd.DataFrame:
    """
    Baixa as séries do FRED e retorna DataFrame com preços mensais.

    WTI e Brent são séries diárias no FRED — resample para média mensal.
    Soja e Milho já estão disponíveis em frequência mensal.

    Nota sobre abril/2020: o WTI registrou preço diário negativo em
    20/abr/2020. A média mensal de abril permaneceu positiva (~16,5
    USD/barril) — nenhuma exclusão foi aplicada.
    """
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError("Execute: pip install fredapi")

    fred   = Fred(api_key=api_key)
    frames = {}

    for nome, ticker in COMMODITIES.items():
        print(f"  Baixando {nome} ({ticker})...")
        serie = fred.get_series(
            ticker,
            observation_start=DATA_INICIO,
            observation_end=DATA_FIM,
        )
        serie = serie.resample("MS").mean()
        frames[nome] = serie

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    return df.dropna(how="all")


def dados_simulados() -> pd.DataFrame:
    """
    Gera dados SINTÉTICOS para execução sem chave FRED.

    Os processos fBm aproximam comportamentos qualitativos descritos
    na literatura, mas NÃO reproduzem os dados reais do FRED.
    Remover ou manter como fallback documentado após obter chave real.
    """
    np.random.seed(42)
    datas = pd.date_range(DATA_INICIO, DATA_FIM, freq="MS")
    n     = len(datas)

    def fbm_approx(n: int, H: float, base: float = 100.0) -> np.ndarray:
        ruido = np.random.randn(n)
        pesos = np.array([k ** (H - 0.5) for k in range(1, n + 1)])
        serie = np.convolve(ruido, pesos / pesos.sum(), mode="full")[:n]
        return base * np.exp(np.cumsum(serie * 0.03))

    df = pd.DataFrame({
        "WTI (Petróleo)"   : fbm_approx(n, H=0.65, base=30),
        "Brent (Petróleo)" : fbm_approx(n, H=0.63, base=32),
        "Soja"             : fbm_approx(n, H=0.52, base=250),
        "Milho"            : fbm_approx(n, H=0.51, base=150),
    }, index=datas)

    # Simular colapso de demanda Covid-19
    idx = df.index.get_loc("2020-03-01")
    df.iloc[idx:idx + 3, [0, 1]] *= 0.55

    return df
