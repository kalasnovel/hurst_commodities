"""
Constantes globais do pipeline Hurst.
"""

# ── Chave FRED ────────────────────────────────────────────────────────────────
# Substitua pela sua chave gratuita antes de rodar com dados reais.
# Registro em: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = ""

# ── Séries FRED ──────────────────────────────────────────────────────────────
COMMODITIES = {
    "WTI (Petróleo)"   : "DCOILWTICO",
    "Brent (Petróleo)" : "DCOILBRENTEU",
    "Soja"             : "PSOYBUSDM",
    "Milho"            : "PMAIZMTUSDM",
}

# ── Paleta de cores ───────────────────────────────────────────────────────────
CORES = {
    "WTI (Petróleo)"   : "#D85A30",
    "Brent (Petróleo)" : "#BA7517",
    "Soja"             : "#1D9E75",
    "Milho"            : "#7F77DD",
}

# ── Período de análise ────────────────────────────────────────────────────────
DATA_INICIO = "1990-01-01"
DATA_FIM    = "2024-12-31"

# ── Parâmetros metodológicos ──────────────────────────────────────────────────
JANELA_ROLL   = 60    # meses para Hurst rolling (5 anos)
MIN_OBS_HURST = 32    # mínimo de observações para R/S confiável
N_BOOT        = 500   # amostras de bootstrap em bloco (Künsch, 1989)

# ── Eventos históricos (usados nas figuras) ───────────────────────────────────
EVENTOS = {
    "Crise asiática\n1997"   : "1997-07-01",
    "Crise financeira\n2008" : "2008-09-01",
    "Queda OPEP\n2014"       : "2014-11-01",
    "Covid-19\n2020"         : "2020-03-01",
    "Choque energético\n2022": "2022-02-01",
}
