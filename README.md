# Memória Longa em Commodities: Expoente de Hurst

Pipeline de análise para estimação do Expoente de Hurst em séries de preços mensais de commodities (WTI, Brent, Soja e Milho), cobrindo o período de janeiro de 1990 a dezembro de 2024.

---

## Estrutura do projeto

```
hurst_commodities/
├── config.py                          # Constantes globais
├── run.py                             # Entry point — executa o pipeline completo
├── requirements.txt
├── data/
│   ├── collector.py                   # Coleta FRED + fallback simulado
│   └── preprocessor.py               # Retornos logarítmicos e forward fill
├── estimators/
│   ├── rs.py                          # Método R/S (Rescaled Range)
│   ├── dfa.py                         # DFA-1 (Detrended Fluctuation Analysis)
│   └── bootstrap.py                   # Bootstrap em bloco — intervalos de confiança
├── analysis/
│   ├── global_stats.py                # Tabelas 1 e 2 (estatísticas e H global)
│   ├── rolling.py                     # H rolling (janela deslizante)
│   └── comparacao_estimadores.py      # Discrepância H_RS vs H_DFA
└── visualization/
    ├── style.py                       # Estilo global das figuras
    ├── fig1_acf.py                    # ACF dos retornos por commodity
    ├── fig2_loglog_rs.py              # Log-log plot R/S (diagnóstico)
    ├── fig3_hurst_barras.py           # H global com IC 95%
    ├── fig4_regimes.py                # Mapa de regimes ao longo do tempo
    └── fig5_heatmap.py                # Heatmap trienal por commodity
```

---

## Instalação

```bash
pip install -r requirements.txt
```

Dependências: `fredapi`, `pandas`, `numpy`, `matplotlib`, `scipy`, `statsmodels`

---

## Configuração

Antes de rodar com dados reais, insira sua chave FRED em `config.py`:

```python
FRED_API_KEY = "sua_chave_aqui"
```

Chave gratuita disponível em: https://fred.stlouisfed.org/docs/api/api_key.html

Sem chave, o pipeline roda com dados simulados.

---

## Execução

```bash
cd hurst_commodities
python run.py
```

---

## Outputs

Todos os arquivos são salvos em `./outputs/`:

| Arquivo | Conteúdo |
|---|---|
| `tabela1_estatisticas_descritivas.csv` | N, média, desvio-padrão, assimetria, curtose, Jarque-Bera, ADF por commodity |
| `tabela2_hurst_global.csv` | H_RS e H_DFA com IC 95% (bootstrap em bloco) e classificação de regime |
| `tabela3_comparacao_estimadores.csv` | Discrepância absoluta entre H_RS e H_DFA por commodity |
| `tabela4_rolling_hurst.csv` | Série temporal de H rolling (DFA, janela 60 meses) por commodity |
| `fig1_acf.png` | ACF dos retornos logarítmicos mensais (lags 1–24) |
| `fig2_loglog_rs.png` | Log-log plot R/S por commodity — inclinação = H_RS |
| `fig3_hurst_barras.png` | H global (DFA) com barras de erro IC 95% |
| `fig4_mapa_regimes.png` | Mapa de regimes de persistência ao longo do tempo |
| `fig5_heatmap_regimes.png` | Heatmap de H (DFA) por período trienal e commodity |

---

## Parâmetros metodológicos

Definidos em `config.py`:

| Parâmetro | Valor | Descrição |
|---|---|---|
| `JANELA_ROLL` | 60 meses | Janela deslizante para H rolling |
| `MIN_OBS_HURST` | 32 | Mínimo de observações para estimação confiável do R/S |
| `N_BOOT` | 500 | Amostras de bootstrap em bloco (Künsch, 1989) |

---

## Notas metodológicas

**R/S vs DFA:** O DFA é o estimador principal por ser robusto à não-estacionariedade. O R/S clássico não controla autocorrelações de curto prazo e pode inflar H. A `tabela3_comparacao_estimadores.csv` quantifica essa discrepância — valores acima de 0,10 devem ser adicionados como limitações.

**Retornos, não preços:** R/S e DFA são aplicados sobre retornos logarítmicos mensais. Retornos são estacionários em primeiro momento, condição relevante para a validade do R/S.

**Bootstrap em bloco:** O tamanho do bloco segue a regra b = ⌊N^(1/3)⌋, mínimo 4, conforme Künsch (1989). Os IC 95% são construídos pelo método percentil com 500 amostras.

---

## Referências dos métodos implementados

- Hurst, H. E. (1951). Long-term storage capacity of reservoirs. *Transactions of the American Society of Civil Engineers*, 116, 770–799.
- Peng, C.-K. et al. (1994). Mosaic organization of DNA nucleotides. *Physical Review E*, 49(2), 1685–1689.
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics*, 17(3), 1217–1241.
