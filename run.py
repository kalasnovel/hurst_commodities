"""
Entry point do pipeline Hurst.

Uso:
    python run.py

Saídas em ./outputs/:
    tabela1_estatisticas_descritivas.csv
    tabela2_hurst_global.csv           — H_RS e H_DFA com IC 95% + teste Lo (Vq)
    tabela3_comparacao_estimadores.csv — discrepância entre estimadores
    tabela4_sensibilidade_bloco.csv    — IC DFA para 3 tamanhos de bloco
    tabela5_rolling_60m.csv            — H rolling, janela 60 meses
    tabela6_rolling_84m.csv            — H rolling, janela 84 meses
    tabela7_rolling_120m.csv           — H rolling, janela 120 meses
    fig1_acf.png
    fig2_loglog_rs.png
    fig3_hurst_barras.png
    fig4_mapa_regimes.png
    fig5_heatmap_regimes.png
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from config import FRED_API_KEY, N_BOOT

from data.collector    import coletar_dados, dados_simulados
from data.preprocessor import preprocessar

from analysis.global_stats           import (tabela1_estatisticas,
                                              tabela2_hurst,
                                              tabela_sensibilidade_bloco)
from analysis.rolling                import hurst_rolling_all, hurst_rolling_robustez
from analysis.comparacao_estimadores import tabela_comparacao_estimadores

from visualization.style             import aplicar_estilo
from visualization.fig1_acf          import fig1_acf
from visualization.fig2_loglog_rs    import fig2_loglog_rs
from visualization.fig3_hurst_barras import fig3_hurst_global
from visualization.fig4_regimes      import fig4_mapa_regimes
from visualization.fig5_heatmap      import fig5_heatmap_regimes


def main():
    os.makedirs("outputs", exist_ok=True)
    aplicar_estilo()

    print("=" * 60)
    print("MEMÓRIA LONGA EM COMMODITIES — EXPOENTE DE HURST")
    print("=" * 60)

    # ── 1. Coleta ────────────────────────────────────────────────────────────
    if FRED_API_KEY == "SUA_CHAVE_AQUI":
        print("\n[AVISO] Usando dados SIMULADOS — não citar no artigo.")
        df_precos = dados_simulados()
    else:
        print("\n[1/7] Coletando dados do FRED...")
        df_precos = coletar_dados(FRED_API_KEY)

    print(f"  ✓ {len(df_precos)} observações | "
          f"{df_precos.index[0].date()} → {df_precos.index[-1].date()}")

    # ── 2. Pré-processamento ─────────────────────────────────────────────────
    print("\n[2/7] Pré-processando séries...")
    precos, retornos = preprocessar(df_precos)
    print(f"  ✓ {len(retornos)} retornos logarítmicos mensais")

    # ── 3. Estatísticas descritivas ──────────────────────────────────────────
    print("\n[3/7] Estatísticas descritivas...")
    tab1 = tabela1_estatisticas(retornos)
    print(tab1.to_string())
    tab1.to_csv("outputs/tabela1_estatisticas_descritivas.csv")

    # ── 4. Hurst global + Lo + sensibilidade de bloco ────────────────────────
    print(f"\n[4/7] Calculando H_RS, H_DFA com bootstrap seed=42 (n={N_BOOT}) + teste Lo...")
    tab2 = tabela2_hurst(retornos, n_boot=N_BOOT)
    print("\n── Tabela 2: Hurst global ──")
    print(tab2.to_string())
    tab2.to_csv("outputs/tabela2_hurst_global.csv")

    tab3 = tabela_comparacao_estimadores(retornos)
    print("\n── Tabela 3: Discrepância entre estimadores ──")
    print(tab3.to_string())
    tab3.to_csv("outputs/tabela3_comparacao_estimadores.csv")

    print("\n  Calculando sensibilidade ao tamanho do bloco...")
    tab4 = tabela_sensibilidade_bloco(retornos, n_boot=N_BOOT)
    print("\n── Tabela 4: Sensibilidade do IC ao tamanho do bloco ──")
    print(tab4.to_string())
    tab4.to_csv("outputs/tabela4_sensibilidade_bloco.csv")

    # ── 5. Rolling (3 janelas) ───────────────────────────────────────────────
    print("\n[5/7] Calculando H rolling (60, 84, 120 meses)...")
    rolling = hurst_rolling_robustez(retornos)
    for janela, df_r in rolling.items():
        fname = f"outputs/tabela{4 + list(rolling.keys()).index(janela) + 1}_rolling_{janela}m.csv"
        df_r.to_csv(fname)
        print(f"  ✓ Rolling {janela} meses → {fname}")

    # ── 6. Figuras ───────────────────────────────────────────────────────────
    print("\n[6/7] Gerando figuras...")

    fig1_acf(retornos, savepath="outputs/fig1_acf.png")
    print("  ✓ Figura 1: ACF")

    fig2_loglog_rs(retornos, "outputs/fig2_loglog_rs.png")
    print("  ✓ Figura 2: log-log R/S")

    fig3_hurst_global(tab2, "outputs/fig3_hurst_barras.png")
    print("  ✓ Figura 3: H global com IC 95%")

    fig4_mapa_regimes(retornos, "outputs/fig4_mapa_regimes.png")
    print("  ✓ Figura 4: mapa de regimes")

    fig5_heatmap_regimes(retornos, "outputs/fig5_heatmap_regimes.png")
    print("  ✓ Figura 5: heatmap quinquenal")

    # ── 7. Resultados ───────────────────────────────────────────────────────────
    print("\n[7/7] Resultados:")
    print("─" * 60)
    for idx, row in tab2.iterrows():
        h_dfa = row["H (DFA)"]
        ic    = row["IC 95% DFA"]
        reg   = row["Regime (DFA)"]
        vq    = row["Vq (Lo)"]
        dec   = row["Decisão Lo (5%)"]
        sinal = "↑" if h_dfa > 0.55 else ("↓" if h_dfa < 0.45 else "→")
        print(f"  {sinal} {idx}")
        print(f"    H_DFA = {h_dfa:.4f}  IC {ic}  → {reg}")
        print(f"    Vq(Lo) = {vq:.4f}  {dec}")

if __name__ == "__main__":
    main()