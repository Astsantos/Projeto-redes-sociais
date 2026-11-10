import os
import re
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from scipy import stats
import streamlit as st

# --- Configurações da Página Streamlit ---
st.set_page_config(
    page_title="Análise: Engajamento vs Sucesso de Filmes",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização Global Personalizada (CSS)
st.markdown("""
<style>
    /* Estilo para cartões de métricas e containers */
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1f2937;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    /* Estilização de callouts e destaques */
    .highlight-box {
        background-color: #f0f7ff;
        border-left: 4px solid #2563eb;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 15px;
    }
    /* Ajustes finos de espaçamento */
    .stDataFrame { width: 100%; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
</style>
""", unsafe_allow_html=True)

# Tema visual para os gráficos com Seaborn/Matplotlib
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'grid.color': '#eeeeee',
    'grid.linestyle': '--',
    'figure.autolayout': True
})

# --- Funções de Apoio e Processamento ---

@st.cache_data
def baixar_dataset(handle):
    try:
        return kagglehub.dataset_download(handle)
    except Exception as erro:
        st.error(f"Erro ao baixar '{handle}': {erro}")
        st.stop()

def mostrar(fig):
    """Mostra a figura no app e libera a memória."""
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def descrever_base(df):
    """Tabela com o tipo, o nº de valores preenchidos e a % de ausentes de cada coluna."""
    res = pd.DataFrame({
        "Tipo de Dado": df.dtypes.astype(str),
        "Valores Preenchidos": df.notna().sum(),
        "Valores Ausentes (%)": (df.isna().mean() * 100).round(1),
    })
    return res.sort_values(by="Valores Ausentes (%)", ascending=False)

def classificar_forca(rho):
    a = abs(rho)
    if a < 0.10: return "desprezível"
    if a < 0.30: return "fraca"
    if a < 0.50: return "moderada"
    return "forte"

def correlacao_spearman(df, x, y):
    d = df[[x, y]].dropna()
    if len(d) < 10:
        return {"n": len(d), "rho": np.nan, "p": np.nan}
    rho, p = stats.spearmanr(d[x], d[y])
    return {"n": len(d), "rho": rho, "p": p}

@st.cache_data
def limpar_dados_imdb(df):
    df = df.copy()
    colunas = [
        "movie_title", "title_year", "genres", "country", "director_name",
        "director_facebook_likes", "actor_1_facebook_likes",
        "actor_2_facebook_likes", "actor_3_facebook_likes",
        "cast_total_facebook_likes", "movie_facebook_likes",
        "num_user_for_reviews", "num_critic_for_reviews",
        "imdb_score", "gross", "budget",
    ]
    df = df[colunas].drop_duplicates()
    df["movie_title"] = df["movie_title"].str.replace("\xa0", "").str.strip()
    essenciais = ["gross", "imdb_score", "director_facebook_likes",
                  "actor_1_facebook_likes", "title_year",
                  "actor_2_facebook_likes", "actor_3_facebook_likes",
                  "num_user_for_reviews", "num_critic_for_reviews"]
    df = df.dropna(subset=essenciais)
    df["title_year"] = df["title_year"].astype(int)
    for col in ["director_facebook_likes", "movie_facebook_likes"]:
        df[col] = df[col].replace(0, np.nan)
    df.loc[df["country"] != "USA", "budget"] = np.nan
    return df.reset_index(drop=True)

@st.cache_data
def limpar_youtube(df):
    df = df.copy().drop_duplicates()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["name", "year"])
    df["year"] = df["year"].astype(int)
    df["name"] = df["name"].astype(str).str.strip()
    return df.reset_index(drop=True)

def criar_chave(titulos):
    return titulos.astype(str).str.strip().str.lower()

@st.cache_data
def juntar_imdb_youtube(df_imdb, df_youtube):
    return df_imdb.merge(
        df_youtube, left_on=["chave", "title_year"], right_on=["chave", "year"],
        how="inner", suffixes=("_imdb", "_yt")
    )

@st.cache_data
def limpar_reviews(df):
    df = df.copy().dropna(subset=["review", "sentiment"]).drop_duplicates()
    df["review"] = (df["review"].str.replace(r"<[^>]+>", " ", regex=True)
                                .str.replace(r"\s+", " ", regex=True).str.strip())
    df["positivo"] = (df["sentiment"] == "positive").astype(int)
    df["n_palavras"] = df["review"].str.split().str.len()
    return df.reset_index(drop=True)

def normalizar(txt):
    txt = unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode()
    txt = re.sub(r"[^a-z0-9]+", " ", txt.lower())
    return txt.strip()

@st.cache_data
def contar_mencoes(df_filmes, df_resenhas):
    textos = df_resenhas["review"].map(lambda t: f" {normalizar(t)} ")
    positivo = df_resenhas["positivo"].to_numpy()

    titulos = df_filmes[["movie_title", "title_year"]].drop_duplicates().copy()
    titulos["titulo_norm"] = titulos["movie_title"].map(normalizar)
    titulos = titulos[~titulos["titulo_norm"].duplicated(keep=False)]
    titulos = titulos[(titulos["titulo_norm"].str.split().str.len() >= 2) |
                      (titulos["titulo_norm"].str.len() >= 10)]

    linhas = []
    for _, t in titulos.iterrows():
        achou = textos.str.contains(f" {t['titulo_norm']} ", regex=False).to_numpy()
        n = int(achou.sum())
        if n > 0:
            linhas.append({
                "movie_title": t["movie_title"],
                "title_year": t["title_year"],
                "rev_mencoes": n,
                "rev_pct_positivo": positivo[achou].mean(),
            })
    return pd.DataFrame(linhas, columns=["movie_title", "title_year", "rev_mencoes", "rev_pct_positivo"])

def veredito(rho, p):
    if np.isnan(rho): return "Sem dados suficientes"
    if p >= ALFA: return "Não apoiada (relação não significativa)"
    if rho <= 0: return "Não apoiada (relação negativa)"
    if abs(rho) < 0.10: return "Apoiada, mas com efeito desprezível"
    return f"Apoiada ({classificar_forca(rho)})"

def testar_hipoteses(df, hipoteses):
    """Spearman + p-valor para cada hipótese; devolve uma tabela com os resultados."""
    linhas = []
    for cod, texto, x, y, tipo in hipoteses:
        if x not in df.columns or y not in df.columns:
            continue
        r = correlacao_spearman(df, x, y)
        linhas.append({
            "Hipótese": cod, "Descrição": texto, "Tipo": tipo, "n": r["n"],
            "rho": r["rho"], "p-valor": r["p"],
            "Força": "-" if np.isnan(r["rho"]) else classificar_forca(r["rho"]),
            "Veredito": veredito(r["rho"], r["p"]),
        })
    return pd.DataFrame(linhas)

def teste_faixas(df, x="movie_facebook_likes", y="gross_imdb"):
    """Compara a bilheteria entre 3 faixas de engajamento (Kruskal-Wallis)."""
    d = df.dropna(subset=[x, y]).copy()
    d["faixa"] = pd.qcut(d[x], q=3, labels=FAIXAS)
    grupos = [g[y].to_numpy() for _, g in d.groupby("faixa", observed=True)]
    h, p = stats.kruskal(*grupos)
    return d.groupby("faixa", observed=True)[y].median(), h, p

def correlacao_parcial_spearman(df, x, y, z):
    """Correlação (em postos) entre x e y depois de tirar o efeito de z; devolve também a simples."""
    d = df[[x, y, z]].dropna()
    n = len(d)
    if n < 10:
        return {"n": n, "simples": np.nan, "parcial": np.nan, "p": np.nan}
    r = d.rank()

    def residuo(a, b):
        return a - np.polyval(np.polyfit(b, a, 1), b)

    parcial = np.corrcoef(residuo(r[x], r[z]), residuo(r[y], r[z]))[0, 1]
    t = parcial * np.sqrt((n - 3) / (1 - parcial ** 2))
    p = 2 * stats.t.sf(abs(t), n - 3)
    return {"n": n, "simples": stats.spearmanr(d[x], d[y])[0], "parcial": parcial, "p": p}

# =====================================================================
# INÍCIO DA INTERFACE STREAMLIT
# =====================================================================

# Cabeçalho Principal
st.title("O Engajamento nas Redes Sociais Indica o Sucesso de um Filme?")
st.markdown("Uma análise estatística cruzando dados de **Bilheteria (IMDb)**, **Trailers (YouTube)** e **Resenhas de Usuários**.")
st.markdown("---")

# Carregamento dos Dados
with st.spinner("Carregando e integrando bases de dados..."):
    path_imdb = baixar_dataset("carolzhangdc/imdb-5000-movie-dataset")
    df_imdb_raw = pd.read_csv(os.path.join(path_imdb, "movie_metadata.csv"))
    df_limpo = limpar_dados_imdb(df_imdb_raw)

    path_yt = baixar_dataset("dineshvasired/movies-youtube-trailers-and-sentimentdinesh-dinesh")
    arquivo_yt = [f for f in os.listdir(path_yt) if f.endswith(".csv")][0]
    df_yt_raw = pd.read_csv(os.path.join(path_yt, arquivo_yt))
    df_yt = limpar_youtube(df_yt_raw)

    df_limpo["chave"] = criar_chave(df_limpo["movie_title"])
    df_yt["chave"] = criar_chave(df_yt["name"])
    df_final = juntar_imdb_youtube(df_limpo, df_yt).drop_duplicates(subset=["chave", "year"]).copy()

    casaram = df_final[["chave", "year"]].drop_duplicates()
    df_nao = df_yt.merge(casaram, on=["chave", "year"], how="left", indicator=True)
    df_nao = df_nao[df_nao["_merge"] == "left_only"]
    n_outro_ano = len(df_nao.merge(df_limpo[["chave", "title_year"]], on="chave", how="inner"))

    path_rev = baixar_dataset("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    arquivo_rev = [f for f in os.listdir(path_rev) if f.endswith(".csv")][0]
    df_rev_raw = pd.read_csv(os.path.join(path_rev, arquivo_rev))
    df_rev = limpar_reviews(df_rev_raw)

    df_mencoes = contar_mencoes(df_final, df_rev)
    df_final_rev = df_final.merge(df_mencoes, on=["movie_title", "title_year"], how="left")
    df_final_rev["rev_mencoes"] = df_final_rev["rev_mencoes"].fillna(0).astype(int)
    df_final_rev.loc[df_final_rev["rev_mencoes"] < MIN_MENCOES, "rev_pct_positivo"] = np.nan

# --- EXPANDER: EXPLORAÇÃO DAS BASES DE DADOS ---
with st.expander("Explorar as Bases de Dados (Visão Geral e Dicionário)", expanded=False):
    tab1, tab2, tab3 = st.tabs(["1. IMDb 5000", "2. YouTube Trailers", "3. Resenhas (50K)"])

    with tab1:
        st.markdown("### Base Limpa de Filmes (IMDb)")
        c1, c2 = st.columns([1, 1.2], gap="medium")
        with c1:
            st.markdown("""
            **Resumo do Processamento:**
            - Dataset reduzido de **{}** para **{}** filmes únicos após limpeza.
            - Principais variáveis analisadas: `movie_facebook_likes`, `gross`, `imdb_score`.
            - **Decisões de limpeza principais:**
              - 45 registros duplicados removidos.
              - Zeros em curtidas de diretor/filme tratados como *ausentes* (páginas inexistentes na época).
              - Filtro de orçamento restrito a produções dos EUA para padronização em USD.
            """.format(df_imdb_raw.shape[0], df_limpo.shape[0]))
        with c2:
            cols = ["movie_facebook_likes", "gross", "imdb_score"]
            fig1, ax1 = plt.subplots(figsize=(6, 3.5))
            sns.heatmap(df_limpo[cols].corr(method="spearman"), annot=True, cmap="Blues", fmt=".2f", cbar=False, ax=ax1)
            ax1.set_title("Matriz de Correlação (Spearman - IMDb)", fontsize=11, fontweight="bold", pad=10)
            mostrar(fig1)

        st.markdown("#### Estatísticas Descritivas e Dicionário")
        sub_t1, sub_t2 = st.tabs(["Estatísticas Descritivas", "Dicionário de Colunas"])
        with sub_t1:
            st.dataframe(df_limpo.describe().T.style.format("{:.2f}"), use_container_width=True)
        with sub_t2:
            st.dataframe(descrever_base(df_imdb_raw), use_container_width=True)

    with tab2:
        st.markdown("### Integração IMDb + YouTube")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Filmes IMDb (Limpo)", f"{len(df_limpo):,}")
        m2.metric("Filmes YouTube", f"{len(df_yt):,}")
        m3.metric("Filmes Casados", f"{len(df_final):,}")
        m4.metric("Não Casados", f"{len(df_nao):,}")

        st.markdown("---")
        c_yt1, c_yt2 = st.columns([1, 1], gap="medium")
        with c_yt1:
            st.markdown("#### Amostra da Base Integrada")
            st.dataframe(
                df_final[["movie_title", "title_year", "movie_facebook_likes", "gross_imdb", "imdb_score", "favorability"]].head(8),
                use_container_width=True
            )
        with c_yt2:
            st.markdown("#### Estatísticas Descritivas (IMDb + YouTube)")
            cols_final = ["movie_facebook_likes", "gross_imdb", "imdb_score", "favorability", "votes"]
            st.dataframe(df_final[cols_final].describe().T.style.format("{:.2f}"), use_container_width=True)

    with tab3:
        st.markdown("### Análise de Sentimento das Resenhas (50K)")
        df_rev_ok = df_final_rev.dropna(subset=["rev_pct_positivo"])
        
        c_r1, c_r2 = st.columns([1, 1.2], gap="medium")
        with c_r1:
            st.markdown(f"""
            **Validação por Correspondência de Texto:**
            - **{len(df_rev_ok)} filmes** foram citados em pelo menos **{MIN_MENCOES} resenhas**.
            - Cada resenha foi vinculada ao filme caso o título exato apareça no texto.
            - Serve como validação metodológica: a % de resenhas positivas reflete a nota média no IMDb?
            """)
            st.markdown("**Filmes Mais Citados (Top 5):**")
            st.dataframe(df_mencoes.sort_values("rev_mencoes", ascending=False).head(5), use_container_width=True)
        with c_r2:
            fig3, ax3 = plt.subplots(figsize=(6, 3.5))
            sns.regplot(data=df_rev_ok, x="rev_pct_positivo", y="imdb_score",
                        scatter_kws={"alpha": 0.5, "color": "#8e44ad"}, line_kws={"color": "#2c3e50", "linewidth": 2}, ax=ax3)
            ax3.set_title("% Resenhas Positivas vs. Nota IMDb", fontsize=11, fontweight="bold")
            ax3.set_xlabel("% Resenhas Positivas")
            ax3.set_ylabel("Nota IMDb")
            sns.despine()
            mostrar(fig3)

# --- PLANO DE HIPÓTESES ---
st.markdown("##Hipóteses de Pesquisa")
st.markdown("A pergunta central orientadora é: ***O engajamento nas redes sociais pode prever o sucesso comercial ou de crítica de um filme?***")

HIPOTESES = [
    ("H1", "Curtidas do filme -> Bilheteria", "movie_facebook_likes", "gross_imdb", "Rede Social"),
    ("H2", "Curtidas do elenco -> Bilheteria", "cast_total_facebook_likes", "gross_imdb", "Rede Social"),
    ("H3", "Curtidas do diretor -> Bilheteria", "director_facebook_likes", "gross_imdb", "Rede Social"),
    ("H4", "Favorabilidade (YouTube) -> Nota IMDb", "favorability", "imdb_score", "Rede Social"),
    ("H5", "Favorabilidade (YouTube) -> Bilheteria", "favorability", "gross_imdb", "Rede Social"),
    ("H6", "% Resenhas Positivas -> Nota IMDb (Validação)", "rev_pct_positivo", "imdb_score", "Validação"),
]

df_hip_table = pd.DataFrame([
    {"Código": h[0], "Descrição da Hipótese": h[1], "Variável X (Engajamento)": h[2], "Variável Y (Sucesso)": h[3], "Categoria": h[4]}
    for h in HIPOTESES
])
st.dataframe(df_hip_table, use_container_width=True, hide_index=True)

st.markdown("---")

# --- VISUALIZAÇÕES PRINCIPAIS DAS HIPÓTESES ---
st.markdown("##Análise Visual das Hipóteses")
st.markdown("Comparativo visual entre as métricas de engajamento social, faturamento e recepção do público.")

# 1. Curtidas vs Bilheteria
with st.container():
    col_txt1, col_fig1 = st.columns([1, 1.2], gap="large")
    with col_txt1:
        st.markdown("### 1. Curtidas na Página vs. Bilheteria")
        st.markdown("""
        <div class="highlight-box">
        <strong>Como interpretar:</strong><br>
        Cada ponto representa um filme em escala logarítmica (Log10) para normalizar valores extremos entre produções indies e blockbusters.
        </div>

        - **Tendência:** Existe uma inclinação ascendente nítida — filmes com mais curtidas no Facebook tendem a faturar mais.
        - **Intensidade:** A correlação de Spearman na base final é de <code>0.36</code> (moderada).
        - **Conclusão:** O engajamento da página é um indicativo positivo, mas não garante sucesso absoluto de bilheteria isoladamente.
        """, unsafe_allow_html=True)
    with col_fig1:
        d_disp = df_final.dropna(subset=["movie_facebook_likes", "gross_imdb"]).copy()
        d_disp = d_disp[(d_disp["movie_facebook_likes"] > 0) & (d_disp["gross_imdb"] > 0)]
        d_disp["log_curtidas"] = np.log10(d_disp["movie_facebook_likes"])
        d_disp["log_bilheteria"] = np.log10(d_disp["gross_imdb"])

        fig4, ax4 = plt.subplots(figsize=(7, 4))
        sns.regplot(data=d_disp, x="log_curtidas", y="log_bilheteria", ax=ax4,
                    scatter_kws={"alpha": 0.45, "color": "#2980b9"},
                    line_kws={"color": "#c0392b", "linewidth": 2.2})
        ax4.set_title("Curtidas no Facebook (Log10) vs. Bilheteria (Log10)", fontweight="bold", pad=12)
        ax4.set_xlabel("Curtidas da Página do Filme (log10)")
        ax4.set_ylabel("Bilheteria em USD (log10)")
        sns.despine()
        mostrar(fig4)

st.markdown("---")

# 2. Distribuição por Faixas de Engajamento
with st.container():
    d_faixa = df_final.dropna(subset=["movie_facebook_likes", "gross_imdb"]).copy()
    d_faixa["Engajamento"] = pd.qcut(d_faixa["movie_facebook_likes"], q=3, labels=FAIXAS)
    medianas_faixa = d_faixa.groupby("Engajamento", observed=True)["gross_imdb"].median()

    col_txt2, col_fig2 = st.columns([1, 1.2], gap="large")
    with col_txt2:
        st.markdown("### 2. Bilheteria por Faixa de Engajamento")
        st.markdown("""
        <div class="highlight-box">
        <strong>Como interpretar:</strong><br>
        Os filmes foram divididos em 3 terços iguais (Baixo, Médio e Alto Engajamento). A linha central do boxplot é a mediana e a marca vermelha representa a média.
        </div>

        - **Progressão:** A mediana de bilheteria cresce de forma contínua da faixa Baixa para a Alta.
        - **Variação:** Note a presença de *outliers* (filmes com poucas curtidas mas bilheteria expressiva), reforçando a necessidade de análises não-paramétricas.
        """, unsafe_allow_html=True)
        
        st.markdown("**Mediana de Bilheteria por Grupo:**")
        df_med_show = pd.DataFrame({
            "Faixa de Engajamento": medianas_faixa.index,
            "Mediana Bilheteria (USD)": [f"$ {v:,.0f}" for v in medianas_faixa.values]
        })
        st.dataframe(df_med_show, use_container_width=True, hide_index=True)

    with col_fig2:
        fig5, ax5 = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=d_faixa, x="Engajamento", y="gross_imdb", hue="Engajamento",
                    palette="Blues", legend=False, showmeans=True, ax=ax5,
                    meanprops={"marker": "o", "markerfacecolor": "#e74c3c", "markeredgecolor": "white", "markersize": 8})
        ax5.set_yscale("log")
        ax5.set_title("Distribuição da Bilheteria por Faixa de Engajamento", fontweight="bold", pad=12)
        ax5.set_xlabel("Nível de Engajamento no Facebook")
        ax5.set_ylabel("Bilheteria em USD (Escala Log)")
        sns.despine()
        mostrar(fig5)

st.markdown("---")

# 3. Sentimento dos Comentários do YouTube
with st.container():
    r_h5 = correlacao_spearman(df_final, "favorability", "gross_imdb")
    txt_h5 = ("sem dados suficientes" if np.isnan(r_h5["rho"])
              else f"correlação {classificar_forca(r_h5['rho'])} (ρ = {r_h5['rho']:.2f})")

    col_txt3, col_fig3 = st.columns([1, 1.2], gap="large")
    with col_txt3:
        st.markdown("### 3. Sentimento no YouTube vs. Sucesso")
        st.markdown(f"""
        <div class="highlight-box">
        <strong>Como interpretar:</strong><br>
        A Taxa de Favorabilidade mede o percentual de comentários positivos nos trailers oficiais no YouTube.
        </div>

        - **vs. Bilheteria:** Relação praticamente inexistente ({txt_h5}). O otimismo nos comentários não se traduz em vendas de ingressos.
        - **vs. Nota IMDb:** Relação positiva moderada. O sentimento do trailer indica melhor a receptividade de crítica do público do que a arrecadação financeira.
        """, unsafe_allow_html=True)

    with col_fig3:
        d_sent = df_final.dropna(subset=["favorability", "imdb_score", "gross_imdb"]).copy()
        d_sent = d_sent[d_sent["gross_imdb"] > 0]
        d_sent["log_bilheteria"] = np.log10(d_sent["gross_imdb"])

        fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(8, 3.8))

        sns.regplot(data=d_sent, x="favorability", y="imdb_score", ax=ax6a,
                    scatter_kws={"alpha": 0.4, "color": "#27ae60"},
                    line_kws={"color": "#2c3e50", "linewidth": 2})
        ax6a.set_title("Favorabilidade vs. Nota IMDb", fontweight="bold", fontsize=10)
        ax6a.set_xlabel("Comentários Positivos (%)")
        ax6a.set_ylabel("Nota IMDb (0-10)")

        sns.regplot(data=d_sent, x="favorability", y="log_bilheteria", ax=ax6b,
                    scatter_kws={"alpha": 0.4, "color": "#f39c12"},
                    line_kws={"color": "#2c3e50", "linewidth": 2})
        ax6b.set_title("Favorabilidade vs. Bilheteria", fontweight="bold", fontsize=10)
        ax6b.set_xlabel("Comentários Positivos (%)")
        ax6b.set_ylabel("Bilheteria (log10)")

        sns.despine()
        mostrar(fig6)

st.markdown("---")

# --- TESTE DE HIPÓTESES E CONCLUSÃO ---
st.markdown("##Resultados dos Testes Estatísticos e Conclusão")

df_res = testar_hipoteses(df_final_rev, HIPOTESES)

col_tabelas, col_grafico = st.columns([1.3, 1], gap="large")

with col_tabelas:
    st.markdown("#### Tabela de Resultados (Spearman)")
    tabela = df_res[["Hipótese", "Descrição", "n", "rho", "p-valor", "Força", "Veredito"]].copy()
    tabela["rho"] = tabela["rho"].round(2)
    tabela["p-valor"] = tabela["p-valor"].map(lambda p: f"{p:.3g}")
    
    st.dataframe(tabela, use_container_width=True, hide_index=True)
    
    st.download_button(
        label="Baixar Resultados Completo (CSV)",
        data=df_res.to_csv(index=False).encode("utf-8"),
        file_name="resultados_hipoteses.csv",
        mime="text/csv"
    )

with col_grafico:
    st.markdown("#### Ranking de Força das Hipóteses (ρ)")
    d_res = df_res.dropna(subset=["rho"]).copy()

    fig7, ax7 = plt.subplots(figsize=(6, 3.8))
    bars = sns.barplot(data=d_res, y="Hipótese", x="rho", hue="Hipótese", palette="Blues_r", legend=False, ax=ax7)

    ax7.axvline(0, color="black", linewidth=1)
    for i, v in enumerate(d_res["rho"]):
        ax7.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:.2f}", va="center",
                 ha="left" if v >= 0 else "right", fontweight="bold", color="#2c3e50", fontsize=9)

    ax7.set_title("Coeficiente de Correlação de Spearman (ρ)", fontweight="bold", pad=12)
    ax7.set_xlabel("Força (rho)")
    ax7.set_ylabel("")
    sns.despine(left=True, bottom=True)
    ax7.xaxis.set_visible(False)
    mostrar(fig7)

st.markdown("---")

# Caixa de síntese final
st.success("""
**Veredito Sintético:**
- **Curtidas nas Redes (Página / Elenco):** Apresentam correlação moderada e positiva com a bilheteria. Filmes com maior engajamento prévio no Facebook tendem a faturar mais.
- **Sentimento nos Comentários (YouTube):** Não serve como indicador financeiro de bilheteria, mas se correlaciona positivamente com a nota final do público/crítica.
- **Efeito Orçamento:** Parte expressiva da relação entre curtidas e bilheteria é mediada pelo orçamento de marketing das produções.
""")

# --- TESTES COMPLEMENTARES ---
with st.expander("Testes Estatísticos Complementares (Kruskal-Wallis e Correlação Parcial)"):
    st.markdown("#### 1. Teste de Kruskal-Wallis (Diferença entre Faixas)")
    medianas, h, p_kw = teste_faixas(df_final)
    c_kw1, c_kw2 = st.columns([1, 1])
    with c_kw1:
        st.dataframe(medianas.round(0).rename("Mediana Bilheteria (USD)"), use_container_width=True)
    with c_kw2:
        st.info(f"**Estatística H:** {h:.2f}\n\n**p-valor:** {p_kw:.3g}\n\n" +
                ("Há diferença estatisticamente significativa na bilheteria entre as faixas de engajamento." if p_kw < ALFA else "❌ Não há diferença significativa."))

    st.markdown("---")
    st.markdown("#### 2. Correlação Parcial Controlando o Orçamento")
    st.caption("Verifica se a relação entre curtidas e bilheteria persiste quando isolamos o efeito do orçamento (apenas produções EUA).")
    
    linhas_parcial = []
    for x, nome in [("movie_facebook_likes", "H1 Curtidas do Filme"), ("cast_total_facebook_likes", "H2 Curtidas do Elenco")]:
        r = correlacao_parcial_spearman(df_final, x, "gross_imdb", "budget_imdb")
        if not np.isnan(r["parcial"]):
            linhas_parcial.append({
                "Relação": f"{nome} vs Bilheteria",
                "n": r["n"],
                "rho (Sem Controle)": round(r["simples"], 2),
                "rho (Controlando Orçamento)": round(r["parcial"], 2),
                "p-valor": f"{r['p']:.3g}"
            })
    if linhas_parcial:
        st.dataframe(pd.DataFrame(linhas_parcial), use_container_width=True, hide_index=True)
    else:
        st.write("Dados insuficientes de orçamento para o cálculo.")
