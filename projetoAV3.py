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
st.set_page_config(page_title="Análise: Engajamento vs Sucesso", layout="wide", initial_sidebar_state="expanded")

# Tema visual mais limpo e elegante para os gráficos
sns.set_theme(style="ticks", palette="muted")

# Configuração da autenticação do Kaggle
try:
    os.environ["KAGGLE_API_TOKEN"] = st.secrets["KAGGLE_API_TOKEN"]
except Exception:
    st.sidebar.warning("Token do Kaggle não configurado. O download pode falhar sem autenticação prévia.")

ALFA = 0.05        
MIN_MENCOES = 3    

# --- Funções de Apoio e Processamento ---

@st.cache_data
def baixar_dataset(handle):
    try:
        return kagglehub.dataset_download(handle)
    except Exception as erro:
        st.error(f"Erro ao baixar '{handle}': {erro}")
        st.stop()

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

# =====================================================================
# INÍCIO DA INTERFACE
# =====================================================================

st.title("🎬 O engajamento nas redes sociais indica o sucesso de um filme?")
st.markdown("Uma análise cruzando dados de Bilheteria (IMDb), Trailers (YouTube) e Resenhas de Usuários.")
st.markdown("---")

# Carregamento invisível dos dados para o usuário não ficar esperando em cada aba
with st.spinner("Processando bases de dados..."):
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
    
    path_rev = baixar_dataset("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    arquivo_rev = [f for f in os.listdir(path_rev) if f.endswith(".csv")][0]
    df_rev_raw = pd.read_csv(os.path.join(path_rev, arquivo_rev))
    df_rev = limpar_reviews(df_rev_raw)
    
    df_mencoes = contar_mencoes(df_final, df_rev)
    df_final_rev = df_final.merge(df_mencoes, on=["movie_title", "title_year"], how="left")
    df_final_rev["rev_mencoes"] = df_final_rev["rev_mencoes"].fillna(0).astype(int)
    df_final_rev.loc[df_final_rev["rev_mencoes"] < MIN_MENCOES, "rev_pct_positivo"] = np.nan

# --- Explorando as Bases (Usando Expander para limpar a tela) ---
with st.expander("🔍 Explorar as Bases de Dados (Visão Geral)"):
    tab1, tab2, tab3 = st.tabs(["1. IMDb 5000", "2. YouTube", "3. Resenhas (50K)"])
    
    with tab1:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Base Limpa:**")
            st.write(f"De {df_imdb_raw.shape[0]} para {df_limpo.shape[0]} filmes únicos.")
            st.markdown("Variáveis mais usadas: `movie_facebook_likes`, `gross`, `imdb_score`.")
        with c2:
            cols = ["movie_facebook_likes", "gross", "imdb_score"]
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            sns.heatmap(df_limpo[cols].corr(method="spearman"), annot=True, cmap="Blues", fmt=".2f", ax=ax1)
            ax1.set_title("Correlação IMDb Original")
            st.pyplot(fig1)

    with tab2:
        st.write(f"**Integração:** {len(df_final)} filmes casaram entre IMDb e YouTube (via Título e Ano).")
        st.dataframe(df_final[["movie_title", "title_year", "gross_imdb", "favorability"]].head())

    with tab3:
        df_rev_ok = df_final_rev.dropna(subset=["rev_pct_positivo"])
        st.write(f"**Análise de Sentimento:** {len(df_rev_ok)} filmes foram citados em pelo menos {MIN_MENCOES} resenhas.")
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        sns.regplot(data=df_rev_ok, x="rev_pct_positivo", y="imdb_score", 
                    scatter_kws={"alpha": 0.5, "color": "#9467bd"}, line_kws={"color": "#111", "linewidth": 2}, ax=ax3)
        ax3.set_title("% Resenhas Positivas vs Nota IMDb")
        sns.despine()
        st.pyplot(fig3)

# --- VISUALIZAÇÕES PRINCIPAIS (TEXTO NA ESQUERDA, GRÁFICO NA DIREITA) ---
st.header("📊 Análise Visual das Hipóteses")
st.markdown("Como as diferentes métricas de engajamento se comportam em relação ao faturamento e à nota dos filmes?")
st.write("")

# 1. Curtidas vs Bilheteria
col_txt1, col_fig1 = st.columns([1, 2])
with col_txt1:
    st.subheader("Curtidas vs. Bilheteria")
    st.markdown("""
    **Como ler:** 
    Cada ponto representa um filme. Os eixos estão em escala logarítmica para acomodar tanto os filmes independentes quanto os grandes blockbusters.
    
    **Leitura dos dados:**
    A nuvem de pontos sobe da esquerda para a direita. Filmes com **mais curtidas tendem a ter maior bilheteria**. 
    
    Contudo, a correlação caiu para `0.36` na base final, indicando que curtidas são uma pista importante, mas não garantem sozinhas o sucesso financeiro.
    """)
with col_fig1:
    d_disp = df_final.dropna(subset=["movie_facebook_likes", "gross_imdb"]).copy()
    d_disp = d_disp[(d_disp["movie_facebook_likes"] > 0) & (d_disp["gross_imdb"] > 0)]
    d_disp["log_curtidas"] = np.log10(d_disp["movie_facebook_likes"])
    d_disp["log_bilheteria"] = np.log10(d_disp["gross_imdb"])
    
    fig4, ax4 = plt.subplots(figsize=(8, 4.5))
    sns.regplot(data=d_disp, x="log_curtidas", y="log_bilheteria", ax=ax4,
                scatter_kws={"alpha": 0.5, "color": "#3498db"},
                line_kws={"color": "#e74c3c", "linewidth": 2})
    ax4.set_title("Curtidas da Página do Filme vs. Bilheteria (Log10)", fontweight="bold", pad=15)
    ax4.set_xlabel("Curtidas no Facebook (log10)")
    ax4.set_ylabel("Bilheteria (log10)")
    sns.despine()
    st.pyplot(fig4)

st.markdown("---")

# 2. Distribuição por Faixas
col_txt2, col_fig2 = st.columns([1, 2])
with col_txt2:
    st.subheader("Bilheteria por Nível de Engajamento")
    st.markdown("""
    **Como ler:** 
    Dividimos os filmes em três grupos de tamanho igual baseados no volume de curtidas no Facebook. A linha central da caixa é a mediana e a bolinha vermelha é a média.
    
    **Leitura dos dados:**
    Observe como a mediana (e as caixas como um todo) "sobem" degrau por degrau do baixo para o alto engajamento. 
    
    Existe sobreposição (filmes com pouco engajamento que estouraram de bilheteria), mas a tendência geral é muito clara a favor dos mais engajados.
    """)
with col_fig2:
    FAIXAS = ["Baixo", "Médio", "Alto"]
    d_faixa = df_final.dropna(subset=["movie_facebook_likes", "gross_imdb"]).copy()
    d_faixa["Engajamento"] = pd.qcut(d_faixa["movie_facebook_likes"], q=3, labels=FAIXAS)
    
    fig5, ax5 = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=d_faixa, x="Engajamento", y="gross_imdb", hue="Engajamento",
                palette="Blues", legend=False, showmeans=True, ax=ax5,
                meanprops={"marker": "o", "markerfacecolor": "#e74c3c", "markeredgecolor": "white", "markersize": 8})
    ax5.set_yscale("log")
    ax5.set_title("Distribuição da Bilheteria por Faixa de Engajamento", fontweight="bold", pad=15)
    ax5.set_xlabel("Nível de Engajamento")
    ax5.set_ylabel("Bilheteria em USD (Escala Log)")
    sns.despine()
    st.pyplot(fig5)

st.markdown("---")

# 3. Sentimento YouTube
col_txt3, col_fig3 = st.columns([1, 2])
with col_txt3:
    st.subheader("Comentários do Trailer (YouTube)")
    st.markdown("""
    **Como ler:** 
    O eixo horizontal representa a Taxa de Favorabilidade (percentual de comentários positivos no YouTube).
    
    **Leitura dos dados:**
    Diferente das curtidas, o sentimento dos comentários formam uma nuvem muito larga e quase sem direção. 
    
    A favorabilidade tem uma correlação ínfima com a bilheteria. Ela reflete um pouco melhor a **Nota do IMDb** (qualidade percebida pelo público), mas definitivamente não serve como bússola comercial.
    """)
with col_fig3:
    d_sent = df_final.dropna(subset=["favorability", "imdb_score", "gross_imdb"]).copy()
    d_sent = d_sent[d_sent["gross_imdb"] > 0]
    d_sent["log_bilheteria"] = np.log10(d_sent["gross_imdb"])

    fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    sns.regplot(data=d_sent, x="favorability", y="imdb_score", ax=ax6a,
                scatter_kws={"alpha": 0.4, "color": "#2ecc71"},
                line_kws={"color": "#2c3e50", "linewidth": 2})
    ax6a.set_title("Favorabilidade vs. Nota IMDb", fontweight="bold")
    ax6a.set_xlabel("Comentários Positivos (%)")
    ax6a.set_ylabel("Nota no IMDb (0 - 10)")
    
    sns.regplot(data=d_sent, x="favorability", y="log_bilheteria", ax=ax6b,
                scatter_kws={"alpha": 0.4, "color": "#f39c12"},
                line_kws={"color": "#2c3e50", "linewidth": 2})
    ax6b.set_title("Favorabilidade vs. Bilheteria", fontweight="bold")
    ax6b.set_xlabel("Comentários Positivos (%)")
    ax6b.set_ylabel("Bilheteria (log10)")
    
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig6)

# --- RESULTADOS DOS TESTES E CONCLUSÃO ---
st.header("🎯 Conclusão e Teste de Hipóteses")

HIPOTESES = [
    ("H1", "Curtidas do filme -> Bilheteria", "movie_facebook_likes", "gross_imdb"),
    ("H2", "Curtidas do elenco -> Bilheteria", "cast_total_facebook_likes", "gross_imdb"),
    ("H3", "Curtidas do diretor -> Bilheteria", "director_facebook_likes", "gross_imdb"),
    ("H4", "Favorabilidade (YouTube) -> Nota IMDb", "favorability", "imdb_score"),
    ("H5", "Favorabilidade (YouTube) -> Bilheteria", "favorability", "gross_imdb"),
]

def veredito(rho, p):
    if np.isnan(rho): return "Sem dados suficientes"
    if p >= ALFA: return "Não apoiada"
    if rho <= 0: return "Não apoiada"
    if abs(rho) < 0.10: return "Apoiada (desprezível)"
    return f"Apoiada ({classificar_forca(rho)})"

linhas = []
for cod, texto, x, y in HIPOTESES:
    if x in df_final_rev.columns and y in df_final_rev.columns:
        r = correlacao_spearman(df_final_rev, x, y)
        linhas.append({
            "Hipótese": cod, "Descrição": texto,
            "Força (Rho)": r["rho"], "Veredito": veredito(r["rho"], r["p"])
        })
df_res = pd.DataFrame(linhas)

col_txt4, col_fig4 = st.columns([1.2, 1.5])

with col_txt4:
    st.markdown("""
    Rodamos correlações matemáticas severas (Spearman) para validar cientificamente os gráficos acima.
    """)
    st.dataframe(df_res[["Descrição", "Veredito"]], hide_index=True)
    
    st.success("""
    **Veredito Final:** O engajamento nas redes sociais **é um indicador parcial do sucesso**. 
    
    Curtidas da página do filme funcionam bem para prever bilheteria. Já o sentimento de comentários de trailer ajuda muito pouco (serve mais como termômetro de nota do que de dinheiro).
    
    *(Lembrete: Filmes maiores possuem orçamentos enormes de marketing que impulsionam simultaneamente os likes e a bilheteria).*
    """)

with col_fig4:
    d_res = df_res.dropna(subset=["Força (Rho)"]).copy()
    
    fig7, ax7 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=d_res, y="Hipótese", x="Força (Rho)", color="#3498db", ax=ax7)
    
    ax7.axvline(0, color="black", linewidth=1)
    for i, v in enumerate(d_res["Força (Rho)"]):
        ax7.text(v + 0.01, i, f"{v:.2f}", va="center", fontweight="bold", color="#333")
        
    ax7.set_title("Força da Correlação por Hipótese (Spearman)", fontweight="bold", pad=15)
    ax7.set_xlabel("Força (rho)")
    ax7.set_ylabel("")
    sns.despine(left=True, bottom=True)
    ax7.xaxis.set_visible(False) # Remove o eixo X que fica redundante com os números na barra
    st.pyplot(fig7)
