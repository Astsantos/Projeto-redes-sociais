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
st.set_page_config(page_title="Análise de Filmes: Engajamento vs Sucesso", layout="wide")

# Configuração da autenticação do Kaggle
try:
    os.environ["KAGGLE_API_TOKEN"] = st.secrets["KAGGLE_API_TOKEN"]
except Exception:
    st.sidebar.warning("Token do Kaggle não encontrado em st.secrets. O download pode falhar se não houver autenticação prévia.")

sns.set_theme(style="whitegrid")

ALFA = 0.05        
MIN_MENCOES = 3    

# --- Funções de Apoio e Processamento (com Cache para otimizar o Streamlit) ---

@st.cache_data
def baixar_dataset(handle):
    try:
        return kagglehub.dataset_download(handle)
    except Exception as erro:
        st.error(f"Não foi possível baixar '{handle}': {erro}")
        st.stop()

def descrever_base(df, nome):
    st.write(f"**{nome}:** {df.shape[0]} linhas x {df.shape[1]} colunas")
    resumo = pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "preenchidos": df.notna().sum(),
        "ausentes (%)": (df.isna().mean() * 100).round(1),
    })
    return resumo

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

def descrever_relacao(df, x, y, rotulo):
    r = correlacao_spearman(df, x, y)
    if np.isnan(r["rho"]):
        st.write(f"**{rotulo}:** poucos dados (n = {r['n']}).")
        return r
    sentido = "positiva" if r["rho"] > 0 else "negativa"
    signif = "estatisticamente significativa" if r["p"] < ALFA else "não significativa"
    st.write(f"**{rotulo}:** Spearman = {r['rho']:.2f} (n = {r['n']}, p = {r['p']:.3g}) -> relação {classificar_forca(r['rho'])} {sentido}, {signif}.")
    return r

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
# INÍCIO DO APLICATIVO
# =====================================================================

st.title("🎬 O engajamento nas redes sociais indica o sucesso de um filme?")
st.markdown("Esta aplicação reproduz os passos de integração e análise de três bases de dados de cinema.")

# --- 1. IMDB 5000 ---
st.header("1. Base Principal: IMDB 5000")
with st.spinner("Baixando e carregando base IMDB..."):
    path = baixar_dataset("carolzhangdc/imdb-5000-movie-dataset")
    df_imdb_raw = pd.read_csv(os.path.join(path, "movie_metadata.csv"))

df_limpo = limpar_dados_imdb(df_imdb_raw)

with st.expander("Exploração Inicial (Após Limpeza)"):
    st.write(f"Antes: {df_imdb_raw.shape} | Depois: {df_limpo.shape}")
    cols = ["movie_facebook_likes", "cast_total_facebook_likes", "gross", "imdb_score"]
    
    # CORREÇÃO: Gráfico Histograma IMDB
    fig1, axes1 = plt.subplots(2, 2, figsize=(10, 6))
    df_limpo[cols].hist(bins=30, ax=axes1)
    plt.tight_layout()
    st.pyplot(fig1)

    # CORREÇÃO: Gráfico de Correlação IMDB
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sns.heatmap(df_limpo[cols].corr(method="spearman"), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    ax2.set_title("Correlação entre engajamento e sucesso")
    st.pyplot(fig2)

# --- 2. YouTube ---
st.header("2. Base de Trailers do YouTube e Integração")
with st.spinner("Carregando e integrando dados do YouTube..."):
    path_yt = baixar_dataset("dineshvasired/movies-youtube-trailers-and-sentimentdinesh-dinesh")
    arquivo_yt = [f for f in os.listdir(path_yt) if f.endswith(".csv")][0]
    df_yt_raw = pd.read_csv(os.path.join(path_yt, arquivo_yt))
    
    df_yt = limpar_youtube(df_yt_raw)
    df_limpo["chave"] = criar_chave(df_limpo["movie_title"])
    df_yt["chave"] = criar_chave(df_yt["name"])
    
    df_final = juntar_imdb_youtube(df_limpo, df_yt).drop_duplicates(subset=["chave", "year"]).copy()

with st.expander("Detalhes da Integração IMDb + YouTube"):
    cols_hist = ["movie_facebook_likes", "gross_imdb", "imdb_score", "favorability"]
    
    # CORREÇÃO: Histograma YouTube
    fig3, axes3 = plt.subplots(2, 2, figsize=(10, 6))
    df_final[cols_hist].hist(bins=30, ax=axes3)
    plt.tight_layout()
    st.pyplot(fig3)

    cols_corr = ["movie_facebook_likes", "cast_total_facebook_likes", "director_facebook_likes",
                 "favorability", "gross_imdb", "imdb_score", "votes"]
                 
    # CORREÇÃO: Correlação YouTube
    fig4, ax4 = plt.subplots(figsize=(9, 6))
    sns.heatmap(df_final[cols_corr].corr(method="spearman"), annot=True, cmap="coolwarm", fmt=".2f", ax=ax4)
    ax4.set_title("Correlação (base final IMDb + YouTube)")
    st.pyplot(fig4)

# --- 3. 50K Reviews ---
st.header("3. Análise de Sentimento nas Resenhas (50K Reviews)")
with st.spinner("Carregando resenhas e contando menções (pode levar 1 minuto)..."):
    path_rev = baixar_dataset("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    arquivo_rev = [f for f in os.listdir(path_rev) if f.endswith(".csv")][0]
    df_rev_raw = pd.read_csv(os.path.join(path_rev, arquivo_rev))
    
    df_rev = limpar_reviews(df_rev_raw)
    df_mencoes = contar_mencoes(df_final, df_rev)
    df_final_rev = df_final.merge(df_mencoes, on=["movie_title", "title_year"], how="left")
    df_final_rev["rev_mencoes"] = df_final_rev["rev_mencoes"].fillna(0).astype(int)
    df_final_rev.loc[df_final_rev["rev_mencoes"] < MIN_MENCOES, "rev_pct_positivo"] = np.nan

with st.expander("Resultados do cruzamento de Resenhas"):
    df_rev_ok = df_final_rev.dropna(subset=["rev_pct_positivo"])
    if len(df_rev_ok) >= 10:
        cols_rev = ["rev_mencoes", "rev_pct_positivo", "imdb_score", "favorability", "movie_facebook_likes", "gross_imdb"]
        
        # CORREÇÃO: Gráfico duplo de Resenhas
        fig5, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.heatmap(df_rev_ok[cols_rev].corr(method="spearman"), annot=True, cmap="coolwarm", fmt=".2f", ax=axes[0])
        axes[0].set_title("Correlação (filmes citados nas resenhas)", fontweight="bold")
        sns.regplot(data=df_rev_ok, x="rev_pct_positivo", y="imdb_score",
                    scatter_kws={"alpha": 0.55, "color": "#9467bd"},
                    line_kws={"color": "#111111", "linewidth": 2}, ax=axes[1])
        axes[1].set_title("% de resenhas positivas vs. Nota IMDb", fontweight="bold")
        axes[1].set_xlabel("% de resenhas positivas")
        axes[1].set_ylabel("Nota no IMDb (0 - 10)")
        plt.tight_layout()
        st.pyplot(fig5)

# --- 4. Hipóteses e Visualizações ---
st.header("4. Visualizações das Hipóteses")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Curtidas vs. Bilheteria")
    d_disp = df_final.dropna(subset=["movie_facebook_likes", "gross_imdb"]).copy()
    d_disp = d_disp[(d_disp["movie_facebook_likes"] > 0) & (d_disp["gross_imdb"] > 0)]
    d_disp["log_curtidas"] = np.log10(d_disp["movie_facebook_likes"])
    d_disp["log_bilheteria"] = np.log10(d_disp["gross_imdb"])
    
    # CORREÇÃO: Gráfico Dispersão
    fig6, ax6 = plt.subplots(figsize=(9, 5))
    sns.regplot(data=d_disp, x="log_curtidas", y="log_bilheteria", ax=ax6,
                scatter_kws={"alpha": 0.55, "color": "#1f77b4"},
                line_kws={"color": "#d62728", "linewidth": 2})
    ax6.set_title("Curtidas da Página vs. Bilheteria Bruta (log10)", fontweight="bold")
    ax6.set_xlabel("Curtidas no Facebook (log10)")
    ax6.set_ylabel("Bilheteria no IMDb em USD (log10)")
    st.pyplot(fig6)

with col2:
    st.subheader("Distribuição por Faixas")
    FAIXAS = ["Baixo Engajamento", "Médio Engajamento", "Alto Engajamento"]
    d_faixa = df_final.dropna(subset=["movie_facebook_likes", "gross_imdb"]).copy()
    d_faixa["faixa_engajamento"] = pd.qcut(d_faixa["movie_facebook_likes"], q=3, labels=FAIXAS)
    
    # CORREÇÃO: Gráfico de Caixas (Boxplot)
    fig7, ax7 = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=d_faixa, x="faixa_engajamento", y="gross_imdb", hue="faixa_engajamento",
                palette="Blues", legend=False, showmeans=True, ax=ax7,
                meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"})
    ax7.set_yscale("log")
    ax7.set_title("Distribuição da Bilheteria por Faixa de Engajamento", fontweight="bold")
    ax7.set_xlabel("Volume de Curtidas na Rede Social")
    ax7.set_ylabel("Bilheteria em USD (Escala Log)")
    st.pyplot(fig7)

st.subheader("Sentimento no Trailer (YouTube)")
d_sent = df_final.dropna(subset=["favorability", "imdb_score", "gross_imdb"]).copy()
d_sent = d_sent[d_sent["gross_imdb"] > 0]
d_sent["log_bilheteria"] = np.log10(d_sent["gross_imdb"])

# CORREÇÃO: Gráficos de Sentimento Lado a Lado
fig8, axes8 = plt.subplots(1, 2, figsize=(14, 5))
sns.regplot(data=d_sent, x="favorability", y="imdb_score", ax=axes8[0],
            scatter_kws={"alpha": 0.45, "color": "#2ca02c"},
            line_kws={"color": "#111111", "linewidth": 2})
axes8[0].set_title("Sentimento nos Comentários vs. Nota IMDb", fontweight="bold")
axes8[0].set_xlabel("Taxa de Favorabilidade (YouTube)")
axes8[0].set_ylabel("Nota no IMDb (0 - 10)")

sns.regplot(data=d_sent, x="favorability", y="log_bilheteria", ax=axes8[1],
            scatter_kws={"alpha": 0.45, "color": "#ff7f0e"},
            line_kws={"color": "#111111", "linewidth": 2})
axes8[1].set_title("Sentimento vs. Bilheteria", fontweight="bold")
axes8[1].set_xlabel("Taxa de Favorabilidade (YouTube)")
axes8[1].set_ylabel("Bilheteria em USD (log10)")
st.pyplot(fig8)

# --- 5. Teste de Hipóteses Matemático ---
st.header("5. Teste das Hipóteses")

HIPOTESES = [
    ("H1", "Curtidas do filme -> bilheteria", "movie_facebook_likes", "gross_imdb", "rede social"),
    ("H2", "Curtidas do elenco -> bilheteria", "cast_total_facebook_likes", "gross_imdb", "rede social"),
    ("H3", "Curtidas do diretor -> bilheteria", "director_facebook_likes", "gross_imdb", "rede social"),
    ("H4", "Favorabilidade do trailer -> nota IMDb", "favorability", "imdb_score", "rede social"),
    ("H5", "Favorabilidade do trailer -> bilheteria", "favorability", "gross_imdb", "rede social"),
    ("H6", "% de resenhas positivas -> nota IMDb", "rev_pct_positivo", "imdb_score", "validação"),
]

def veredito(rho, p):
    if np.isnan(rho): return "Sem dados suficientes"
    if p >= ALFA: return "Não apoiada"
    if rho <= 0: return "Não apoiada"
    if abs(rho) < 0.10: return "Apoiada (desprezível)"
    return f"Apoiada ({classificar_forca(rho)})"

linhas = []
for cod, texto, x, y, tipo in HIPOTESES:
    if x in df_final_rev.columns and y in df_final_rev.columns:
        r = correlacao_spearman(df_final_rev, x, y)
        linhas.append({
            "hipotese": cod, "descricao": texto, "tipo": tipo,
            "n": r["n"], "rho": r["rho"], "p_valor": r["p"],
            "forca": "-" if np.isnan(r["rho"]) else classificar_forca(r["rho"]),
            "veredito": veredito(r["rho"], r["p"])
        })
df_res = pd.DataFrame(linhas)
st.dataframe(df_res[["hipotese", "descricao", "n", "rho", "p_valor", "forca", "veredito"]])

# CORREÇÃO: Gráfico de Resultados
d_res = df_res.dropna(subset=["rho"]).copy()
d_res["significância"] = np.where(d_res["p_valor"] < ALFA, "p < 0,05", "p ≥ 0,05")
fig9, ax9 = plt.subplots(figsize=(9, 0.7 * len(d_res) + 2))
sns.barplot(data=d_res, y="hipotese", x="rho", hue="significância", dodge=False, ax=ax9,
            palette={"p < 0,05": "#1f77b4", "p ≥ 0,05": "#bbbbbb"})
ax9.axvline(0, color="black", linewidth=1)
for i, v in enumerate(d_res["rho"]):
    ax9.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:.2f}",
             va="center", ha="left" if v >= 0 else "right")
ax9.set_title("Correlação de Spearman por hipótese", fontweight="bold")
ax9.set_xlabel("Correlação (rho)")
ax9.set_ylabel("")
st.pyplot(fig9)
