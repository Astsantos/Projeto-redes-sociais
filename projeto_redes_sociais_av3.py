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

ALFA = 0.05        # nível de significância dos testes
MIN_MENCOES = 3    # mínimo de resenhas citando um filme para ele entrar na análise

# --- Funções de Apoio e Processamento (com Cache para otimizar o Streamlit) ---

@st.cache_data
def baixar_dataset(handle):
    """Baixa um dataset do Kaggle e devolve a pasta com os arquivos."""
    try:
        return kagglehub.dataset_download(handle)
    except Exception as erro:
        st.error(f"Não foi possível baixar '{handle}': {erro}")
        st.info("Dicas: configure as st.secrets, aceite os termos na página do dataset e confira se ele ainda existe.")
        st.stop()

def descrever_base(df, nome):
    """Devolve uma tabela com tipo e valores ausentes de cada coluna."""
    st.write(f"**{nome}:** {df.shape[0]} linhas x {df.shape[1]} colunas")
    resumo = pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "preenchidos": df.notna().sum(),
        "ausentes (%)": (df.isna().mean() * 100).round(1),
    })
    return resumo

def classificar_forca(rho):
    """Classifica a força de uma correlação pelo valor absoluto."""
    a = abs(rho)
    if a < 0.10: return "desprezível"
    if a < 0.30: return "fraca"
    if a < 0.50: return "moderada"
    return "forte"

def correlacao_spearman(df, x, y):
    """Correlação de Spearman entre x e y."""
    d = df[[x, y]].dropna()
    if len(d) < 10:
        return {"n": len(d), "rho": np.nan, "p": np.nan}
    rho, p = stats.spearmanr(d[x], d[y])
    return {"n": len(d), "rho": rho, "p": p}

def descrever_relacao(df, x, y, rotulo):
    """Imprime em uma frase o resultado da correlação entre x e y."""
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
    """Limpa o dataset IMDB 5000 para a análise de redes sociais x sucesso."""
    df = df.copy()
    colunas = [
        "movie_title", "title_year", "genres", "country", "director_name",
        "director_facebook_likes", "actor_1_facebook_likes",
        "actor_2_facebook_likes", "actor_3_facebook_likes",
        "cast_total_facebook_likes", "movie_facebook_likes",
        "num_user_for_reviews", "num_critic_for_reviews",
        "imdb_score", "gross", "budget",
    ]
    df = df[colunas]
    df = df.drop_duplicates()
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
    """Limpa a base do YouTube antes de juntar com o IMDb."""
    df = df.copy()
    df = df.drop_duplicates()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["name", "year"])
    df["year"] = df["year"].astype(int)
    df["name"] = df["name"].astype(str).str.strip()
    return df.reset_index(drop=True)

def criar_chave(titulos):
    return titulos.astype(str).str.strip().str.lower()

@st.cache_data
def juntar_imdb_youtube(df_imdb, df_youtube):
    """Junta as bases por título + ano (só os filmes que existem nas duas)."""
    return df_imdb.merge(
        df_youtube,
        left_on=["chave", "title_year"],
        right_on=["chave", "year"],
        how="inner",
        suffixes=("_imdb", "_yt"),
    )

@st.cache_data
def limpar_reviews(df):
    """Limpa as resenhas: duplicadas, tags HTML e espaços extras."""
    df = df.copy()
    df = df.dropna(subset=["review", "sentiment"])
    df = df.drop_duplicates()
    df["review"] = (df["review"]
                    .str.replace(r"<[^>]+>", " ", regex=True)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip())
    df["positivo"] = (df["sentiment"] == "positive").astype(int)
    df["n_palavras"] = df["review"].str.split().str.len()
    return df.reset_index(drop=True)

def normalizar(txt):
    """Minúsculas, sem acento e sem pontuação."""
    txt = unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode()
    txt = re.sub(r"[^a-z0-9]+", " ", txt.lower())
    return txt.strip()

@st.cache_data
def contar_mencoes(df_filmes, df_resenhas):
    """Para cada filme, conta as resenhas que citam o título e a % delas que é positiva."""
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
st.markdown("""
Esta aplicação reproduz os passos de integração e análise de três bases de dados de cinema (IMDb, YouTube e resenhas).
""")

# --- 1 e 2. Carregamento e Limpeza do IMDB ---
st.header("1. Base Principal: IMDB 5000")
with st.spinner("Baixando e carregando base IMDB..."):
    path = baixar_dataset("carolzhangdc/imdb-5000-movie-dataset")
    df_imdb_raw = pd.read_csv(os.path.join(path, "movie_metadata.csv"))

with st.expander("Visualizar dados originais do IMDB 5000"):
    st.dataframe(descrever_base(df_imdb_raw, "IMDB 5000 (original)"))

st.markdown("""
**Colunas que mais usamos:** `movie_title`, `title_year`, `genres`, `country`, `imdb_score`, `gross`, `budget`, `num_user_for_reviews`, `num_critic_for_reviews` e as curtidas no Facebook.
""")

df_limpo = limpar_dados_imdb(df_imdb_raw)

with st.expander("Exploração Inicial (Após Limpeza)"):
    st.write(f"Antes: {df_imdb_raw.shape} | Depois: {df_limpo.shape}")
    st.write("Estatísticas descritivas:")
    st.dataframe(df_limpo.describe().T)
    
    st.write("Amostra dos dados limpos:")
    st.dataframe(df_limpo[["movie_title", "title_year", "country", "movie_facebook_likes", "gross", "imdb_score"]].head(10))

    cols = ["movie_facebook_likes", "cast_total_facebook_likes", "gross", "imdb_score"]
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    df_limpo[cols].hist(bins=30, ax=fig1.axes)
    plt.tight_layout()
    st.pyplot(fig1)

    fig2 = plt.figure(figsize=(7, 5))
    sns.heatmap(df_limpo[cols].corr(method="spearman"), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlação entre engajamento e sucesso")
    st.pyplot(fig2)


# --- 4 e 5. Base do YouTube e Integração ---
st.header("2. Base de Trailers do YouTube e Integração")

with st.spinner("Carregando e integrando dados do YouTube..."):
    path_yt = baixar_dataset("dineshvasired/movies-youtube-trailers-and-sentimentdinesh-dinesh")
    arquivo_yt = [f for f in os.listdir(path_yt) if f.endswith(".csv")][0]
    df_yt_raw = pd.read_csv(os.path.join(path_yt, arquivo_yt))
    
    df_yt = limpar_youtube(df_yt_raw)
    
    df_limpo["chave"] = criar_chave(df_limpo["movie_title"])
    df_yt["chave"] = criar_chave(df_yt["name"])
    
    df_final = juntar_imdb_youtube(df_limpo, df_yt)
    df_final = df_final.drop_duplicates(subset=["chave", "year"]).copy()

with st.expander("Detalhes da Integração IMDb + YouTube"):
    st.write(f"Filmes no IMDb limpo: {len(df_limpo)}")
    st.write(f"Filmes no YouTube: {len(df_yt)}")
    st.write(f"**Filmes que casaram na base final:** {len(df_final)}")
    
    cols_hist = ["movie_facebook_likes", "gross_imdb", "imdb_score", "favorability"]
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    df_final[cols_hist].hist(bins=30, ax=fig3.axes)
    plt.tight_layout()
    st.pyplot(fig3)

    cols_corr = ["movie_facebook_likes", "cast_total_facebook_likes", "director_facebook_likes",
                 "favorability", "gross_imdb", "imdb_score", "votes"]
    fig4 = plt.figure(figsize=(9, 6))
    sns.heatmap(df_final[cols_corr].corr(method="spearman"), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlação (base final IMDb + YouTube)")
    st.pyplot(fig4)


# --- 6. Terceira base: IMDB Dataset of 50K Movie Reviews ---
st.header("3. Análise de Sentimento nas Resenhas (50K Reviews)")
st.markdown("Procuramos menções dos títulos da base final no meio de 50.000 resenhas do IMDb para calcular o percentual de aprovação.")

with st.spinner("Carregando resenhas e contando menções (isso pode levar cerca de 1 minuto)..."):
    path_rev = baixar_dataset("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    arquivo_rev = [f for f in os.listdir(path_rev) if f.endswith(".csv")][0]
    df_rev_raw = pd.read_csv(os.path.join(path_rev, arquivo_rev))
    
    df_rev = limpar_reviews(df_rev_raw)
    df_mencoes = contar_mencoes(df_final, df_rev)
    
    df_final_rev = df_final.merge(df_mencoes, on=["movie_title", "title_year"], how="left")
    df_final_rev["rev_mencoes"] = df_final_rev["rev_mencoes"].fillna(0).astype(int)
    df_final_rev.loc[df_final_rev["rev_mencoes"] < MIN_MENCOES, "rev_pct_positivo"] = np.nan

with st.expander("Resultados do cruzamento de Resenhas"):
    n_ok = df_final_rev["rev_pct_positivo"].notna().sum()
    st.write(f"Filmes da base final citados em pelo menos {MIN_MENCOES} resenhas: **{n_ok}** de {len(df_final_rev)}")
    
    df_rev_ok = df_final_rev.dropna(subset=["rev_pct_positivo"])
    if len(df_rev_ok) >= 10:
        cols_rev = ["rev_mencoes", "rev_pct_positivo", "imdb_score", "favorability", "movie_facebook_likes", "gross_imdb"]
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
        
        descrever_relacao(df_rev_ok, "rev_pct_positivo", "imdb_score", "% de resenhas positivas x nota do IMDb")


# --- 7 e 8. Hipóteses e Visualizações ---
st.header("4. Visualizações das Hipóteses")

st.markdown("""
A pergunta principal é: *o engajamento nas redes sociais pode indicar o sucesso de um filme?*
Formulamos 5 hipóteses relacionando curtidas e sentimento de trailer com bilheteria e nota no IMDb.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Curtidas vs. Bilheteria")
    def grafico_dispersao(df):
        d = df.dropna(subset=["movie_facebook_likes", "gross_imdb"]).copy()
        d = d[(d["movie_facebook_likes"] > 0) & (d["gross_imdb"] > 0)]
        d["log_curtidas"] = np.log10(d["movie_facebook_likes"])
        d["log_bilheteria"] = np.log10(d["gross_imdb"])
        fig = plt.figure(figsize=(9, 5))
        sns.regplot(data=d, x="log_curtidas", y="log_bilheteria",
                    scatter_kws={"alpha": 0.55, "color": "#1f77b4"},
                    line_kws={"color": "#d62728", "linewidth": 2})
        plt.title("Curtidas da Página vs. Bilheteria Bruta (log10)", fontweight="bold")
        plt.xlabel("Curtidas no Facebook (log10)")
        plt.ylabel("Bilheteria no IMDb em USD (log10)")
        plt.tight_layout()
        st.pyplot(fig)

    grafico_dispersao(df_final)

with col2:
    st.subheader("Distribuição por Faixas")
    FAIXAS = ["Baixo Engajamento", "Médio Engajamento", "Alto Engajamento"]
    def grafico_faixas(df):
        d = df.dropna(subset=["movie_facebook_likes", "gross_imdb"]).copy()
        d["faixa_engajamento"] = pd.qcut(d["movie_facebook_likes"], q=3, labels=FAIXAS)
        fig = plt.figure(figsize=(9, 5))
        sns.boxplot(data=d, x="faixa_engajamento", y="gross_imdb", hue="faixa_engajamento",
                    palette="Blues", legend=False, showmeans=True,
                    meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"})
        plt.yscale("log")
        plt.title("Distribuição da Bilheteria por Faixa de Engajamento", fontweight="bold")
        plt.xlabel("Volume de Curtidas na Rede Social")
        plt.ylabel("Bilheteria em USD (Escala Log)")
        plt.tight_layout()
        st.pyplot(fig)
        return d

    df_bins = grafico_faixas(df_final)

st.subheader("Sentimento no Trailer (YouTube)")
def grafico_sentimento(df):
    d = df.dropna(subset=["favorability", "imdb_score", "gross_imdb"]).copy()
    d = d[d["gross_imdb"] > 0]
    d["log_bilheteria"] = np.log10(d["gross_imdb"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.regplot(data=d, x="favorability", y="imdb_score", ax=axes[0],
                scatter_kws={"alpha": 0.45, "color": "#2ca02c"},
                line_kws={"color": "#111111", "linewidth": 2})
    axes[0].set_title("Sentimento nos Comentários vs. Nota IMDb", fontweight="bold")
    axes[0].set_xlabel("Taxa de Favorabilidade (YouTube)")
    axes[0].set_ylabel("Nota no IMDb (0 - 10)")

    sns.regplot(data=d, x="favorability", y="log_bilheteria", ax=axes[1],
                scatter_kws={"alpha": 0.45, "color": "#ff7f0e"},
                line_kws={"color": "#111111", "linewidth": 2})
    axes[1].set_title("Sentimento vs. Bilheteria", fontweight="bold")
    axes[1].set_xlabel("Taxa de Favorabilidade (YouTube)")
    axes[1].set_ylabel("Bilheteria em USD (log10)")
    plt.tight_layout()
    st.pyplot(fig)

grafico_sentimento(df_final)


# --- 9. Teste de Hipóteses Matemático ---
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
    if p >= ALFA: return "Não apoiada (relação não significativa)"
    if rho <= 0: return "Não apoiada (relação negativa)"
    if abs(rho) < 0.10: return "Apoiada, mas com efeito desprezível"
    return f"Apoiada ({classificar_forca(rho)})"

def testar_hipoteses(df, hipoteses):
    linhas = []
    for cod, texto, x, y, tipo in hipoteses:
        if x not in df.columns or y not in df.columns: continue
        r = correlacao_spearman(df, x, y)
        linhas.append({
            "hipotese": cod, "descricao": texto, "tipo": tipo, "x": x, "y": y,
            "n": r["n"], "rho": r["rho"], "p_valor": r["p"],
            "forca": "-" if np.isnan(r["rho"]) else classificar_forca(r["rho"]),
            "veredito": veredito(r["rho"], r["p"]),
        })
    return pd.DataFrame(linhas)

df_res = testar_hipoteses(df_final_rev, HIPOTESES)
tabela = df_res[["hipotese", "descricao", "n", "rho", "p_valor", "forca", "veredito"]].copy()
st.dataframe(tabela)

def grafico_resultados(df_res):
    d = df_res.dropna(subset=["rho"]).copy()
    d["significância"] = np.where(d["p_valor"] < ALFA, "p < 0,05", "p ≥ 0,05")
    fig = plt.figure(figsize=(9, 0.7 * len(d) + 2))
    sns.barplot(data=d, y="hipotese", x="rho", hue="significância", dodge=False,
                palette={"p < 0,05": "#1f77b4", "p ≥ 0,05": "#bbbbbb"})
    plt.axvline(0, color="black", linewidth=1)
    for i, v in enumerate(d["rho"]):
        plt.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:.2f}",
                 va="center", ha="left" if v >= 0 else "right")
    plt.title("Correlação de Spearman por hipótese", fontweight="bold")
    plt.xlabel("Correlação (rho)")
    plt.ylabel("")
    plt.tight_layout()
    st.pyplot(fig)

grafico_resultados(df_res)

with st.expander("Variáveis de Confusão: Orçamento e Kruskal-Wallis"):
    st.write("O orçamento explica a relação? (correlação parcial)")
    
    def correlacao_parcial_spearman(df, x, y, z):
        d = df[[x, y, z]].dropna()
        n = len(d)
        if n < 10: return {"n": n, "simples": np.nan, "parcial": np.nan, "p": np.nan}
        r = d.rank()
        def residuo(a, b):
            coef = np.polyfit(b, a, 1)
            return a - np.polyval(coef, b)
        parcial = np.corrcoef(residuo(r[x], r[z]), residuo(r[y], r[z]))[0, 1]
        t = parcial * np.sqrt((n - 3) / (1 - parcial ** 2))
        p = 2 * stats.t.sf(abs(t), n - 3)
        return {"n": n, "simples": stats.spearmanr(d[x], d[y])[0], "parcial": parcial, "p": p}

    for x, nome in [("movie_facebook_likes", "H1 curtidas do filme"), ("cast_total_facebook_likes", "H2 curtidas do elenco")]:
        r = correlacao_parcial_spearman(df_final, x, "gross_imdb", "budget_imdb")
        if np.isnan(r["parcial"]): continue
        st.write(f"- **{nome} x bilheteria** (n={r['n']}): Simples rho={r['simples']:.2f} -> Controlando Orçamento rho={r['parcial']:.2f} (p={r['p']:.3g})")

# --- 10. Conclusão ---
st.header("6. Conclusões Finais")

rede = df_res[df_res["tipo"] == "rede social"].dropna(subset=["rho"])
apoiadas = rede[rede["veredito"].str.startswith("Apoiada")]

st.markdown(f"**Hipóteses sobre redes sociais apoiadas pelos dados:** {len(apoiadas)} de {len(rede)}.")

if apoiadas.empty:
    st.error("Resposta: os dados não sustentam que o engajamento nas redes sociais indique o sucesso do filme.")
else:
    melhor = apoiadas.loc[apoiadas["rho"].idxmax()]
    if melhor["rho"] < 0.30:
        leitura = "traz alguma informação sobre o sucesso, mas pouca: sozinho, não é um bom indicador."
    elif melhor["rho"] < 0.50:
        leitura = "é um indicador parcial do sucesso: ajuda a prever, mas deixa muita coisa de fora."
    else:
        leitura = "é um bom indicador do sucesso."
        
    st.success(f"**Resposta Final:** O engajamento nas redes sociais {leitura}")
    st.markdown(f"A relação mais forte foi **{melhor['hipotese']} ({melhor['descricao']})** com força {melhor['forca']} (rho = {melhor['rho']:.2f}).")
    
st.info("Lembrete: Correlação não prova causa. Filmes maiores costumam ter mais divulgação (curtidas) e mais bilheteria ao mesmo tempo.")
