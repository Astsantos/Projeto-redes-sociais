"""
Impacto das redes sociais no sucesso de filmes (Streamlit)
Projeto de Análise de Dados (Sistemas Especialistas) - Amanda, Larissa e Raquel

Como rodar:
    pip install -r requirements.txt
    streamlit run projeto_redes_sociais_av3.py

Dados: o app baixa as 3 bases do Kaggle (precisa de KAGGLE_USERNAME e KAGGLE_KEY)
ou lê CSVs locais nas pastas dados/imdb5000, dados/youtube e dados/reviews.
"""
import os
import re
import unicodedata

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Redes sociais x sucesso de filmes", page_icon="🎬", layout="wide")
sns.set_theme(style="whitegrid")

ALFA = 0.05  # nível de significância dos testes
FAIXAS = ["Baixo Engajamento", "Médio Engajamento", "Alto Engajamento"]

# O Streamlit Cloud guarda as credenciais em "Secrets"; o kagglehub lê das variáveis de ambiente
for _k in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
    try:
        if _k in st.secrets and _k not in os.environ:
            os.environ[_k] = str(st.secrets[_k])
    except Exception:
        pass  # sem arquivo de secrets (rodando localmente): segue sem erro


# ============================================================ Carregamento
def ler_base(handle, pasta_local, arquivo=None):
    """Lê o CSV de uma base: primeiro da pasta local dados/<pasta_local>, senão baixa do Kaggle."""
    pasta = os.path.join("dados", pasta_local)
    if not (os.path.isdir(pasta) and any(f.endswith(".csv") for f in os.listdir(pasta))):
        pasta = kagglehub.dataset_download(handle)
    if arquivo and os.path.exists(os.path.join(pasta, arquivo)):
        return pd.read_csv(os.path.join(pasta, arquivo))
    csvs = sorted(f for f in os.listdir(pasta) if f.endswith(".csv"))
    return pd.read_csv(os.path.join(pasta, csvs[0]))


def descrever_base(df):
    """Tabela com tipo, preenchidos e % de ausentes de cada coluna."""
    return pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "preenchidos": df.notna().sum(),
        "ausentes (%)": (df.isna().mean() * 100).round(1),
    })


# ============================================================ Limpeza
def limpar_dados(df):
    """Limpa o dataset IMDB 5000 para a análise de redes sociais x sucesso."""
    df = df.copy()

    # Mantém só as colunas usadas nas hipóteses
    colunas = [
        "movie_title", "title_year", "genres", "country", "director_name",
        "director_facebook_likes", "actor_1_facebook_likes",
        "actor_2_facebook_likes", "actor_3_facebook_likes",
        "cast_total_facebook_likes", "movie_facebook_likes",
        "num_user_for_reviews", "num_critic_for_reviews",
        "imdb_score", "gross", "budget",
    ]
    df = df[colunas]

    # Duplicados
    df = df.drop_duplicates()

    # Padroniza texto (título vem com espaço invisível no fim)
    df["movie_title"] = df["movie_title"].str.replace("\xa0", "").str.strip()

    # Remove linhas sem as variáveis essenciais
    essenciais = ["gross", "imdb_score", "director_facebook_likes",
                  "actor_1_facebook_likes", "title_year",
                  "actor_2_facebook_likes", "actor_3_facebook_likes",
                  "num_user_for_reviews", "num_critic_for_reviews"]
    df = df.dropna(subset=essenciais)

    # Tipos
    df["title_year"] = df["title_year"].astype(int)

    # Zero em curtidas de diretor/filme = provável página inexistente
    for col in ["director_facebook_likes", "movie_facebook_likes"]:
        df[col] = df[col].replace(0, np.nan)

    # Orçamento fora dos EUA pode estar em moeda local
    df.loc[df["country"] != "USA", "budget"] = np.nan

    return df.reset_index(drop=True)


def limpar_youtube(df):
    """Limpa a base do YouTube antes de juntar com o IMDb."""
    df = df.copy()
    df = df.drop_duplicates()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["name", "year"])
    df["year"] = df["year"].astype(int)
    df["name"] = df["name"].astype(str).str.strip()
    return df.reset_index(drop=True)


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


# ============================================================ Integração
def criar_chave(titulos):
    """Título em minúsculas e sem espaços nas pontas, para casar as bases."""
    return titulos.astype(str).str.strip().str.lower()


def juntar_imdb_youtube(df_imdb, df_youtube):
    """Junta as bases por título + ano (só os filmes que existem nas duas)."""
    return df_imdb.merge(df_youtube, left_on=["chave", "title_year"],
                         right_on=["chave", "year"], how="inner",
                         suffixes=("_imdb", "_yt"))


def normalizar(txt):
    """Minúsculas, sem acento e sem pontuação (para comparar títulos e textos)."""
    txt = unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", txt.lower()).strip()


def contar_mencoes(df_filmes, df_resenhas):
    """Para cada filme, conta as resenhas que citam o título e a % delas que é positiva."""
    textos = df_resenhas["review"].map(lambda t: f" {normalizar(t)} ")  # espaços = palavra inteira
    positivo = df_resenhas["positivo"].to_numpy()

    titulos = df_filmes[["movie_title", "title_year"]].drop_duplicates().copy()
    titulos["titulo_norm"] = titulos["movie_title"].map(normalizar)
    # Descarta títulos repetidos (refilmagens) e títulos curtos demais
    titulos = titulos[~titulos["titulo_norm"].duplicated(keep=False)]
    titulos = titulos[(titulos["titulo_norm"].str.split().str.len() >= 2) |
                      (titulos["titulo_norm"].str.len() >= 10)]

    linhas = []
    for _, t in titulos.iterrows():
        achou = textos.str.contains(f" {t['titulo_norm']} ", regex=False).to_numpy()
        n = int(achou.sum())
        if n > 0:
            linhas.append({"movie_title": t["movie_title"], "title_year": t["title_year"],
                           "rev_mencoes": n, "rev_pct_positivo": positivo[achou].mean()})
    return pd.DataFrame(linhas, columns=["movie_title", "title_year", "rev_mencoes", "rev_pct_positivo"])


def montar_final_rev(df_final, df_mencoes, min_mencoes):
    """Junta as informações das resenhas na base final."""
    d = df_final.merge(df_mencoes, on=["movie_title", "title_year"], how="left")
    d["rev_mencoes"] = d["rev_mencoes"].fillna(0).astype(int)
    d.loc[d["rev_mencoes"] < min_mencoes, "rev_pct_positivo"] = np.nan  # poucas resenhas: não confiável
    return d


@st.cache_resource(show_spinner="Carregando e preparando as 3 bases (na primeira vez pode levar alguns minutos)...")
def preparar_bases():
    """Baixa, limpa e integra as três bases. O resultado fica em cache."""
    imdb = ler_base("carolzhangdc/imdb-5000-movie-dataset", "imdb5000", "movie_metadata.csv")
    limpo = limpar_dados(imdb)

    yt = ler_base("paultimothymooney/movies-youtube-trailers-sentiment", "youtube")
    yt_limpo = limpar_youtube(yt)

    limpo["chave"] = criar_chave(limpo["movie_title"])
    yt_limpo["chave"] = criar_chave(yt_limpo["name"])
    final = juntar_imdb_youtube(limpo, yt_limpo)
    n_casaram = len(final)
    final = final.drop_duplicates(subset=["chave", "year"]).copy()

    rev = ler_base("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews", "reviews")
    rev_limpo = limpar_reviews(rev)
    mencoes = contar_mencoes(final, rev_limpo)

    return {
        "imdb_raw": imdb.shape, "imdb_tab": descrever_base(imdb), "limpo": limpo,
        "yt_raw": yt.shape, "yt_tab": descrever_base(yt), "yt": yt_limpo,
        "final": final, "n_casaram": n_casaram,
        "rev_raw": rev.shape, "rev_n": len(rev_limpo),
        "rev": rev_limpo[["sentiment", "positivo", "n_palavras"]],  # sem o texto (economiza memória)
        "mencoes": mencoes,
    }


# ============================================================ Estatística
def classificar_forca(rho):
    """Classifica a força de uma correlação pelo valor absoluto."""
    a = abs(rho)
    if a < 0.10:
        return "desprezível"
    if a < 0.30:
        return "fraca"
    if a < 0.50:
        return "moderada"
    return "forte"


def correlacao_spearman(df, x, y):
    """Spearman entre x e y nas linhas que têm os dois valores (os dados são muito assimétricos)."""
    d = df[[x, y]].dropna()
    if len(d) < 10:
        return {"n": len(d), "rho": np.nan, "p": np.nan}
    rho, p = stats.spearmanr(d[x], d[y])
    return {"n": len(d), "rho": rho, "p": p}


def frase_relacao(df, x, y, rotulo):
    """Resultado da correlação entre x e y em uma frase."""
    r = correlacao_spearman(df, x, y)
    if np.isnan(r["rho"]):
        return f"**{rotulo}:** poucos dados (n = {r['n']})."
    sentido = "positiva" if r["rho"] > 0 else "negativa"
    signif = "estatisticamente significativa" if r["p"] < ALFA else "não significativa"
    return (f"**{rotulo}:** Spearman = {r['rho']:.2f} (n = {r['n']}, p = {r['p']:.3g}) "
            f"→ relação {classificar_forca(r['rho'])} {sentido}, {signif}.")


HIPOTESES = [
    # (código, descrição, x (engajamento), y (sucesso), tipo)
    ("H1", "Curtidas do filme -> bilheteria", "movie_facebook_likes", "gross_imdb", "rede social"),
    ("H2", "Curtidas do elenco -> bilheteria", "cast_total_facebook_likes", "gross_imdb", "rede social"),
    ("H3", "Curtidas do diretor -> bilheteria", "director_facebook_likes", "gross_imdb", "rede social"),
    ("H4", "Favorabilidade do trailer -> nota IMDb", "favorability", "imdb_score", "rede social"),
    ("H5", "Favorabilidade do trailer -> bilheteria", "favorability", "gross_imdb", "rede social"),
    ("H6", "% de resenhas positivas -> nota IMDb (validação)", "rev_pct_positivo", "imdb_score", "validação"),
]


def veredito(rho, p):
    """Diz se a hipótese (relação positiva) é apoiada pelos dados."""
    if np.isnan(rho):
        return "Sem dados suficientes"
    if p >= ALFA:
        return "Não apoiada (relação não significativa)"
    if rho <= 0:
        return "Não apoiada (relação negativa)"
    if abs(rho) < 0.10:
        return "Apoiada, mas com efeito desprezível"
    return f"Apoiada ({classificar_forca(rho)})"


def testar_hipoteses(df):
    """Roda Spearman para cada hipótese e devolve uma tabela com os resultados."""
    linhas = []
    for cod, texto, x, y, tipo in HIPOTESES:
        r = correlacao_spearman(df, x, y)
        linhas.append({
            "hipotese": cod, "descricao": texto, "tipo": tipo, "n": r["n"],
            "rho": r["rho"], "p_valor": r["p"],
            "forca": "-" if np.isnan(r["rho"]) else classificar_forca(r["rho"]),
            "veredito": veredito(r["rho"], r["p"]),
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
    """Correlação (em postos) entre x e y depois de tirar o efeito de z, comparada com a simples."""
    d = df[[x, y, z]].dropna()
    n = len(d)
    if n < 10:
        return {"n": n, "simples": np.nan, "parcial": np.nan, "p": np.nan}
    r = d.rank()

    def residuo(a, b):
        return a - np.polyval(np.polyfit(b, a, 1), b)

    parcial = np.corrcoef(residuo(r[x], r[z]), residuo(r[y], r[z]))[0, 1]
    t = parcial * np.sqrt((n - 3) / (1 - parcial ** 2))
    p = 2 * stats.t.sf(abs(t), n - 3)  # p-valor aproximado
    return {"n": n, "simples": stats.spearmanr(d[x], d[y])[0], "parcial": parcial, "p": p}


def linhas_conclusao(df_res):
    """Veredito de cada hipótese e resposta à pergunta principal."""
    saida = []
    for _, l in df_res.iterrows():
        rho = "sem dados" if np.isnan(l["rho"]) else f"rho = {l['rho']:.2f}, n = {l['n']}"
        saida.append(f"**{l['hipotese']}** ({l['descricao']}): {l['veredito']} [{rho}]")

    rede = df_res[df_res["tipo"] == "rede social"].dropna(subset=["rho"])
    apoiadas = rede[rede["veredito"].str.startswith("Apoiada")]
    resumo = [f"Hipóteses sobre redes sociais apoiadas pelos dados: **{len(apoiadas)} de {len(rede)}**."]
    if apoiadas.empty:
        resumo.append("**Resposta:** os dados não sustentam que o engajamento nas redes sociais indique o sucesso do filme.")
        return saida, resumo

    melhor = apoiadas.loc[apoiadas["rho"].idxmax()]
    resumo.append(f"A relação mais forte foi {melhor['hipotese']} ({melhor['descricao']}): "
                  f"rho = {melhor['rho']:.2f}, {melhor['forca']}.")
    if melhor["rho"] < 0.30:
        leitura = "traz alguma informação sobre o sucesso, mas pouca: sozinho, não é um bom indicador."
    elif melhor["rho"] < 0.50:
        leitura = "é um indicador parcial do sucesso: ajuda a prever, mas deixa muita coisa de fora."
    else:
        leitura = "é um bom indicador do sucesso."
    resumo.append(f"**Resposta:** o engajamento nas redes sociais {leitura}")
    resumo.append("Lembrete: correlação não prova causa.")
    return saida, resumo


# ============================================================ Gráficos (devolvem a figura)
def mostrar(fig):
    st.pyplot(fig)
    plt.close(fig)


def fig_histogramas(df, cols):
    axes = df[cols].hist(bins=30, figsize=(10, 6))
    fig = axes.flat[0].figure
    fig.tight_layout()
    return fig


def fig_heatmap(df, cols, titulo):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.heatmap(df[cols].corr(method="spearman"), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title(titulo, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_dispersao(df):
    """Curtidas da página do filme x bilheteria, em log10."""
    d = df.dropna(subset=["movie_facebook_likes", "gross_imdb"]).copy()
    d = d[(d["movie_facebook_likes"] > 0) & (d["gross_imdb"] > 0)]
    d["log_curtidas"] = np.log10(d["movie_facebook_likes"])
    d["log_bilheteria"] = np.log10(d["gross_imdb"])
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.regplot(data=d, x="log_curtidas", y="log_bilheteria", ax=ax,
                scatter_kws={"alpha": 0.55, "color": "#1f77b4"},
                line_kws={"color": "#d62728", "linewidth": 2})
    ax.set_title("Curtidas da Página do Filme vs. Bilheteria Bruta (log10)", fontweight="bold")
    ax.set_xlabel("Curtidas no Facebook (log10)")
    ax.set_ylabel("Bilheteria no IMDb em USD (log10)")
    fig.tight_layout()
    return fig


def fig_sentimento(df):
    """Favorabilidade dos comentários do trailer x nota IMDb e x bilheteria (log10)."""
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
    axes[1].set_title("Sentimento nos Comentários vs. Bilheteria", fontweight="bold")
    axes[1].set_xlabel("Taxa de Favorabilidade (YouTube)")
    axes[1].set_ylabel("Bilheteria em USD (log10)")
    fig.tight_layout()
    return fig


def fig_faixas(df):
    """Boxplot da bilheteria por faixa de curtidas (tercis). Devolve a figura e a base com a faixa."""
    d = df.dropna(subset=["movie_facebook_likes", "gross_imdb"]).copy()
    d["faixa_engajamento"] = pd.qcut(d["movie_facebook_likes"], q=3, labels=FAIXAS)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=d, x="faixa_engajamento", y="gross_imdb", hue="faixa_engajamento",
                palette="Blues", legend=False, showmeans=True, ax=ax,
                meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"})
    ax.set_yscale("log")
    ax.set_title("Distribuição da Bilheteria por Faixa de Engajamento", fontweight="bold")
    ax.set_xlabel("Volume de Curtidas na Rede Social")
    ax.set_ylabel("Bilheteria em USD (Escala Log)")
    fig.tight_layout()
    return fig, d


def fig_resultados(df_res):
    """Barras com a correlação de cada hipótese; azul = significativa (p < 0,05)."""
    d = df_res.dropna(subset=["rho"]).copy()
    d["significância"] = np.where(d["p_valor"] < ALFA, "p < 0,05", "p ≥ 0,05")
    fig, ax = plt.subplots(figsize=(9, 0.7 * len(d) + 2))
    sns.barplot(data=d, y="hipotese", x="rho", hue="significância", dodge=False, ax=ax,
                palette={"p < 0,05": "#1f77b4", "p ≥ 0,05": "#bbbbbb"})
    ax.axvline(0, color="black", linewidth=1)
    for i, v in enumerate(d["rho"]):
        ax.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:.2f}",
                va="center", ha="left" if v >= 0 else "right")
    ax.set_title("Correlação de Spearman por hipótese", fontweight="bold")
    ax.set_xlabel("Correlação (rho)")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


# ============================================================ Seções do app
def secao_visao_geral():
    st.title("🎬 Impacto das redes sociais no sucesso de filmes")
    st.caption("Projeto de Análise de Dados (Sistemas Especialistas) — Amanda, Larissa e Raquel")
    st.markdown("""
**Pergunta principal:** O engajamento nas redes sociais pode indicar o sucesso de um filme?

**Como medimos:**
- *Engajamento nas redes:* curtidas no Facebook (filme, elenco e diretor) e sentimento dos comentários dos trailers no YouTube.
- *Sucesso:* bilheteria (`gross`) e nota no IMDb (`imdb_score`).

### Bases de dados (todas públicas, no Kaggle)
| Base | Origem | O que traz | Por que usamos |
|---|---|---|---|
| IMDB 5000 Movie Dataset | [carolzhangdc/imdb-5000-movie-dataset](https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset) | 5.043 filmes × 28 colunas: bilheteria, orçamento, nota e curtidas no Facebook | Base principal: tem o engajamento e o sucesso do mesmo filme |
| Movies YouTube Trailers & Sentiment | [paultimothymooney/movies-youtube-trailers-sentiment](https://www.kaggle.com/datasets/paultimothymooney/movies-youtube-trailers-sentiment) | Sentimento (`favorability`) dos comentários do trailer | Acrescenta uma segunda rede social (YouTube) |
| IMDB Dataset of 50K Movie Reviews | [lakshmi25npathi/imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | 50.000 resenhas em texto, positivas ou negativas | Acrescenta a opinião do público em texto, usada para validar a nota do IMDb |

Use o menu à esquerda para navegar pelas etapas do projeto.
""")


def secao_dados(B):
    st.header("1. Dados e limpeza")

    st.subheader("IMDB 5000")
    c1, c2, c3 = st.columns(3)
    c1.metric("Filmes originais", f"{B['imdb_raw'][0]:,}".replace(",", "."))
    c2.metric("Colunas originais", B["imdb_raw"][1])
    c3.metric("Filmes após a limpeza", f"{len(B['limpo']):,}".replace(",", "."))
    with st.expander("Dicionário das colunas originais (tipo e valores ausentes)"):
        st.dataframe(B["imdb_tab"], width="stretch")
    st.markdown("""
**Decisões de limpeza**
- **Duplicados:** removemos os filmes repetidos, pois contariam o mesmo filme duas vezes.
- **Títulos:** removemos espaços invisíveis no final para padronizar o texto.
- **Bilheteria e curtidas ausentes:** removemos os filmes sem essas variáveis, essenciais às hipóteses.
- **Zeros em curtidas de diretor e de filme:** tratados como "sem dado", pois quase metade dos filmes com zero curtidas não é realista (provavelmente a página não existia).
- **Orçamento:** mantido só para filmes dos EUA. Os maiores valores eram de filmes da Coreia do Sul, Japão e Índia, provavelmente em moeda local.
- **Bilheteria alta (blockbusters):** mantida, pois são valores reais.
""")

    st.subheader("YouTube (comentários dos trailers)")
    c1, c2 = st.columns(2)
    c1.metric("Filmes originais", f"{B['yt_raw'][0]:,}".replace(",", "."))
    c2.metric("Filmes após a limpeza", f"{len(B['yt']):,}".replace(",", "."))
    with st.expander("Dicionário das colunas (tipo e valores ausentes)"):
        st.dataframe(B["yt_tab"], width="stretch")
    st.markdown("""
**Decisões de limpeza**
- **Duplicados:** linhas totalmente repetidas foram removidas.
- **Nome e ano ausentes:** removidos, porque o casamento com o IMDb depende dos dois.
- **Tipos:** o ano virou número inteiro e o nome teve os espaços das pontas removidos.
- **Favorabilidade ausente:** os filmes continuam na base e só deixam de entrar nas análises que usam essa variável.
""")

    st.subheader("Integração IMDb + YouTube")
    st.markdown("A ligação é feita pelo **título (em minúsculas) + ano**. Só o título não basta, porque existem filmes com o mesmo nome em anos diferentes.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filmes no IMDb limpo", len(B["limpo"]))
    c2.metric("Filmes no YouTube", len(B["yt"]))
    c3.metric("Linhas que casaram", B["n_casaram"])
    c4.metric("Filmes únicos na base final", len(B["final"]))

    st.subheader("Resenhas (IMDB 50K)")
    c1, c2 = st.columns(2)
    c1.metric("Resenhas originais", f"{B['rev_raw'][0]:,}".replace(",", "."))
    c2.metric("Resenhas após a limpeza", f"{B['rev_n']:,}".replace(",", "."))
    st.markdown("Removemos resenhas duplicadas, tags HTML (`<br />`) e espaços extras.")


def secao_exploracao(B):
    st.header("2. Exploração dos dados")
    limpo, final = B["limpo"], B["final"]

    st.subheader("Base do IMDb (após a limpeza)")
    st.dataframe(limpo.describe().T, width="stretch")
    cols = ["movie_facebook_likes", "cast_total_facebook_likes", "gross", "imdb_score"]
    c1, c2 = st.columns(2)
    with c1:
        mostrar(fig_histogramas(limpo, cols))
    with c2:
        mostrar(fig_heatmap(limpo, cols, "Correlação entre engajamento e sucesso"))
    st.markdown("""
**Principais observações**
- Curtidas e bilheteria são muito assimétricas: poucos filmes concentram valores enormes (mediana bem menor que a média).
- `movie_facebook_likes` tem valores preenchidos só em cerca de metade da base.
- A correlação mais forte é entre curtidas do filme e bilheteria (0,60).
- A fama do elenco tem relação fraca com bilheteria (0,38) e quase nula com a nota do IMDb (0,05).
- Bilheteria e nota do IMDb quase não se relacionam (0,10).
- Correlação não prova causa: as relações são testadas na seção de hipóteses.
""")

    st.subheader("Base final (IMDb + YouTube)")
    cols_final = ["movie_facebook_likes", "director_facebook_likes", "cast_total_facebook_likes",
                  "gross_imdb", "budget_imdb", "imdb_score", "favorability", "votes", "runtime"]
    st.dataframe(final[[c for c in cols_final if c in final.columns]].describe().T, width="stretch")
    cols_hist = ["movie_facebook_likes", "gross_imdb", "imdb_score", "favorability"]
    cols_corr = ["movie_facebook_likes", "cast_total_facebook_likes", "director_facebook_likes",
                 "favorability", "gross_imdb", "imdb_score", "votes"]
    c1, c2 = st.columns(2)
    with c1:
        mostrar(fig_histogramas(final, cols_hist))
    with c2:
        mostrar(fig_heatmap(final, cols_corr, "Correlação (base final IMDb + YouTube)"))
    st.markdown("""
**Principais observações**
- Esses filmes são mais populares que a média do IMDb 5000 (mediana de curtidas do filme de 17.000 contra 5.000 na base original).
- Curtidas e bilheteria continuam muito assimétricas; nota do IMDb e favorabilidade têm formato de sino.
- A correlação entre curtidas do filme e bilheteria caiu de 0,60 (base do IMDb) para 0,36 (base final).
- A favorabilidade dos comentários tem correlação fraca com todas as variáveis (0,05 a 0,27), a maior com a nota do IMDb.
- O número de votos no IMDb é a variável mais ligada às demais (0,64 com curtidas do filme, 0,63 com bilheteria).
- **Limitações:** base concentrada em uma época, só filmes com trailer comentado no YouTube, e correlação não prova causa.
""")


def secao_resenhas(B, df_final_rev, min_mencoes):
    st.header("3. Terceira base: 50 mil resenhas")
    st.markdown("""
A base tem só duas colunas (`review` e `sentiment`), **sem título nem ano do filme**. Por isso não dá para juntar linha a linha. Para ligar as resenhas aos filmes, procuramos o título de cada filme dentro do texto das resenhas.
""")
    rev = B["rev"]
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=rev, x="sentiment", hue="sentiment", order=["positive", "negative"],
                      palette=["#2ca02c", "#d62728"], legend=False, ax=ax)
        ax.set_title("Resenhas por sentimento", fontweight="bold")
        ax.set_xlabel("Sentimento")
        ax.set_ylabel("Quantidade")
        mostrar(fig)
    with c2:
        limite = rev["n_palavras"].quantile(0.99)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=rev[rev["n_palavras"] <= limite], x="n_palavras", hue="sentiment", bins=50, ax=ax)
        ax.set_title("Tamanho das resenhas (sem o 1% mais longo)", fontweight="bold")
        ax.set_xlabel("Número de palavras")
        mostrar(fig)

    st.subheader("Ligação com os filmes: título dentro do texto")
    st.markdown(f"""
Uma resenha "cita" um filme quando o título aparece no texto (sem diferenciar maiúsculas, acentos e pontuação, e só como palavra inteira). Para cada filme calculamos `rev_mencoes` (quantas resenhas citam o título) e `rev_pct_positivo` (% delas que é positiva).

Cuidados para reduzir erros: só usamos títulos com 2 ou mais palavras ou com 10 ou mais letras; descartamos títulos repetidos (refilmagens); e só analisamos filmes com pelo menos **{min_mencoes}** resenhas (ajuste no menu à esquerda).
""")
    ok = df_final_rev.dropna(subset=["rev_pct_positivo"])
    c1, c2 = st.columns(2)
    c1.metric("Filmes citados em pelo menos 1 resenha", len(B["mencoes"]))
    c2.metric(f"Filmes com pelo menos {min_mencoes} resenhas", len(ok))

    if len(ok) < 10:
        st.warning(f"Só {len(ok)} filmes com resenhas suficientes. Diminua o mínimo de resenhas no menu à esquerda.")
    else:
        st.dataframe(ok[["movie_title", "title_year", "rev_mencoes", "rev_pct_positivo", "imdb_score", "favorability"]]
                     .sort_values("rev_mencoes", ascending=False).head(10), width="stretch")
        cols_rev = ["rev_mencoes", "rev_pct_positivo", "imdb_score", "favorability", "movie_facebook_likes", "gross_imdb"]
        c1, c2 = st.columns(2)
        with c1:
            mostrar(fig_heatmap(ok, cols_rev, "Correlação (filmes citados nas resenhas)"))
        with c2:
            fig, ax = plt.subplots(figsize=(8, 5.5))
            sns.regplot(data=ok, x="rev_pct_positivo", y="imdb_score", ax=ax,
                        scatter_kws={"alpha": 0.55, "color": "#9467bd"},
                        line_kws={"color": "#111111", "linewidth": 2})
            ax.set_title("% de resenhas positivas vs. Nota IMDb", fontweight="bold")
            ax.set_xlabel("% de resenhas positivas (base de 50K resenhas)")
            ax.set_ylabel("Nota no IMDb (0 - 10)")
            mostrar(fig)
        st.info(frase_relacao(ok, "rev_pct_positivo", "imdb_score", "% de resenhas positivas x nota do IMDb"))
        st.info(frase_relacao(ok, "rev_mencoes", "gross_imdb", "Nº de resenhas que citam o filme x bilheteria"))
    st.markdown("""
**Como ler:** as resenhas vêm do próprio IMDb, então a % de resenhas positivas *deveria* acompanhar a nota do IMDb. Se acompanha, a ligação por título está funcionando.

**Limitações:** a base não informa o filme de cada resenha (citar o título não garante que a resenha seja sobre ele); ela não aumenta o número de filmes, só acrescenta opinião em texto para os filmes citados; e a maioria das resenhas é de filmes até o início dos anos 2010.
""")


def secao_hipoteses():
    st.header("4. Hipóteses e plano de análise")
    st.markdown("""
Cada hipótese liga uma medida de engajamento a uma medida de sucesso e espera uma relação **positiva**.

| Código | Hipótese | Engajamento (x) | Sucesso (y) |
|---|---|---|---|
| H1 | Filmes com mais curtidas na página do Facebook têm maior bilheteria | `movie_facebook_likes` | `gross_imdb` |
| H2 | Filmes com elenco mais curtido têm maior bilheteria | `cast_total_facebook_likes` | `gross_imdb` |
| H3 | Filmes com diretor mais curtido têm maior bilheteria | `director_facebook_likes` | `gross_imdb` |
| H4 | Trailers com comentários mais positivos no YouTube têm maior nota no IMDb | `favorability` | `imdb_score` |
| H5 | Trailers com comentários mais positivos no YouTube têm maior bilheteria | `favorability` | `gross_imdb` |
| H6 | *(validação)* Filmes com mais resenhas positivas na base de 50K têm maior nota no IMDb | `rev_pct_positivo` | `imdb_score` |

**Plano de análise**
1. Visualizar as relações (dispersão, sentimento do trailer e faixas de engajamento).
2. Testar cada hipótese com correlação de **Spearman** e p-valor. Escolhemos Spearman porque curtidas e bilheteria são muito assimétricas.
3. Considerar uma hipótese **apoiada pelos dados** quando a correlação é positiva e o p-valor é menor que 0,05. Informamos também a força da relação, porque com muitos filmes até uma relação minúscula pode ser significativa.
4. Comparar a bilheteria entre faixas de engajamento (baixo, médio, alto) com o teste de Kruskal-Wallis.
5. Checar se o **orçamento** explica a relação, com a correlação parcial controlando o orçamento.
""")


def secao_visualizacoes(B):
    st.header("5. Visualizações")
    final = B["final"]

    st.subheader("Gráfico 1: Curtidas vs. Bilheteria")
    st.caption("Cada ponto é um filme. Os eixos estão em log10, porque poucos filmes têm valores enormes. A reta vermelha é a tendência ajustada nos valores em log.")
    mostrar(fig_dispersao(final))
    st.info(frase_relacao(final, "movie_facebook_likes", "gross_imdb", "Curtidas do filme x bilheteria"))
    st.markdown("**Leitura:** filmes com mais curtidas tendem a ter mais bilheteria, mas os pontos ficam espalhados em volta da reta. Na análise exploratória, essa correlação caiu de 0,60 (base IMDb) para 0,36 (base final): as curtidas dão uma pista da bilheteria, mas não a determinam.")

    st.subheader("Gráfico 2: Sentimento no trailer vs. Nota e Bilheteria")
    st.caption("O eixo x é a taxa de favorabilidade dos comentários do trailer no YouTube (quanto maior, mais positivos).")
    mostrar(fig_sentimento(final))
    st.info(frase_relacao(final, "favorability", "imdb_score", "Favorabilidade x nota IMDb"))
    st.info(frase_relacao(final, "favorability", "gross_imdb", "Favorabilidade x bilheteria"))
    st.markdown("**Leitura:** os pontos formam uma nuvem larga. A favorabilidade teve correlação fraca com todas as variáveis (0,05 a 0,27), e a maior foi com a nota do IMDb: o sentimento do trailer diz um pouco sobre a *qualidade percebida* e quase nada sobre quanto o filme arrecada.")

    st.subheader("Gráfico 3: Bilheteria por faixa de engajamento")
    st.caption("Três grupos de mesmo tamanho pelas curtidas da página. A linha dentro da caixa é a mediana; o ponto vermelho é a média.")
    fig, d = fig_faixas(final)
    mostrar(fig)
    med = d.groupby("faixa_engajamento", observed=True)["gross_imdb"].median()
    st.dataframe(med.round(0).rename("Mediana da bilheteria (USD)"), width="stretch")
    st.info("A mediana da bilheteria " + ("aumenta" if med.is_monotonic_increasing else "não aumenta de forma contínua")
            + " da faixa de baixo para a de alto engajamento.")
    st.markdown("**Leitura:** as caixas se sobrepõem bastante: existem filmes de baixo engajamento com bilheteria maior que a de filmes de alto engajamento. O teste de Kruskal-Wallis (próxima seção) diz se a diferença entre as faixas é grande demais para ser acaso.")


def secao_testes(B, df_final_rev):
    st.header("6. Teste das hipóteses")
    st.markdown("Uma hipótese é **apoiada pelos dados** quando a correlação de Spearman é positiva e `p < 0,05`.")
    df_res = testar_hipoteses(df_final_rev)

    tabela = df_res[["hipotese", "descricao", "n", "rho", "p_valor", "forca", "veredito"]].copy()
    tabela["rho"] = tabela["rho"].round(2)
    tabela["p_valor"] = tabela["p_valor"].map(lambda p: f"{p:.3g}")
    st.dataframe(tabela, width="stretch", hide_index=True)
    st.download_button("Baixar resultados (CSV)", df_res.to_csv(index=False).encode("utf-8"),
                       "resultados_hipoteses.csv", "text/csv")
    if df_res["rho"].notna().any():
        mostrar(fig_resultados(df_res))

    st.subheader("Comparação entre faixas de engajamento (Kruskal-Wallis)")
    medianas, h, p = teste_faixas(B["final"])
    st.dataframe(medianas.round(0).rename("Mediana da bilheteria (USD)"), width="stretch")
    st.info(f"H = {h:.2f}, p = {p:.3g} → " +
            ("há diferença significativa entre as faixas." if p < ALFA else "não há diferença significativa entre as faixas."))

    st.subheader("O orçamento explica a relação?")
    st.markdown("Filmes caros costumam ter mais divulgação (mais curtidas) e mais bilheteria ao mesmo tempo. Se a correlação cai bastante depois de controlar o orçamento, o engajamento explica pouco além do dinheiro investido. Só temos orçamento dos filmes dos EUA, por isso o *n* é menor.")
    for x, nome in [("movie_facebook_likes", "H1 curtidas do filme"), ("cast_total_facebook_likes", "H2 curtidas do elenco")]:
        r = correlacao_parcial_spearman(B["final"], x, "gross_imdb", "budget_imdb")
        if np.isnan(r["parcial"]):
            st.write(f"{nome}: poucos filmes com orçamento (n = {r['n']}).")
        else:
            st.info(f"**{nome} x bilheteria** (n = {r['n']}): rho = {r['simples']:.2f} → "
                    f"rho controlando o orçamento = {r['parcial']:.2f} (p = {r['p']:.3g})")
    return df_res


def secao_conclusoes(df_res):
    st.header("7. Conclusões e limitações")
    st.subheader("Resposta às hipóteses")
    saida, resumo = linhas_conclusao(df_res)
    for linha in saida:
        st.markdown(f"- {linha}")
    st.markdown("---")
    for linha in resumo:
        st.markdown(linha)

    st.subheader("Principais insights (valores da análise exploratória)")
    st.markdown("""
- **Curtidas do filme são o melhor sinal de bilheteria** entre as medidas de rede social: correlação de 0,60 na base do IMDb e 0,36 na base final.
- **O sentimento do trailer quase não se liga à bilheteria.** Ele se liga um pouco mais à nota do IMDb (0,27): mede mais a qualidade percebida do que o sucesso comercial.
- **A fama do elenco ajuda na bilheteria, mas não na nota:** 0,38 com a bilheteria e 0,05 com a nota do IMDb.
- **O número de votos no IMDb é a variável mais ligada às demais** (0,64 com as curtidas do filme e 0,63 com a bilheteria).
- **Bilheteria e nota quase não se relacionam** (0,10): um filme pode ser sucesso de público sem ser bem avaliado, e o contrário.
""")
    st.subheader("Limitações")
    st.markdown("""
- **Viés de seleção:** a base final só inclui filmes com trailer comentado no YouTube e é mais popular que a média.
- **Período:** os filmes se concentram entre 2006 e 2012, e as curtidas do Facebook foram medidas em um momento só.
- **Dados ausentes:** a página do filme no Facebook só existe para cerca de metade dos filmes, e o orçamento só foi mantido para filmes dos EUA.
- **Resenhas:** a ligação com os filmes é feita pelo título dentro do texto, o que gera erros.
- **Causalidade:** correlação não prova causa.
""")
    st.subheader("Próximos passos")
    st.markdown("""
- Incluir dados de redes sociais atuais (Instagram, TikTok, X).
- Usar um modelo de análise de sentimento sobre o texto dos comentários e das resenhas.
- Testar um modelo de previsão (por exemplo, regressão) combinando orçamento, elenco e engajamento.
""")


# ============================================================ Navegação
SECOES = ["Visão geral", "1. Dados e limpeza", "2. Exploração", "3. Resenhas (50K)",
          "4. Hipóteses", "5. Visualizações", "6. Teste das hipóteses", "7. Conclusões"]

st.sidebar.title("Navegação")
secao = st.sidebar.radio("Seção", SECOES, label_visibility="collapsed")
min_mencoes = st.sidebar.slider("Mínimo de resenhas por filme", 1, 10, 3,
                                help="Filmes citados em menos resenhas ficam fora da análise das resenhas.")

if secao == "Visão geral":
    secao_visao_geral()
    st.stop()
if secao == "4. Hipóteses":
    secao_hipoteses()
    st.stop()

try:
    B = preparar_bases()
except Exception as erro:
    st.error(f"Não foi possível carregar os dados: {erro}")
    st.markdown("""
**Como resolver**
- No Streamlit Cloud: em *App settings > Secrets*, adicione `KAGGLE_USERNAME = "seu_usuario"` e `KAGGLE_KEY = "sua_chave"` (a chave é gerada em kaggle.com > Settings > API).
- Se o erro for 403, abra a página de cada dataset no Kaggle e aceite os termos de uso.
- Alternativa sem Kaggle: coloque os CSVs em `dados/imdb5000/`, `dados/youtube/` e `dados/reviews/` no repositório.
""")
    st.stop()

df_final_rev = montar_final_rev(B["final"], B["mencoes"], min_mencoes)

if secao == "1. Dados e limpeza":
    secao_dados(B)
elif secao == "2. Exploração":
    secao_exploracao(B)
elif secao == "3. Resenhas (50K)":
    secao_resenhas(B, df_final_rev, min_mencoes)
elif secao == "5. Visualizações":
    secao_visualizacoes(B)
elif secao == "6. Teste das hipóteses":
    secao_testes(B, df_final_rev)
elif secao == "7. Conclusões":
    secao_conclusoes(testar_hipoteses(df_final_rev))
