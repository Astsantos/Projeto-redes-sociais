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
FAIXAS = ["Baixo", "Médio", "Alto"]
 
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
    st.pyplot(fig)
    plt.close(fig)
 
def descrever_base(df):
    """Tabela com o tipo, o nº de valores preenchidos e a % de ausentes de cada coluna."""
    return pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "preenchidos": df.notna().sum(),
        "ausentes (%)": (df.isna().mean() * 100).round(1),
    })
 
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
    p = 2 * stats.t.sf(abs(t), n - 3)   # p-valor aproximado (t com n-3 graus de liberdade)
    return {"n": n, "simples": stats.spearmanr(d[x], d[y])[0], "parcial": parcial, "p": p}
 
# =====================================================================
# INÍCIO DA INTERFACE
# =====================================================================
 
st.title("O engajamento nas redes sociais indica o sucesso de um filme?")
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
 
    # Filmes do YouTube que não casaram com o IMDb (e quantos existem lá com outro ano)
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
 
# --- Explorando as Bases (Usando Expander para limpar a tela) ---
with st.expander("Explorar as Bases de Dados (Visão Geral)"):
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
            mostrar(fig1)
 
        st.markdown("#### Exploração inicial")
        st.markdown("**Estatísticas descritivas da base limpa**")
        st.dataframe(df_limpo.describe().T)
        st.markdown("""
**Principais observações da exploração inicial**
 
- A base tem 4059 filmes e 16 variáveis, depois da limpeza.
- Curtidas e bilheteria são muito assimétricas: poucos filmes concentram valores enormes (mediana bem menor que a média).
- `movie_facebook_likes` tem só 2198 valores preenchidos (cerca de metade da base).
- A correlação mais forte é entre curtidas do filme e bilheteria (0,60).
- A fama do elenco tem relação fraca com bilheteria (0,38) e quase nula com a nota do IMDb (0,05).
- Bilheteria e nota do IMDb quase não se relacionam (0,10).
- Correlação não prova causa: as relações precisam ser testadas na análise das hipóteses.
""")
        st.markdown("**Amostra dos dados limpos**")
        st.dataframe(df_limpo[["movie_title", "title_year", "country", "movie_facebook_likes", "gross", "imdb_score"]].head(10))
 
        st.markdown("**Dicionário das colunas originais (tipo e valores ausentes)**")
        st.dataframe(descrever_base(df_imdb_raw))
        st.markdown("""
**Decisões de limpeza**
- **Duplicados:** removemos 45 filmes repetidos, pois contariam o mesmo filme duas vezes.
- **Títulos:** removemos espaços invisíveis no final para padronizar o texto.
- **Bilheteria e curtidas ausentes:** removemos os filmes sem essas variáveis, essenciais às hipóteses. O dataset passou de 5.043 para 4.059 filmes.
- **Zeros em curtidas de diretor e de filme:** tratados como "sem dado", pois quase metade dos filmes com zero curtidas não é realista (provavelmente a página não existia).
- **Orçamento:** mantido só para filmes dos EUA. Os maiores valores eram de filmes da Coreia do Sul, Japão e Índia, provavelmente em moeda local. Restaram 3.002 filmes com orçamento.
- **Bilheteria alta (blockbusters):** mantida, pois são valores reais.
""")
 
    with tab2:
        st.write(f"**Integração:** {len(df_final)} filmes casaram entre IMDb e YouTube (via Título e Ano).")
        st.markdown("A ligação entre as bases é feita pelo **título (em minúsculas) + ano**. Só o título não basta, porque existem filmes com o mesmo nome em anos diferentes.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Filmes no IMDb limpo", len(df_limpo))
        m2.metric("Filmes no YouTube", len(df_yt))
        m3.metric("Casaram (únicos)", len(df_final))
        m4.metric("Não casaram", len(df_nao))
 
        st.markdown("**Amostra da base final**")
        st.dataframe(df_final[["movie_title", "title_year", "movie_facebook_likes", "gross_imdb", "imdb_score", "favorability"]].head(10))
 
        st.markdown("**Estatísticas descritivas da base final (IMDb + YouTube)**")
        cols_final = ["movie_facebook_likes", "director_facebook_likes", "cast_total_facebook_likes",
                      "gross_imdb", "imdb_score", "favorability", "votes", "runtime"]
        st.dataframe(df_final[[c for c in cols_final if c in df_final.columns]].describe().T)
        st.markdown("""
**Principais observações da base final (IMDb + YouTube)**
 
- A base final tem 642 filmes únicos, de 1980 a 2016, concentrados entre 2006 e 2012.
- Esses filmes são mais populares que a média do IMDb 5000: a mediana das curtidas do filme é 17.000 (antes 5.000) e a da bilheteria é ~33,5 milhões (antes ~25 milhões).
- `movie_facebook_likes` tem 329 valores, `director_facebook_likes` 534 e `budget` 476.
- Curtidas e bilheteria continuam muito assimétricas; `imdb_score` e `favorability` têm formato de sino.
- A correlação entre curtidas do filme e bilheteria caiu de 0,60 (base do IMDb) para 0,36 (base final).
- A favorabilidade dos comentários tem correlação fraca com todas as variáveis (0,05 a 0,27), a maior com a nota do IMDb.
- O número de votos no IMDb é a variável mais ligada às demais (0,64 com curtidas do filme, 0,63 com bilheteria).
- Limitações: base concentrada em uma época, só filmes com trailer comentado no YouTube, e correlação não prova causa.
""")
 
        st.markdown("**Filmes do YouTube que não casaram com o IMDb (primeiros 15)**")
        st.caption(f"{len(df_nao)} não casaram; desses, {n_outro_ano} existem no IMDb com outro ano.")
        st.dataframe(df_nao[["name", "year"]].head(15))
 
        st.markdown("**Dicionário das colunas originais do YouTube (tipo e valores ausentes)**")
        st.dataframe(descrever_base(df_yt_raw))
        st.markdown("""
**Decisões de limpeza (YouTube)**
- **Duplicados:** linhas totalmente repetidas foram removidas.
- **Nome e ano ausentes:** removidos, porque o casamento com o IMDb depende dos dois.
- **Tipos:** o ano virou número inteiro e o nome teve os espaços das pontas removidos.
- **Favorabilidade ausente:** não removemos nesta etapa. Os filmes continuam na base e só deixam de entrar nas análises que usam essa variável.
""")
 
    with tab3:
        df_rev_ok = df_final_rev.dropna(subset=["rev_pct_positivo"])
        st.write(f"**Análise de Sentimento:** {len(df_rev_ok)} filmes foram citados em pelo menos {MIN_MENCOES} resenhas.")
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        sns.regplot(data=df_rev_ok, x="rev_pct_positivo", y="imdb_score",
                    scatter_kws={"alpha": 0.5, "color": "#9467bd"}, line_kws={"color": "#111", "linewidth": 2}, ax=ax3)
        ax3.set_title("% Resenhas Positivas vs Nota IMDb")
        sns.despine()
        mostrar(fig3)
 
        st.markdown("""
**Ligação com os filmes: título dentro do texto.** A base de resenhas não diz de qual filme cada resenha fala. Por isso, uma resenha "cita" um filme quando o título aparece no texto (sem diferenciar maiúsculas, acentos e pontuação, e só como palavra inteira). Para cada filme calculamos `rev_mencoes` (quantas resenhas citam o título) e `rev_pct_positivo` (porcentagem delas que é positiva).
 
Cuidados para reduzir erros:
- Só usamos títulos com 2 ou mais palavras, ou com 10 ou mais letras (títulos como "Up" ou "Fury" aparecem em muitos textos sem falar do filme).
- Títulos que se repetem na base (refilmagens) são descartados, porque não dá para saber de qual ano é a resenha.
- Só analisamos filmes com pelo menos `MIN_MENCOES` resenhas, pois uma porcentagem calculada com 1 ou 2 resenhas não é confiável.
""")
 
        st.markdown("**Amostra das resenhas (depois da limpeza)**")
        st.caption(f"{len(df_rev_raw)} resenhas originais, {len(df_rev)} depois de remover duplicadas e tags HTML.")
        st.dataframe(df_rev[["review", "sentiment", "positivo", "n_palavras"]].head())
 
        st.markdown("**Filmes mais citados nas resenhas (top 10)**")
        st.dataframe(df_mencoes.sort_values("rev_mencoes", ascending=False).head(10))
 
        st.markdown(f"**Filmes com pelo menos {MIN_MENCOES} resenhas (primeiros 10)**")
        st.dataframe(df_rev_ok[["movie_title", "title_year", "rev_mencoes", "rev_pct_positivo", "imdb_score", "favorability"]].head(10))
 
        st.markdown("""
**Como ler:** as resenhas vêm do próprio IMDb, então a % de resenhas positivas *deveria* acompanhar a nota do IMDb. Se acompanha, a ligação por título está funcionando.
 
**Limitações desta base:**
- A base não informa o filme de cada resenha. Citar o título não garante que a resenha seja sobre aquele filme (ela pode comparar com outro, ou falar de uma sequência).
- Ela não aumenta o número de filmes (continuam os mesmos da base final): acrescenta informação de opinião em texto só para os filmes que foram citados.
- As resenhas são do IMDb e a maioria é de filmes até o início dos anos 2010, então filmes mais recentes têm pouca ou nenhuma citação.
""")
 
# --- HIPÓTESES E PLANO DE ANÁLISE ---
st.header("Hipóteses e Plano de Análise")
 
HIPOTESES = [
    # (código, descrição, x (engajamento), y (sucesso), tipo)
    ("H1", "Curtidas do filme -> Bilheteria", "movie_facebook_likes", "gross_imdb", "rede social"),
    ("H2", "Curtidas do elenco -> Bilheteria", "cast_total_facebook_likes", "gross_imdb", "rede social"),
    ("H3", "Curtidas do diretor -> Bilheteria", "director_facebook_likes", "gross_imdb", "rede social"),
    ("H4", "Favorabilidade (YouTube) -> Nota IMDb", "favorability", "imdb_score", "rede social"),
    ("H5", "Favorabilidade (YouTube) -> Bilheteria", "favorability", "gross_imdb", "rede social"),
    ("H6", "% de resenhas positivas -> Nota IMDb (validação)", "rev_pct_positivo", "imdb_score", "validação"),
]
 
st.markdown("""
A pergunta principal é: *o engajamento nas redes sociais pode indicar o sucesso de um filme?* Para respondê-la, formulamos as hipóteses abaixo. Cada uma liga uma medida de engajamento a uma medida de sucesso e espera uma relação **positiva**.
 
| Código | Hipótese | Engajamento (x) | Sucesso (y) |
|---|---|---|---|
| H1 | Filmes com mais curtidas na página do Facebook têm maior bilheteria | `movie_facebook_likes` | `gross_imdb` |
| H2 | Filmes com elenco mais curtido têm maior bilheteria | `cast_total_facebook_likes` | `gross_imdb` |
| H3 | Filmes com diretor mais curtido têm maior bilheteria | `director_facebook_likes` | `gross_imdb` |
| H4 | Trailers com comentários mais positivos no YouTube têm maior nota no IMDb | `favorability` | `imdb_score` |
| H5 | Trailers com comentários mais positivos no YouTube têm maior bilheteria | `favorability` | `gross_imdb` |
| H6 | *(validação)* Filmes com mais resenhas positivas na base de 50K têm maior nota no IMDb | `rev_pct_positivo` | `imdb_score` |
 
**Plano de análise**
1. Visualizar as relações (gráficos abaixo): dispersão, faixas de engajamento e sentimento do trailer.
2. Testar cada hipótese com correlação de **Spearman** e p-valor (tabela de resultados ao final). Escolhemos Spearman porque curtidas e bilheteria são muito assimétricas, com poucos filmes concentrando valores enormes.
3. Considerar uma hipótese **apoiada pelos dados** quando a correlação é positiva e o p-valor é menor que 0,05. Também informamos a força da relação (desprezível, fraca, moderada ou forte), porque com muitos filmes até uma relação minúscula pode ser significativa.
4. Comparar a bilheteria entre faixas de engajamento (baixo, médio, alto) com o teste de Kruskal-Wallis.
5. Checar se o **orçamento** explica a relação: filmes caros têm mais curtidas e mais bilheteria, então calculamos a correlação parcial controlando o orçamento.
""")
 
st.markdown("---")
 
# --- VISUALIZAÇÕES PRINCIPAIS (TEXTO NA ESQUERDA, GRÁFICO NA DIREITA) ---
st.header("Análise Visual das Hipóteses")
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
    mostrar(fig4)
 
st.markdown("---")
 
# 2. Distribuição por Faixas
d_faixa = df_final.dropna(subset=["movie_facebook_likes", "gross_imdb"]).copy()
d_faixa["Engajamento"] = pd.qcut(d_faixa["movie_facebook_likes"], q=3, labels=FAIXAS)
medianas_faixa = d_faixa.groupby("Engajamento", observed=True)["gross_imdb"].median()
 
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
    st.markdown("**Mediana da bilheteria por faixa (USD)**")
    st.dataframe(medianas_faixa.round(0).rename("Mediana da bilheteria"))
    st.caption("A mediana " + ("aumenta" if medianas_faixa.is_monotonic_increasing else "não aumenta de forma contínua")
               + " da faixa de baixo para a de alto engajamento.")
with col_fig2:
    fig5, ax5 = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=d_faixa, x="Engajamento", y="gross_imdb", hue="Engajamento",
                palette="Blues", legend=False, showmeans=True, ax=ax5,
                meanprops={"marker": "o", "markerfacecolor": "#e74c3c", "markeredgecolor": "white", "markersize": 8})
    ax5.set_yscale("log")
    ax5.set_title("Distribuição da Bilheteria por Faixa de Engajamento", fontweight="bold", pad=15)
    ax5.set_xlabel("Nível de Engajamento")
    ax5.set_ylabel("Bilheteria em USD (Escala Log)")
    sns.despine()
    mostrar(fig5)
 
st.markdown("---")
 
# 3. Sentimento YouTube
r_h5 = correlacao_spearman(df_final, "favorability", "gross_imdb")
txt_h5 = ("sem dados suficientes" if np.isnan(r_h5["rho"])
          else f"correlação {classificar_forca(r_h5['rho'])} (ρ = {r_h5['rho']:.2f})")
 
col_txt3, col_fig3 = st.columns([1, 2])
with col_txt3:
    st.subheader("Comentários do Trailer (YouTube)")
    st.markdown(f"""
    **Como ler:** 
    O eixo horizontal representa a Taxa de Favorabilidade (percentual de comentários positivos no YouTube).
    
    **Leitura dos dados:**
    Diferente das curtidas, o sentimento dos comentários formam uma nuvem muito larga e quase sem direção. 
    
    A favorabilidade tem {txt_h5} com a bilheteria. Ela reflete um pouco melhor a **Nota do IMDb** (qualidade percebida pelo público), mas definitivamente não serve como bússola comercial.
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
    mostrar(fig6)
 
# --- RESULTADOS DOS TESTES E CONCLUSÃO ---
st.header("Conclusão e Teste de Hipóteses")
st.markdown("""
Rodamos correlações de **Spearman** para validar estatisticamente os gráficos acima. Uma hipótese é **apoiada pelos dados** quando a correlação é positiva e `p < 0,05`.
""")
 
df_res = testar_hipoteses(df_final_rev, HIPOTESES)
 
# Tabela completa de resultados (n, rho, p-valor, força e veredito)
tabela = df_res[["Hipótese", "Descrição", "n", "rho", "p-valor", "Força", "Veredito"]].copy()
tabela["rho"] = tabela["rho"].round(2)
tabela["p-valor"] = tabela["p-valor"].map(lambda p: f"{p:.3g}")
st.dataframe(tabela, hide_index=True)
st.download_button("Baixar resultados (CSV)", df_res.to_csv(index=False).encode("utf-8"),
                   "resultados_hipoteses.csv", "text/csv")
 
col_txt4, col_fig4 = st.columns([1.2, 1.5])
 
with col_txt4:
    st.success("""
    **Veredito Final:** O engajamento nas redes sociais **é um indicador parcial do sucesso**. 
    
    Curtidas da página do filme funcionam bem para prever bilheteria. Já o sentimento de comentários de trailer ajuda muito pouco (serve mais como termômetro de nota do que de dinheiro).
    
    *(Lembrete: Filmes maiores possuem orçamentos enormes de marketing que impulsionam simultaneamente os likes e a bilheteria).*
    """)
 
with col_fig4:
    d_res = df_res.dropna(subset=["rho"]).copy()
 
    fig7, ax7 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=d_res, y="Hipótese", x="rho", color="#3498db", ax=ax7)
 
    ax7.axvline(0, color="black", linewidth=1)
    for i, v in enumerate(d_res["rho"]):
        ax7.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:.2f}", va="center",
                 ha="left" if v >= 0 else "right", fontweight="bold", color="#333")
 
    ax7.set_title("Força da Correlação por Hipótese (Spearman)", fontweight="bold", pad=15)
    ax7.set_xlabel("Força (rho)")
    ax7.set_ylabel("")
    sns.despine(left=True, bottom=True)
    ax7.xaxis.set_visible(False)  # Remove o eixo X que fica redundante com os números na barra
    mostrar(fig7)
 
# --- TESTES COMPLEMENTARES ---
with st.expander("Testes complementares: faixas de engajamento e efeito do orçamento"):
    st.markdown("**Kruskal-Wallis: a bilheteria muda entre as faixas de curtidas do filme?**")
    medianas, h, p_kw = teste_faixas(df_final)
    st.dataframe(medianas.round(0).rename("Mediana da bilheteria (USD)"))
    st.info(f"H = {h:.2f}, p = {p_kw:.3g} → " +
            ("há diferença significativa entre as faixas." if p_kw < ALFA else "não há diferença significativa entre as faixas."))
 
    st.markdown("**O orçamento explica a relação?**")
    st.markdown("Filmes caros costumam ter mais divulgação (mais curtidas) e mais bilheteria ao mesmo tempo. Se a correlação cai bastante depois de controlar o orçamento, o engajamento explica pouco além do dinheiro investido. Só temos orçamento dos filmes dos EUA, por isso o *n* é menor.")
    linhas_parcial = []
    for x, nome in [("movie_facebook_likes", "H1 curtidas do filme"), ("cast_total_facebook_likes", "H2 curtidas do elenco")]:
        r = correlacao_parcial_spearman(df_final, x, "gross_imdb", "budget_imdb")
        if not np.isnan(r["parcial"]):
            linhas_parcial.append({"Relação": f"{nome} x bilheteria", "n": r["n"],
                                   "rho (sem controle)": round(r["simples"], 2),
                                   "rho (controlando orçamento)": round(r["parcial"], 2),
                                   "p-valor": f"{r['p']:.3g}"})
    if linhas_parcial:
        st.dataframe(pd.DataFrame(linhas_parcial), hide_index=True)
    else:
        st.write("Poucos filmes com orçamento para calcular.")
 
