"""
app.py — Aplicação web de Análise de Componentes Principais (PCA) em Streamlit.
Executar localmente:  python -m streamlit run app.py
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_wine

import pca_core as pc
import relatorio as rel

st.set_page_config(page_title="Análise PCA", page_icon="🧭", layout="wide")

PALETA = px.colors.qualitative.Safe
NOMES_MANUAIS_PADRAO = "Var1, Var2, Var3, Var4"


# ----------------------------------------------------------------------------
# Carregamento de dados
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def carregar_exemplo() -> pd.DataFrame:
    dados = load_wine(as_frame=True)
    df = dados.frame.copy()
    df["target"] = df["target"].map({0: "Cultivar A", 1: "Cultivar B", 2: "Cultivar C"})
    return df.rename(columns={"target": "Cultivar"})


def tabela_manual_inicial(nomes: list[str], n_linhas: int) -> pd.DataFrame:
    """Tabela de partida para inserção manual (com valores de exemplo se usar os nomes padrão)."""
    if nomes == [n.strip() for n in NOMES_MANUAIS_PADRAO.split(",")]:
        base = pd.DataFrame({
            "Var1": [10.0, 12.5, 9.8, 14.2, 11.0, 13.1, 8.9, 12.0],
            "Var2": [100.0, 110.0, 95.0, 130.0, 105.0, 118.0, 90.0, 112.0],
            "Var3": [3.2, 3.8, 2.9, 4.1, 3.5, 3.6, 3.0, 3.9],
            "Var4": [50.0, 60.0, 48.0, 72.0, 54.0, 63.0, 45.0, 58.0],
        })
        if n_linhas > len(base):
            base = pd.concat([base, pd.DataFrame(np.nan, index=range(n_linhas - len(base)),
                                                 columns=base.columns)], ignore_index=True)
        base = base.head(n_linhas)
    else:
        base = pd.DataFrame(np.nan, index=range(n_linhas), columns=nomes, dtype=float)
    base.insert(0, "ID", [f"Obs{i + 1}" for i in range(len(base))])
    return base


@st.cache_data(show_spinner=False)
def ler_ficheiro(conteudo: bytes, nome: str, decimal: str, folha: str | int = 0) -> pd.DataFrame:
    if nome.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(conteudo), sheet_name=folha)
    return pd.read_csv(io.BytesIO(conteudo), sep=None, engine="python", decimal=decimal)


@st.cache_data(show_spinner=False)
def calcular_paralela(n: int, p: int) -> np.ndarray:
    return pc.analise_paralela(n, p)


def exportar_excel(res: dict, importancia: pd.DataFrame, interpretacao: pd.DataFrame,
                   X: pd.DataFrame, k: int, kmo_var: pd.Series) -> bytes:
    buffer = io.BytesIO()
    pcs = res["variancia"].index[:k]
    with pd.ExcelWriter(buffer, engine="openpyxl") as w:
        res["variancia"].to_excel(w, sheet_name="Variancia")
        importancia.to_excel(w, sheet_name="Importancia_variaveis")
        interpretacao.to_excel(w, sheet_name="Interpretacao_PCs")
        res["loadings"][pcs].to_excel(w, sheet_name="Loadings")
        res["contrib"][pcs].to_excel(w, sheet_name="Contribuicoes_pct")
        res["cos2"][pcs].to_excel(w, sheet_name="Cos2")
        res["vetores"][pcs].to_excel(w, sheet_name="Vetores_proprios")
        res["scores"][pcs].to_excel(w, sheet_name="Scores")
        kmo_var.to_frame().to_excel(w, sheet_name="KMO_por_variavel")
        X.describe().T.to_excel(w, sheet_name="Descritivas")
    return buffer.getvalue()


def sugerir_grupo(df: pd.DataFrame, candidatas: list[str]) -> str | None:
    """Primeira coluna com poucas categorias (boa para colorir)."""
    for c in candidatas:
        if 1 < df[c].nunique() <= min(15, len(df) // 2):
            return c
    return None


def sugerir_id(df: pd.DataFrame, candidatas: list[str]) -> str | None:
    for c in candidatas:
        if df[c].nunique() == len(df):
            return c
    return None


# ----------------------------------------------------------------------------
# Barra lateral — origem dos dados
# ----------------------------------------------------------------------------
st.sidebar.header("Dados")
origem = st.sidebar.radio(
    "Origem dos dados",
    ["Carregar ficheiro", "Inserir manualmente", "Usar exemplo (vinhos)"],
)

st.title("Análise de Componentes Principais")
st.caption(
    "Reduz muitas variáveis correlacionadas a poucas componentes, mostra quais as variáveis que "
    "mais pesam e gera um relatório que explica os resultados e as decisões."
)

df = None
nome_dados = ""
if origem == "Carregar ficheiro":
    ficheiro = st.sidebar.file_uploader("Ficheiro CSV ou Excel", type=["csv", "xlsx", "xls"])
    decimal = st.sidebar.selectbox("Separador decimal (CSV)", [".", ","], index=0)
    if ficheiro is not None:
        folha = 0
        if ficheiro.name.lower().endswith((".xlsx", ".xls")):
            folhas = pd.ExcelFile(io.BytesIO(ficheiro.getvalue())).sheet_names
            folha = st.sidebar.selectbox("Folha", folhas)
        try:
            df = ler_ficheiro(ficheiro.getvalue(), ficheiro.name, decimal, folha)
            nome_dados = ficheiro.name
        except Exception as erro:  # noqa: BLE001
            st.sidebar.error(f"Não foi possível ler o ficheiro: {erro}")
elif origem == "Inserir manualmente":
    nomes_txt = st.sidebar.text_input("Nomes das variáveis (separados por vírgula)", NOMES_MANUAIS_PADRAO)
    n_linhas = int(st.sidebar.number_input("Número inicial de linhas", 3, 1000, 8, 1))
    nomes_vars = list(dict.fromkeys(n.strip() for n in nomes_txt.split(",") if n.strip()))
    st.subheader("Tabela de dados")
    st.caption(
        "Escreva os valores ou cole diretamente do Excel (Ctrl+V). Use a linha vazia no fundo para "
        "acrescentar observações. A coluna ID identifica cada linha nos gráficos."
    )
    base = tabela_manual_inicial(nomes_vars, n_linhas)
    df = st.data_editor(
        base, num_rows="dynamic", width="stretch", hide_index=True,
        key=f"editor_{'|'.join(nomes_vars)}_{n_linhas}",
        column_config={c: st.column_config.NumberColumn(c, format="%.4g") for c in nomes_vars},
    )
    df = df.dropna(how="all", subset=nomes_vars).reset_index(drop=True)
    nome_dados = "Dados inseridos manualmente"
else:
    df = carregar_exemplo()
    nome_dados = "Exemplo: vinhos (UCI Wine)"

if df is None:
    st.info(
        "Carregue um ficheiro CSV ou Excel na barra lateral (uma linha por observação, uma coluna "
        "por variável), insira os dados manualmente, ou escolha o conjunto de exemplo."
    )
    st.stop()

colunas_numericas = df.select_dtypes(include="number").columns.tolist()
colunas_texto = [c for c in df.columns if c not in colunas_numericas]

# ----------------------------------------------------------------------------
# Barra lateral — variáveis e parâmetros
# ----------------------------------------------------------------------------
st.sidebar.header("Variáveis")
variaveis = st.sidebar.multiselect(
    "Variáveis para a PCA (mínimo 2)",
    options=df.columns.tolist(),
    default=colunas_numericas,
    help="Pode escolher quantas quiser. Colunas não numéricas são convertidas quando possível.",
)
restantes = [c for c in df.columns if c not in variaveis]
opcoes_grupo = ["(nenhuma)"] + restantes
g_sug = sugerir_grupo(df, restantes)
grupo = st.sidebar.selectbox("Coluna de grupo (para colorir os gráficos)", opcoes_grupo,
                             index=opcoes_grupo.index(g_sug) if g_sug else 0)
opcoes_id = ["(número da linha)"] + restantes
i_sug = sugerir_id(df, restantes)
rotulo = st.sidebar.selectbox("Coluna de identificação", opcoes_id,
                              index=opcoes_id.index(i_sug) if i_sug else 0)

st.sidebar.header("Parâmetros")
tratamento = st.sidebar.selectbox(
    "Valores em falta",
    ["remover", "media", "mediana"],
    format_func={"remover": "Remover linhas incompletas", "media": "Substituir pela média",
                 "mediana": "Substituir pela mediana"}.get,
)
padronizar = st.sidebar.checkbox(
    "Padronizar variáveis (recomendado)",
    value=True,
    help="Usa a matriz de correlações. Desative só se todas as variáveis estiverem na mesma unidade.",
)
limiar_acum = st.sidebar.slider("Variância acumulada alvo (%)", 50, 99, 80)
limiar_loading = st.sidebar.slider("|Loading| mínimo para interpretar", 0.30, 0.90, 0.50, 0.05)

if len(variaveis) < 2:
    st.warning("Escolha pelo menos 2 variáveis na barra lateral para correr a PCA.")
    st.stop()

X, indice, removidas = pc.preparar_dados(df, variaveis, tratamento)
if removidas:
    st.warning(f"Variáveis removidas por serem constantes ou sem valores numéricos: {', '.join(removidas)}")
if X.shape[1] < 2:
    st.error("Restam menos de 2 variáveis válidas. Escolha outras colunas numéricas ou preencha a tabela.")
    st.stop()
if X.shape[0] <= X.shape[1]:
    st.error(
        f"Há {X.shape[0]} observações completas para {X.shape[1]} variáveis. A PCA precisa de mais "
        "observações do que variáveis (idealmente 5 a 10 vezes mais). Acrescente linhas à tabela."
    )
    st.stop()

grupos = None if grupo == "(nenhuma)" else df.loc[indice, grupo].astype(str)
if rotulo == "(número da linha)":
    nomes = pd.Series([f"Linha {i + 1}" if isinstance(i, (int, np.integer)) else str(i) for i in indice],
                      index=indice)
else:
    nomes = df.loc[indice, rotulo].astype(str)

# ----------------------------------------------------------------------------
# Cálculos
# ----------------------------------------------------------------------------
res = pc.executar_pca(X, padronizar)
kmo_global, kmo_var = pc.teste_kmo(X)
chi2, gl, p_bart = pc.teste_bartlett(X)
paralela = calcular_paralela(X.shape[0], X.shape[1]) if padronizar else None
sugestoes = pc.sugerir_componentes(res["variancia"], limiar_acum, padronizar, paralela)

n_max = len(res["variancia"])
k_sugerido = min(max(2, list(sugestoes.values())[0]), n_max)

st.sidebar.header("Componentes a reter")
k = int(st.sidebar.number_input("Número de componentes (k)", 1, n_max, k_sugerido, 1))
pcs = list(res["variancia"].index[:k])
todas = list(res["variancia"].index)

importancia = pc.importancia_variaveis(res, k)
interpretacao = pc.variaveis_por_componente(res, k, limiar_loading)

contexto = dict(
    res=res, X=X, k=k, kmo_global=kmo_global, kmo_var=kmo_var, chi2=chi2, gl=gl, p_bart=p_bart,
    sugestoes=sugestoes, importancia=importancia, interpretacao=interpretacao, padronizar=padronizar,
    tratamento=tratamento, n_original=len(df), removidas=removidas, grupos=grupos, nomes=nomes,
    limiar_acum=limiar_acum, limiar_loading=limiar_loading, paralela=paralela,
)

# Resumo
c1, c2, c3, c4 = st.columns(4)
c1.metric("Observações usadas", f"{X.shape[0]}",
          f"{len(df) - X.shape[0]} excluídas" if len(df) != X.shape[0] else None, delta_color="off")
c2.metric("Variáveis", X.shape[1])
c3.metric(f"Variância explicada por {k} PC", f"{res['variancia']['% Acumulada'].iloc[k - 1]:.1f}%")
c4.metric("KMO", f"{kmo_global:.3f}", pc.interpretar_kmo(kmo_global), delta_color="off")

abas = st.tabs([
    "Dados", "Adequação", "Variância explicada", "Variáveis relevantes",
    "Círculo de correlações", "Observações", "Passo a passo", "Relatório e exportação",
])

# ----------------------------------------------------------------------------
# 1. Dados
# ----------------------------------------------------------------------------
with abas[0]:
    st.subheader("Pré-visualização")
    st.dataframe(df.head(50), width="stretch")
    st.subheader("Estatística descritiva das variáveis usadas")
    st.dataframe(X.describe().T.style.format("{:.3f}"), width="stretch")
    st.subheader("Matriz de correlações")
    corr = X.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=max(400, 38 * len(corr)))
    st.plotly_chart(fig, width="stretch")
    st.caption("A PCA é útil quando há correlações fortes (|r| ≥ 0,3) entre várias variáveis.")

# ----------------------------------------------------------------------------
# 2. Adequação
# ----------------------------------------------------------------------------
with abas[1]:
    a1, a2 = st.columns(2)
    with a1:
        st.subheader("KMO (Kaiser-Meyer-Olkin)")
        st.metric("KMO global", f"{kmo_global:.3f}", pc.interpretar_kmo(kmo_global), delta_color="off")
        st.write(
            "Mede se as correlações entre variáveis são explicadas por fatores comuns. "
            "Valores ≥ 0,7 são desejáveis; abaixo de 0,5 a PCA não é recomendada."
        )
    with a2:
        st.subheader("Teste de esfericidade de Bartlett")
        st.metric("χ²", f"{chi2:,.2f}")
        st.write(f"Graus de liberdade: **{gl}** · p-valor: **{p_bart:.4g}**")
        if p_bart < 0.05:
            st.success("p < 0,05: as variáveis estão correlacionadas; a PCA faz sentido.")
        else:
            st.error("p ≥ 0,05: não se rejeita que as variáveis sejam independentes; a PCA tem pouco valor.")

    st.subheader("MSA por variável")
    msa = kmo_var.to_frame().assign(Classificação=kmo_var.map(pc.interpretar_kmo)).sort_values("MSA")
    st.dataframe(msa.style.format({"MSA": "{:.3f}"}), width="stretch")
    fracas = msa.index[msa["MSA"] < 0.5].tolist()
    if fracas:
        st.warning(f"Considere retirar variáveis com MSA < 0,5: {', '.join(fracas)}")

# ----------------------------------------------------------------------------
# 3. Variância explicada
# ----------------------------------------------------------------------------
with abas[2]:
    tv = res["variancia"]
    fig = go.Figure()
    fig.add_bar(x=tv.index, y=tv["Valor próprio"], name="Valor próprio",
                marker_color=["#1F6F78" if i < k else "#B8CFD1" for i in range(len(tv))])
    if paralela is not None:
        fig.add_scatter(x=tv.index, y=paralela[:len(tv)], name="Análise paralela (P95)",
                        mode="lines+markers", line=dict(dash="dot", color="#B5651D"))
    if padronizar:
        fig.add_hline(y=1, line_dash="dash", line_color="grey", annotation_text="Kaiser = 1")
    fig.add_scatter(x=tv.index, y=tv["% Acumulada"], name="% acumulada", yaxis="y2",
                    mode="lines+markers", line=dict(color="#6B4C9A"))
    fig.update_layout(
        title="Scree plot (componentes retidas a cor forte)",
        yaxis=dict(title="Valor próprio"),
        yaxis2=dict(title="% acumulada", overlaying="y", side="right", range=[0, 105]),
        legend=dict(orientation="h", y=-0.2),
        height=480,
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Critérios de retenção")
    st.dataframe(pd.Series(sugestoes, name="Nº de componentes").to_frame(), width="stretch")
    st.caption("O valor de k na barra lateral parte do primeiro critério; ajuste-o conforme a interpretação.")

    st.subheader("Tabela de variância")
    st.dataframe(
        tv.style.format("{:.3f}").apply(
            lambda s: ["font-weight: bold" if i < k else "" for i in range(len(s))], axis=0
        ),
        width="stretch",
    )

# ----------------------------------------------------------------------------
# 4. Variáveis relevantes
# ----------------------------------------------------------------------------
with abas[3]:
    st.subheader(f"Ranking das variáveis nas {k} componentes retidas")
    media = 100 / X.shape[1]
    ranking = importancia.reset_index()
    fig = px.bar(
        ranking.sort_values("Contribuição ponderada (%)"),
        x="Contribuição ponderada (%)", y="Variável", orientation="h",
        color="PC dominante", color_discrete_sequence=PALETA,
        height=max(380, 30 * len(ranking)),
    )
    fig.add_vline(x=media, line_dash="dash", line_color="grey",
                  annotation_text=f"Média esperada ({media:.1f}%)")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Contribuição ponderada: peso de cada variável nas componentes retidas, ponderado pela "
        "variância de cada componente. Variáveis acima da linha tracejada são as mais relevantes."
    )
    st.dataframe(
        importancia.style.format({
            "Contribuição ponderada (%)": "{:.2f}",
            "Comunalidade (Σ cos²)": "{:.3f}",
            "Loading na PC dominante": "{:+.3f}",
        }).background_gradient(subset=["Contribuição ponderada (%)"], cmap="Greens"),
        width="stretch",
    )

    st.subheader("Interpretação das componentes")
    st.dataframe(interpretacao.style.format({"% Variância": "{:.2f}"}), width="stretch")

    b1, b2 = st.columns(2)
    with b1:
        st.subheader("Loadings (correlação variável–componente)")
        fig = px.imshow(res["loadings"][pcs], text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto", height=max(400, 32 * X.shape[1]))
        st.plotly_chart(fig, width="stretch")
    with b2:
        st.subheader("Contribuição de cada variável (%)")
        fig = px.imshow(res["contrib"][pcs], text_auto=".1f", color_continuous_scale="Greens",
                        aspect="auto", height=max(400, 32 * X.shape[1]))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Contribuições numa componente")
    pc_escolhida = st.selectbox("Componente", pcs, key="pc_contrib")
    serie = res["contrib"][pc_escolhida].sort_values(ascending=False).reset_index()
    serie.columns = ["Variável", "Contribuição (%)"]
    fig = px.bar(serie, x="Variável", y="Contribuição (%)", color_discrete_sequence=["#1F6F78"])
    fig.add_hline(y=media, line_dash="dash", line_color="grey")
    st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------
# 5. Círculo de correlações
# ----------------------------------------------------------------------------
with abas[4]:
    if n_max < 2:
        st.info("São necessárias pelo menos 2 componentes.")
    else:
        d1, d2 = st.columns(2)
        eixo_x = d1.selectbox("Eixo horizontal", todas, index=0, key="cx")
        eixo_y = d2.selectbox("Eixo vertical", todas, index=1, key="cy")

        L = res["loadings"]
        cos2_plano = res["cos2"][eixo_x] + res["cos2"][eixo_y]
        fig = go.Figure()
        t = np.linspace(0, 2 * np.pi, 200)
        fig.add_scatter(x=np.cos(t), y=np.sin(t), mode="lines", line=dict(color="grey"), showlegend=False)
        fig.add_scatter(x=0.5 * np.cos(t), y=0.5 * np.sin(t), mode="lines",
                        line=dict(color="lightgrey", dash="dot"), showlegend=False)
        for var in L.index:
            x, y = L.loc[var, eixo_x], L.loc[var, eixo_y]
            cor = px.colors.sample_colorscale("Tealgrn", float(np.clip(cos2_plano[var], 0, 1)))[0]
            fig.add_annotation(x=x, y=y, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowwidth=1.6, arrowcolor=cor)
            fig.add_annotation(x=x * 1.1, y=y * 1.1, text=var, showarrow=False, font=dict(size=11))
        fig.update_layout(
            xaxis=dict(range=[-1.2, 1.2], title=f"{eixo_x} ({res['variancia'].loc[eixo_x, '% Variância']:.1f}%)",
                       zeroline=True),
            yaxis=dict(range=[-1.2, 1.2], title=f"{eixo_y} ({res['variancia'].loc[eixo_y, '% Variância']:.1f}%)",
                       zeroline=True, scaleanchor="x", scaleratio=1),
            height=650, title="Círculo de correlações",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Setas longas (perto do círculo) = variável bem representada neste plano. Setas na mesma "
            "direção = variáveis positivamente correlacionadas; opostas = correlação negativa; "
            "perpendiculares = pouco correlacionadas. Cor mais escura = maior cos² no plano."
        )
        st.dataframe(cos2_plano.sort_values(ascending=False).to_frame(f"cos² em {eixo_x}+{eixo_y}")
                     .style.format("{:.3f}"), width="stretch")

# ----------------------------------------------------------------------------
# 6. Observações (scores e biplot)
# ----------------------------------------------------------------------------
with abas[5]:
    S = res["scores"].copy()
    S["Identificação"] = nomes.values
    if grupos is not None:
        S["Grupo"] = grupos.values

    tipo = st.radio("Tipo de gráfico", ["2D", "Biplot 2D", "3D"], horizontal=True)

    if tipo in ("2D", "Biplot 2D"):
        e1, e2 = st.columns(2)
        sx = e1.selectbox("Eixo horizontal", todas, index=0, key="sx")
        sy = e2.selectbox("Eixo vertical", todas, index=min(1, n_max - 1), key="sy")
        fig = px.scatter(
            S, x=sx, y=sy, color="Grupo" if grupos is not None else None,
            hover_name="Identificação", color_discrete_sequence=PALETA, height=620,
        )
        if tipo == "Biplot 2D":
            escala = np.abs(S[[sx, sy]]).max().max() / np.abs(res["loadings"][[sx, sy]]).max().max() * 0.8
            for var in res["loadings"].index:
                x = res["loadings"].loc[var, sx] * escala
                y = res["loadings"].loc[var, sy] * escala
                fig.add_annotation(x=x, y=y, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                                   showarrow=True, arrowhead=2, arrowcolor="#B5651D", arrowwidth=1.5)
                fig.add_annotation(x=x * 1.08, y=y * 1.08, text=var, showarrow=False,
                                   font=dict(color="#B5651D", size=11))
        fig.add_hline(y=0, line_color="lightgrey")
        fig.add_vline(x=0, line_color="lightgrey")
        fig.update_layout(
            xaxis_title=f"{sx} ({res['variancia'].loc[sx, '% Variância']:.1f}%)",
            yaxis_title=f"{sy} ({res['variancia'].loc[sy, '% Variância']:.1f}%)",
        )
        st.plotly_chart(fig, width="stretch")
    else:
        if n_max < 3:
            st.info("O gráfico 3D precisa de pelo menos 3 componentes (3 variáveis).")
        else:
            e1, e2, e3 = st.columns(3)
            sx = e1.selectbox("Eixo X", todas, index=0, key="tx")
            sy = e2.selectbox("Eixo Y", todas, index=1, key="ty")
            sz = e3.selectbox("Eixo Z", todas, index=2, key="tz")
            fig = px.scatter_3d(
                S, x=sx, y=sy, z=sz, color="Grupo" if grupos is not None else None,
                hover_name="Identificação", color_discrete_sequence=PALETA, height=700,
            )
            fig.update_traces(marker=dict(size=4))
            st.plotly_chart(fig, width="stretch")

    st.subheader("Scores das observações")
    st.dataframe(S[["Identificação"] + (["Grupo"] if grupos is not None else []) + pcs]
                 .style.format({p: "{:.3f}" for p in pcs}), width="stretch")

# ----------------------------------------------------------------------------
# 7. Passo a passo (a matemática aplicada aos dados)
# ----------------------------------------------------------------------------
with abas[6]:
    tv = res["variancia"]

    st.subheader("Passo 1 — Padronizar as variáveis")
    razao, v_max, v_min = pc.diagnostico_escala(X)
    escala = pd.DataFrame({"Média (μ)": X.mean(), "Desvio-padrão (σ)": X.std(ddof=1),
                           "Variância": X.var(ddof=1)})
    st.dataframe(escala.style.format("{:.4g}"), width="stretch")
    st.markdown(
        "A PCA procura as direções de maior variância. Sem padronizar, uma variável medida em "
        "unidades grandes (ex.: euros) domina uma medida em unidades pequenas (ex.: anos). "
        "A padronização z = (x − μ) / σ põe todas com média 0 e variância 1, e a PCA passa a usar a "
        "**matriz de correlações** em vez da **matriz de covariâncias**."
    )
    if razao > 10:
        (st.info if padronizar else st.warning)(
            f"Nos seus dados, a variância de **{v_max}** é {razao:,.0f} vezes a de **{v_min}**. "
            + ("A padronização está ativa, por isso este desequilíbrio não afeta os resultados."
               if padronizar else "Sem padronização, as primeiras componentes vão refletir sobretudo "
               f"{v_max}. Ative a padronização na barra lateral.")
        )

    st.subheader("Passo 2 — Valores próprios (eigenvalues)")
    soma = res["variancia_total"]
    st.markdown(
        f"Cada valor próprio λ mede a variância captada por uma componente. A variância total é a "
        f"soma de todos: **Σλ = {soma:.3f}**"
        + (f" (igual ao número de variáveis, {X.shape[1]}, porque os dados estão padronizados)."
           if padronizar else ".")
        + " A percentagem explicada por cada componente é λᵢ / Σλ × 100%."
    )
    calc = pd.DataFrame({
        "Valor próprio (λ)": tv["Valor próprio"],
        "Cálculo": [f"{l:.3f} / {soma:.3f}" for l in tv["Valor próprio"]],
        "% Variância": tv["% Variância"],
        "% Acumulada": tv["% Acumulada"],
    })
    st.dataframe(calc.style.format({"Valor próprio (λ)": "{:.3f}", "% Variância": "{:.2f}%",
                                    "% Acumulada": "{:.2f}%"}), width="stretch")

    st.subheader("Passo 3 — Vetores próprios (eigenvectors) e equações das componentes")
    st.markdown(
        "Cada vetor próprio define a direção de uma componente. Os seus elementos são os "
        "**coeficientes** da combinação linear das variáveis padronizadas. Cada vetor tem "
        "comprimento 1 (a soma dos quadrados dos coeficientes é 1) e é perpendicular aos outros."
    )
    vet = res["vetores"][pcs].copy()
    normas = (vet ** 2).sum()
    st.dataframe(vet.style.format("{:+.4f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
                 width="stretch")
    st.caption("Soma dos quadrados por coluna: " + ", ".join(f"{c} = {v:.4f}" for c, v in normas.items()))
    for comp in pcs:
        st.code(pc.equacao_componente(res, comp), language=None, wrap_lines=True)

    st.subheader("Passo 4 — Projetar uma observação")
    p1, p2 = st.columns(2)
    escolha = p1.selectbox("Observação", list(range(len(X))), format_func=lambda i: nomes.iloc[i])
    comp_proj = p2.selectbox("Componente", pcs, key="comp_proj")
    idx = X.index[escolha]
    passo = pc.projecao_passo_a_passo(X, res, idx, comp_proj)
    if not padronizar:
        passo = passo.rename(columns={"Valor transformado (z)": "Valor centrado (x − μ)",
                                      "Produto z × coef.": "Produto (x − μ) × coef."})
    st.dataframe(passo.style.format("{:.4f}"), width="stretch")
    total = passo.iloc[:, -1].sum()
    st.success(
        f"Score de **{nomes.iloc[escolha]}** em **{comp_proj}** = soma da última coluna = **{total:.4f}** "
        f"(valor calculado pela PCA: {res['scores'].loc[idx, comp_proj]:.4f})."
    )
    st.markdown(
        f"Com as {k} componentes retidas, esta observação passa de {X.shape[1]} valores "
        f"para {k}: **[" + ", ".join(f"{res['scores'].loc[idx, c]:.3f}" for c in pcs) + "]**."
    )

    st.subheader("Passo 5 — Coeficientes vs loadings")
    st.markdown(
        "Há dois conceitos que muitas vezes se chamam ambos *loadings*. Os **coeficientes do vetor "
        "próprio** (passo 3) dizem como se calcula a componente. Os **loadings** usados nos "
        "separadores de interpretação são a **correlação** entre cada variável e a componente: "
        "loading = coeficiente × √λ. Os loadings são mais fáceis de ler porque variam entre −1 e 1."
    )
    comp_cmp = pcs[0]
    cmp = pd.DataFrame({
        "Coeficiente": res["vetores"][comp_cmp],
        "√λ": np.sqrt(tv.loc[comp_cmp, "Valor próprio"]),
        "Coeficiente × √λ": res["vetores"][comp_cmp] * np.sqrt(tv.loc[comp_cmp, "Valor próprio"]),
        "Loading (correlação)": res["loadings"][comp_cmp],
    })
    st.dataframe(cmp.style.format("{:+.4f}"), width="stretch")
    if not padronizar:
        st.caption("Sem padronização, coeficiente × √λ só coincide com a correlação depois de dividir "
                   "pelo desvio-padrão de cada variável.")

# ----------------------------------------------------------------------------
# 8. Relatório e exportação
# ----------------------------------------------------------------------------
with abas[7]:
    secoes = rel.construir_relatorio(contexto)
    for s in secoes:
        st.subheader(s["titulo"])
        for par in s["paragrafos"]:
            st.markdown(par.replace("<sub>i</sub>", "ᵢ"))
        if s.get("tabela") is not None:
            st.dataframe(s["tabela"], width="stretch")

    st.divider()
    st.subheader("Descarregar")
    st.caption(f"Todos os ficheiros refletem as {k} componentes retidas e os parâmetros atuais.")
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        if st.button("Preparar relatório PDF", type="primary"):
            with st.spinner("A gerar o PDF com gráficos…"):
                st.session_state["pdf_relatorio"] = rel.gerar_pdf(secoes, rel.graficos(contexto), nome_dados)
        if "pdf_relatorio" in st.session_state:
            st.download_button("Descarregar relatório (PDF)", st.session_state["pdf_relatorio"],
                               file_name="relatorio_pca.pdf", mime="application/pdf")
    with e2:
        st.download_button("Descarregar relatório (Markdown)", rel.relatorio_markdown(secoes).encode("utf-8"),
                           file_name="relatorio_pca.md", mime="text/markdown")
    with e3:
        st.download_button(
            "Descarregar resultados (Excel)",
            data=exportar_excel(res, importancia, interpretacao, X, k, kmo_var),
            file_name="resultados_pca.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with e4:
        scores_csv = res["scores"][pcs].assign(Identificação=nomes.values)
        st.download_button(
            "Descarregar scores (CSV)",
            data=scores_csv.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
            file_name="scores_pca.csv", mime="text/csv",
        )
    st.caption("Se alterar parâmetros depois de preparar o PDF, carregue de novo em «Preparar relatório PDF».")
