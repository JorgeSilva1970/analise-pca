"""
pca_core.py — Lógica da Análise de Componentes Principais (PCA).
Independente do Streamlit, para poder ser testada e reutilizada (ex.: Colab).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------------
# Preparação dos dados
# ----------------------------------------------------------------------------
def preparar_dados(df: pd.DataFrame, colunas: list[str], tratamento_na: str = "remover"):
    """Converte para numérico, trata valores em falta e remove colunas constantes.

    Devolve (X, indice_linhas_usadas, colunas_removidas).
    """
    X = df[colunas].apply(pd.to_numeric, errors="coerce")

    if tratamento_na == "remover":
        X = X.dropna(axis=0, how="any")
    elif tratamento_na == "media":
        X = X.fillna(X.mean(numeric_only=True))
    elif tratamento_na == "mediana":
        X = X.fillna(X.median(numeric_only=True))

    constantes = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=constantes)
    return X, X.index, constantes


# ----------------------------------------------------------------------------
# Testes de adequação (KMO e Bartlett)
# ----------------------------------------------------------------------------
def teste_kmo(X: pd.DataFrame):
    """Kaiser-Meyer-Olkin global e por variável (MSA)."""
    corr = np.corrcoef(X.values, rowvar=False)
    inv = np.linalg.pinv(corr)
    d = np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
    parcial = -inv / d

    np.fill_diagonal(corr, 0.0)
    np.fill_diagonal(parcial, 0.0)
    r2, p2 = corr ** 2, parcial ** 2

    kmo_global = r2.sum() / (r2.sum() + p2.sum())
    kmo_var = r2.sum(axis=0) / (r2.sum(axis=0) + p2.sum(axis=0))
    return float(kmo_global), pd.Series(kmo_var, index=X.columns, name="MSA")


def teste_bartlett(X: pd.DataFrame):
    """Teste de esfericidade de Bartlett: H0 = matriz de correlações é identidade."""
    n, p = X.shape
    corr = np.corrcoef(X.values, rowvar=False)
    det = np.linalg.det(corr)
    det = max(det, 1e-300)  # evita log(0) com variáveis quase colineares
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(det)
    gl = p * (p - 1) / 2
    p_valor = stats.chi2.sf(chi2, gl)
    return float(chi2), int(gl), float(p_valor)


def interpretar_kmo(valor: float) -> str:
    if valor >= 0.9:
        return "Excelente"
    if valor >= 0.8:
        return "Bom"
    if valor >= 0.7:
        return "Médio"
    if valor >= 0.6:
        return "Razoável"
    if valor >= 0.5:
        return "Mau, mas aceitável"
    return "Inaceitável"


# ----------------------------------------------------------------------------
# PCA
# ----------------------------------------------------------------------------
def executar_pca(X: pd.DataFrame, padronizar: bool = True) -> dict:
    """Corre a PCA com todas as componentes e devolve as tabelas de resultados.

    Com padronização usa z = (x - média) / desvio-padrão amostral, pelo que a matriz
    decomposta é exatamente a matriz de correlações e a soma dos valores próprios
    é igual ao número de variáveis.
    """
    media = X.mean()
    desvio = X.std(ddof=1)
    if padronizar:
        Z = (X - media) / desvio
    else:
        Z = X - media

    n_comp = min(X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(Z.values)

    nomes_pc = [f"PC{i + 1}" for i in range(n_comp)]
    variaveis = list(X.columns)

    valores_proprios = pca.explained_variance_
    var_total = float(np.var(Z.values, axis=0, ddof=1).sum())
    var_expl = valores_proprios / var_total * 100

    tabela_variancia = pd.DataFrame(
        {
            "Valor próprio": valores_proprios,
            "% Variância": var_expl,
            "% Acumulada": np.cumsum(var_expl),
        },
        index=nomes_pc,
    )

    # Vetores próprios (coeficientes das combinações lineares)
    vetores = pd.DataFrame(pca.components_.T, index=variaveis, columns=nomes_pc)

    # Loadings = correlação entre variável original e componente
    Xc = ((X - media) / desvio).values
    Sc = (scores - scores.mean(axis=0)) / scores.std(axis=0, ddof=1)
    loadings = pd.DataFrame(Xc.T @ Sc / (len(X) - 1), index=variaveis, columns=nomes_pc)

    cos2 = loadings ** 2
    contrib = vetores ** 2 * 100  # cada coluna soma 100%

    scores_df = pd.DataFrame(scores, index=X.index, columns=nomes_pc)

    return {
        "modelo": pca,
        "scores": scores_df,
        "variancia": tabela_variancia,
        "variancia_total": var_total,
        "vetores": vetores,
        "loadings": loadings,
        "cos2": cos2,
        "contrib": contrib,
        "padronizado": padronizar,
        "Z": Z,
        "media": media,
        "desvio": desvio,
    }


def equacao_componente(res: dict, pc: str, casas: int = 3) -> str:
    """Escreve a combinação linear de uma componente, ex.: PC1 = 0.600·z(X1) + ..."""
    prefixo = "z" if res["padronizado"] else "c"
    coefs = res["vetores"][pc]
    termos = []
    for i, (var, c) in enumerate(coefs.items()):
        sinal = "-" if c < 0 else ("+" if i else "")
        termos.append(f"{sinal} {abs(c):.{casas}f}·{prefixo}({var})".strip())
    return f"{pc} = " + " ".join(termos)


def projecao_passo_a_passo(X: pd.DataFrame, res: dict, indice, pc: str) -> pd.DataFrame:
    """Mostra o cálculo do score de uma observação numa componente."""
    x = X.loc[indice]
    z = res["Z"].loc[indice]
    coef = res["vetores"][pc]
    tabela = pd.DataFrame(
        {
            "Valor original (x)": x,
            "Média (μ)": res["media"],
            "Desvio-padrão (σ)": res["desvio"],
            "Valor transformado (z)": z,
            f"Coeficiente em {pc}": coef,
            "Produto z × coef.": z * coef,
        }
    )
    tabela.index.name = "Variável"
    return tabela


def diagnostico_escala(X: pd.DataFrame) -> tuple[float, str, str]:
    """Razão entre a maior e a menor variância das variáveis originais."""
    var = X.var(ddof=1)
    var = var[var > 0]
    return float(var.max() / var.min()), var.idxmax(), var.idxmin()


def analise_paralela(n: int, p: int, iteracoes: int = 200, percentil: float = 95, seed: int = 42):
    """Análise paralela de Horn: valores próprios de dados aleatórios (critério de retenção)."""
    rng = np.random.default_rng(seed)
    resultados = np.empty((iteracoes, p))
    for i in range(iteracoes):
        aleatorio = rng.standard_normal((n, p))
        corr = np.corrcoef(aleatorio, rowvar=False)
        resultados[i] = np.sort(np.linalg.eigvalsh(corr))[::-1]
    return np.percentile(resultados, percentil, axis=0)


def sugerir_componentes(tabela_variancia: pd.DataFrame, limiar_acumulado: float,
                        padronizado: bool, limiares_paralela=None) -> dict:
    """Aplica vários critérios de retenção e devolve o nº sugerido por cada um."""
    vp = tabela_variancia["Valor próprio"].values
    acum = tabela_variancia["% Acumulada"].values
    sugestoes = {}

    if padronizado:
        sugestoes["Kaiser (valor próprio > 1)"] = int(max(1, (vp > 1).sum()))

    sugestoes[f"Variância acumulada ≥ {limiar_acumulado:.0f}%"] = int(
        min(len(acum), np.argmax(acum >= limiar_acumulado) + 1)
        if (acum >= limiar_acumulado).any() else len(acum)
    )

    if limiares_paralela is not None and padronizado:
        k = 0
        for real, aleat in zip(vp, limiares_paralela):
            if real > aleat:
                k += 1
            else:
                break
        sugestoes["Análise paralela de Horn"] = int(max(1, k))

    return sugestoes


def importancia_variaveis(res: dict, k: int) -> pd.DataFrame:
    """Ranking das variáveis mais relevantes nas primeiras k componentes.

    - Contribuição ponderada: média das contribuições (%) pesada pela variância de cada PC.
    - Comunalidade: soma dos cos² nas k componentes (quanto da variável é explicado).
    - PC dominante: componente onde a variável tem maior |loading|.
    """
    pcs = res["variancia"].index[:k]
    pesos = res["variancia"].loc[pcs, "% Variância"].values
    contrib_k = res["contrib"][pcs]
    ponderada = (contrib_k.values * pesos).sum(axis=1) / pesos.sum()

    comunalidade = res["cos2"][pcs].sum(axis=1)
    dominante = res["loadings"][pcs].abs().idxmax(axis=1)
    loading_dom = [res["loadings"].loc[v, pc] for v, pc in zip(res["loadings"].index, dominante)]

    tabela = pd.DataFrame(
        {
            "Contribuição ponderada (%)": ponderada,
            "Comunalidade (Σ cos²)": comunalidade.values,
            "PC dominante": dominante.values,
            "Loading na PC dominante": loading_dom,
        },
        index=res["loadings"].index,
    ).sort_values("Contribuição ponderada (%)", ascending=False)

    media_esperada = 100 / len(tabela)
    tabela["Acima da média?"] = np.where(
        tabela["Contribuição ponderada (%)"] > media_esperada, "Sim", "Não"
    )
    tabela.index.name = "Variável"
    return tabela


def variaveis_por_componente(res: dict, k: int, limiar_loading: float = 0.5) -> pd.DataFrame:
    """Lista, para cada PC, as variáveis com |loading| acima do limiar (interpretação)."""
    linhas = []
    for pc in res["variancia"].index[:k]:
        col = res["loadings"][pc]
        fortes = col[col.abs() >= limiar_loading].sort_values(key=np.abs, ascending=False)
        positivas = ", ".join(f"{v} ({l:+.2f})" for v, l in fortes.items() if l > 0) or "—"
        negativas = ", ".join(f"{v} ({l:+.2f})" for v, l in fortes.items() if l < 0) or "—"
        linhas.append(
            {
                "Componente": pc,
                "% Variância": res["variancia"].loc[pc, "% Variância"],
                "Polo positivo": positivas,
                "Polo negativo": negativas,
            }
        )
    return pd.DataFrame(linhas).set_index("Componente")
