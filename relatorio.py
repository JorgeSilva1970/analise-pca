"""
relatorio.py — Relatório automático da PCA: texto explicativo, gráficos e PDF.
O mesmo conteúdo é mostrado na app (Markdown) e exportado para PDF (ReportLab).
"""
from __future__ import annotations

import io
import re
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_JUSTIFY  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

import pca_core as pc  # noqa: E402

COR_PRINCIPAL = "#1F6F78"
COR_SECUNDARIA = "#B5651D"


def _fmt(x: float, casas: int = 2) -> str:
    """Número em formato português (vírgula decimal)."""
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _lista(itens: list[str]) -> str:
    itens = list(itens)
    if not itens:
        return ""
    if len(itens) == 1:
        return itens[0]
    return ", ".join(itens[:-1]) + " e " + itens[-1]


def _rotulo(ctx: dict, indice) -> str:
    nomes = ctx.get("nomes")
    return str(nomes.loc[indice]) if nomes is not None else str(indice)


# ----------------------------------------------------------------------------
# Conteúdo textual
# ----------------------------------------------------------------------------
def construir_relatorio(ctx: dict) -> list[dict]:
    """Gera as secções do relatório.

    ctx deve conter: res, X, k, kmo_global, kmo_var, chi2, gl, p_bart, sugestoes,
    importancia, interpretacao, padronizar, tratamento, n_original, removidas,
    grupos (Series ou None), limiar_acum, limiar_loading.
    """
    res, X, k = ctx["res"], ctx["X"], ctx["k"]
    tv = res["variancia"]
    n, p = X.shape
    pcs = list(tv.index[:k])
    secoes = []

    # 1. Dados e preparação ---------------------------------------------------
    par = [
        f"A análise usou **{n} observações** e **{p} variáveis**: {_lista(list(X.columns))}."
    ]
    excluidas = ctx["n_original"] - n
    if excluidas > 0:
        par.append(f"Foram excluídas {excluidas} linhas com valores em falta.")
    elif ctx["tratamento"] != "remover":
        met = "média" if ctx["tratamento"] == "media" else "mediana"
        par.append(f"Os valores em falta foram substituídos pela {met} de cada variável.")
    if ctx["removidas"]:
        par.append(f"Foram retiradas por serem constantes ou não numéricas: {_lista(ctx['removidas'])}.")

    razao, v_max, v_min = pc.diagnostico_escala(X)
    if ctx["padronizar"]:
        par.append(
            "As variáveis foram **padronizadas** (z = (x − média) / desvio-padrão), pelo que a PCA "
            "foi calculada sobre a **matriz de correlações**. Assim nenhuma variável domina apenas "
            f"por estar numa escala maior: a variância de *{v_max}* é {_fmt(razao, 0)} vezes a de "
            f"*{v_min}*."
            if razao > 10 else
            "As variáveis foram **padronizadas** (z = (x − média) / desvio-padrão), pelo que a PCA "
            "foi calculada sobre a **matriz de correlações** e todas as variáveis partem com o mesmo peso."
        )
    else:
        texto = (
            "As variáveis **não foram padronizadas**; a PCA usou a **matriz de covariâncias**, "
            "o que só é adequado quando todas estão na mesma unidade de medida."
        )
        if razao > 10:
            texto += (
                f" **Atenção:** a variância de *{v_max}* é {_fmt(razao, 0)} vezes a de *{v_min}*, "
                "por isso as primeiras componentes vão refletir sobretudo as variáveis de maior escala. "
                "Recomenda-se ativar a padronização."
            )
        par.append(texto)
    secoes.append({"titulo": "1. Dados e preparação", "paragrafos": par})

    # 2. Adequação ------------------------------------------------------------
    kmo = ctx["kmo_global"]
    par = [
        f"O índice **KMO = {_fmt(kmo, 3)}** ({pc.interpretar_kmo(kmo).lower()}). "
        "Valores a partir de 0,7 indicam que as correlações entre variáveis são suficientemente "
        "fortes para serem resumidas em componentes; abaixo de 0,5 a PCA não é recomendada.",
        f"O **teste de Bartlett** deu qui-quadrado = {_fmt(ctx['chi2'])} com {ctx['gl']} graus de "
        f"liberdade e p-valor = {ctx['p_bart']:.3g}. "
        + ("Como p < 0,05, rejeita-se a hipótese de as variáveis serem independentes: a PCA faz sentido."
           if ctx["p_bart"] < 0.05 else
           "Como p ≥ 0,05, não há evidência de correlação suficiente e os resultados devem ser lidos com cautela."),
    ]
    fracas = ctx["kmo_var"][ctx["kmo_var"] < 0.5]
    if len(fracas):
        par.append(
            (f"A variável {fracas.index[0]} tem" if len(fracas) == 1 else
             f"As variáveis {_lista(list(fracas.index))} têm")
            + " adequação individual (MSA) inferior a 0,5 e "
            + ("pode ser candidata" if len(fracas) == 1 else "podem ser candidatas") + " a sair da análise."
        )
    secoes.append({"titulo": "2. Adequação dos dados à PCA", "paragrafos": par})

    # 3. Valores próprios e decisão ------------------------------------------
    tabela = tv.head(max(k, min(len(tv), 8))).copy()
    tabela = tabela.round(3)
    sug = ctx["sugestoes"]
    par = [
        "Cada componente principal é um **vetor próprio** da matriz analisada e o respetivo "
        "**valor próprio (λ)** mede a variância que ela capta. A percentagem explicada é "
        f"λ<sub>i</sub> / Σλ; aqui a variância total é Σλ = {_fmt(res['variancia_total'])}"
        + (f", igual ao número de variáveis ({p}), porque os dados foram padronizados." if ctx["padronizar"] else "."),
        f"A PC1 explica {_fmt(tv['% Variância'].iloc[0], 1)}% da variância"
        + (f" e a PC2 {_fmt(tv['% Variância'].iloc[1], 1)}%." if len(tv) > 1 else "."),
        "Critérios de retenção aplicados: "
        + "; ".join(f"{nome}: **{v}** componente(s)" for nome, v in sug.items()) + ".",
    ]
    acum_k = tv["% Acumulada"].iloc[k - 1]
    decisao = (
        f"**Decisão:** foram retidas **{k} componentes**, que explicam **{_fmt(acum_k, 1)}%** da "
        f"variância total. Passa-se de {p} para {k} dimensões, perdendo {_fmt(100 - acum_k, 1)}% da informação."
    )
    concordam = [nome for nome, v in sug.items() if v == k]
    if concordam:
        decisao += f" Esta escolha coincide com: {_lista(concordam)}."
    else:
        decisao += " Esta escolha não coincide com nenhum critério automático; justifique-a pela interpretabilidade."
    if acum_k < ctx["limiar_acum"]:
        decisao += (
            f" Nota: fica abaixo do alvo de {ctx['limiar_acum']}% de variância acumulada."
        )
    par.append(decisao)
    secoes.append({"titulo": "3. Valores próprios e número de componentes", "paragrafos": par,
                   "tabela": tabela})

    # 4. Interpretação das componentes ----------------------------------------
    par = [
        f"Para interpretar cada componente consideram-se as variáveis com |loading| ≥ "
        f"{_fmt(ctx['limiar_loading'])}, ou seja, com correlação forte com essa componente. "
        "Variáveis no mesmo polo movem-se em conjunto; polos opostos indicam relação inversa."
    ]
    L = res["loadings"]
    for comp in pcs:
        col = L[comp]
        fortes = col[col.abs() >= ctx["limiar_loading"]].sort_values(key=np.abs, ascending=False)
        pos = [f"{v} ({_fmt(l)})" for v, l in fortes.items() if l > 0]
        neg = [f"{v} ({_fmt(l)})" for v, l in fortes.items() if l < 0]
        frase = f"**{comp}** ({_fmt(tv.loc[comp, '% Variância'], 1)}% da variância): "
        if not len(fortes):
            principal = col.abs().idxmax()
            frase += (f"nenhuma variável ultrapassa o limiar; a mais associada é {principal} "
                      f"({_fmt(col[principal])}). Componente de interpretação difícil.")
        else:
            if pos:
                frase += f"valores altos correspondem a {_lista(pos)}"
            if neg:
                frase += ("; e a valores baixos de " if pos else "valores altos correspondem a valores baixos de ") + _lista(neg)
            frase += "."
        par.append(frase)
    secoes.append({"titulo": "4. Interpretação das componentes", "paragrafos": par,
                   "tabela": L[pcs].round(3)})

    # 5. Variáveis mais relevantes -------------------------------------------
    imp = ctx["importancia"]
    media_esp = 100 / p
    relevantes = imp[imp["Contribuição ponderada (%)"] > media_esp]
    mal_repr = imp[imp["Comunalidade (Σ cos²)"] < 0.5]
    par = [
        "A PCA não elimina variáveis: cria componentes que são combinações de todas elas. "
        "Para identificar as mais relevantes calculou-se a contribuição de cada variável para as "
        f"{k} componentes retidas, ponderada pela variância de cada componente. Se todas "
        f"contribuíssem igualmente, cada uma teria {_fmt(media_esp)}%.",
        f"**Variáveis mais relevantes ({len(relevantes)} de {p}):** "
        + _lista([f"{v} ({_fmt(c)}%)" for v, c in relevantes["Contribuição ponderada (%)"].items()]) + ".",
    ]
    if len(mal_repr):
        par.append(
            (f"A variável {mal_repr.index[0]} tem" if len(mal_repr) == 1 else
             f"As variáveis {_lista(list(mal_repr.index))} têm")
            + f" comunalidade inferior a 0,5: menos de metade da variância é explicada pelas {k} "
            "componentes, pelo que " + ("fica mal representada." if len(mal_repr) == 1 else "ficam mal representadas.")
            + " Se forem importantes para o problema, considere reter mais componentes."
        )
    else:
        par.append(f"Todas as variáveis têm pelo menos 50% da sua variância explicada pelas {k} componentes.")
    tabela_imp = imp.drop(columns=["Acima da média?"]).round(3)
    secoes.append({"titulo": "5. Variáveis mais relevantes", "paragrafos": par, "tabela": tabela_imp})

    # 6. Observações ---------------------------------------------------------
    grupos = ctx.get("grupos")
    S = res["scores"]
    par = [
        "Cada observação foi projetada no novo espaço: o seu score numa componente é a soma dos "
        "valores padronizados multiplicados pelos coeficientes do vetor próprio "
        f"(ver separador Passo a passo). Com {k} componentes, cada observação passa de {p} para {k} números."
    ]
    tabela_grupos = None
    if grupos is not None and grupos.nunique() <= 15:
        cols = pcs[:min(k, 3)]
        tabela_grupos = S[cols].groupby(grupos.values).mean().round(3)
        tabela_grupos.index.name = "Grupo"
        for comp in cols[:2]:
            alto, baixo = tabela_grupos[comp].idxmax(), tabela_grupos[comp].idxmin()
            par.append(
                f"Em **{comp}**, o grupo *{alto}* tem o score médio mais alto ({_fmt(tabela_grupos.loc[alto, comp])}) "
                f"e *{baixo}* o mais baixo ({_fmt(tabela_grupos.loc[baixo, comp])})."
            )
    distancia2 = (S[pcs] ** 2 / tv.loc[pcs, "Valor próprio"]).sum(axis=1)
    atipicas = distancia2[distancia2 > stats.chi2.ppf(0.975, k)]
    if len(atipicas):
        par.append(
            f"{len(atipicas)} observação(ões) estão invulgarmente afastadas do centro no espaço das "
            f"componentes (distância de Mahalanobis acima do percentil 97,5%) e merecem verificação: "
            f"{_lista([_rotulo(ctx, i) for i in atipicas.sort_values(ascending=False).index[:8]])}"
            + (" (entre outras)." if len(atipicas) > 8 else ".")
        )
    secoes.append({"titulo": "6. Observações", "paragrafos": par, "tabela": tabela_grupos})

    # 7. Conclusão -----------------------------------------------------------
    top = list(relevantes.index[:5]) or list(imp.index[:3])
    conclusao = (
        f"Com {k} componentes retêm-se {_fmt(acum_k, 1)}% da informação das {p} variáveis originais. "
        f"As variáveis que mais estruturam os dados são {_lista(top)}. "
    )
    if kmo < 0.6 or ctx["p_bart"] >= 0.05:
        conclusao += "A adequação dos dados é fraca, pelo que estas conclusões são indicativas."
    else:
        conclusao += "Os testes de adequação suportam o uso da PCA neste conjunto de dados."
    secoes.append({"titulo": "7. Conclusão", "paragrafos": [conclusao]})
    return secoes


def relatorio_markdown(secoes: list[dict]) -> str:
    """Versão Markdown do relatório (para descarregar)."""
    linhas = ["# Relatório de Análise de Componentes Principais", "",
              f"Gerado em {datetime.now():%d/%m/%Y %H:%M}", ""]
    for s in secoes:
        linhas += [f"## {s['titulo']}", ""]
        for par in s["paragrafos"]:
            linhas += [re.sub(r"</?sub>", "", par), ""]
        if s.get("tabela") is not None:
            linhas += [s["tabela"].to_markdown(), ""]
    return "\n".join(linhas)


# ----------------------------------------------------------------------------
# Gráficos (matplotlib, para o PDF)
# ----------------------------------------------------------------------------
def _png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def graficos(ctx: dict) -> dict[str, io.BytesIO]:
    res, k = ctx["res"], ctx["k"]
    tv = res["variancia"]
    figs = {}

    # Scree plot
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(1, len(tv) + 1)
    ax.bar(x, tv["Valor próprio"], color=[COR_PRINCIPAL if i < k else "#B8CFD1" for i in range(len(tv))])
    if ctx["padronizar"]:
        ax.axhline(1, ls="--", color="grey", lw=1, label="Kaiser (λ = 1)")
    if ctx.get("paralela") is not None:
        ax.plot(x, ctx["paralela"][:len(tv)], ls=":", marker="o", ms=3, color=COR_SECUNDARIA,
                label="Análise paralela")
    ax.set_xticks(x, tv.index, rotation=45 if len(tv) > 10 else 0)
    ax.set_ylabel("Valor próprio")
    ax2 = ax.twinx()
    ax2.plot(x, tv["% Acumulada"], marker="o", color="#6B4C9A", label="% acumulada")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("% acumulada")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax.set_title("Scree plot (componentes retidas a cor forte)")
    figs["scree"] = _png(fig)

    # Ranking de variáveis
    imp = ctx["importancia"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.32 * len(imp))))
    media = 100 / len(imp)
    cores = [COR_PRINCIPAL if v > media else "#B8CFD1" for v in imp["Contribuição ponderada (%)"]]
    ax.barh(imp.index, imp["Contribuição ponderada (%)"], color=cores)
    ax.axvline(media, ls="--", color="grey", lw=1)
    ax.set_xlabel("Contribuição ponderada (%)")
    ax.set_title(f"Relevância das variáveis nas {k} componentes retidas")
    figs["ranking"] = _png(fig)

    if len(tv) >= 2:
        # Círculo de correlações
        L = res["loadings"]
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(t), np.sin(t), color="grey", lw=1)
        ax.plot(0.5 * np.cos(t), 0.5 * np.sin(t), color="lightgrey", lw=0.8, ls=":")
        for var in L.index:
            x1, y1 = L.loc[var, "PC1"], L.loc[var, "PC2"]
            ax.annotate("", xy=(x1, y1), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=COR_PRINCIPAL, lw=1.3))
            ax.text(x1 * 1.08, y1 * 1.08, var, fontsize=7, ha="center", va="center")
        ax.axhline(0, color="lightgrey", lw=0.8)
        ax.axvline(0, color="lightgrey", lw=0.8)
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect("equal")
        ax.set_xlabel(f"PC1 ({tv.loc['PC1', '% Variância']:.1f}%)")
        ax.set_ylabel(f"PC2 ({tv.loc['PC2', '% Variância']:.1f}%)")
        ax.set_title("Círculo de correlações")
        figs["circulo"] = _png(fig)

        # Scores
        S = res["scores"]
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        grupos = ctx.get("grupos")
        if grupos is not None and grupos.nunique() <= 12:
            for i, g in enumerate(sorted(grupos.unique())):
                m = (grupos == g).values
                ax.scatter(S.loc[m, "PC1"], S.loc[m, "PC2"], s=14, alpha=0.8, label=g,
                           color=plt.cm.tab10(i % 10))
            ax.legend(fontsize=7, frameon=False)
        else:
            ax.scatter(S["PC1"], S["PC2"], s=14, alpha=0.8, color=COR_PRINCIPAL)
        ax.axhline(0, color="lightgrey", lw=0.8)
        ax.axvline(0, color="lightgrey", lw=0.8)
        ax.set_xlabel(f"PC1 ({tv.loc['PC1', '% Variância']:.1f}%)")
        ax.set_ylabel(f"PC2 ({tv.loc['PC2', '% Variância']:.1f}%)")
        ax.set_title("Projeção das observações")
        figs["scores"] = _png(fig)
    return figs


# ----------------------------------------------------------------------------
# PDF
# ----------------------------------------------------------------------------
_SUBSTITUICOES = {"λ": "lambda", "Σ": "soma ", "≥": ">=", "≤": "<=", "μ": "média", "σ": "d.p.",
                  "−": "-", "·": "x", "²": "2"}


def _pdf_texto(texto: str) -> str:
    """Converte Markdown simples para a marcação do ReportLab e remove caracteres sem glifo."""
    for a, b in _SUBSTITUICOES.items():
        texto = texto.replace(a, b)
    sub = re.findall(r"<sub>.*?</sub>", texto)
    texto = texto.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for s in sub:  # repõe as tags <sub> válidas
        texto = texto.replace(s.replace("<", "&lt;").replace(">", "&gt;"), s)
    texto = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", texto)
    texto = re.sub(r"\*(.+?)\*", r"<i>\1</i>", texto)
    return texto


def _tabela_pdf(df: pd.DataFrame, largura: float) -> Table:
    df = df.copy()
    dados = [[_pdf_texto(str(df.index.name or ""))] + [_pdf_texto(str(c)) for c in df.columns]]
    for idx, linha in df.iterrows():
        dados.append([_pdf_texto(str(idx))] + [
            _fmt(v, 3) if isinstance(v, (float, np.floating)) else _pdf_texto(str(v)) for v in linha
        ])
    estilo_cel = ParagraphStyle("cel", fontName="Helvetica", fontSize=7, leading=8.5)
    estilo_cab = ParagraphStyle("cab", parent=estilo_cel, fontName="Helvetica-Bold", textColor=colors.white)
    dados = [[Paragraph(c, estilo_cab if i == 0 else estilo_cel) for c in row] for i, row in enumerate(dados)]
    n_col = len(dados[0])
    primeira = min(largura * 0.32, 4.5 * cm)
    larguras = [primeira] + [(largura - primeira) / (n_col - 1)] * (n_col - 1)
    t = Table(dados, colWidths=larguras, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(COR_PRINCIPAL)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF3F2")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9D6D5")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def _imagem(buf: io.BytesIO, largura_max: float) -> Image:
    img = Image(buf)
    escala = largura_max / img.imageWidth
    img.drawWidth = largura_max
    img.drawHeight = img.imageHeight * escala
    if img.drawHeight > 19 * cm:  # não ultrapassar a página
        f = 19 * cm / img.drawHeight
        img.drawHeight *= f
        img.drawWidth *= f
    return img


def gerar_pdf(secoes: list[dict], figs: dict[str, io.BytesIO], titulo_dados: str = "") -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Relatório PCA", author="App de Análise PCA")
    largura = A4[0] - 4 * cm
    base = getSampleStyleSheet()
    s_titulo = ParagraphStyle("t", parent=base["Title"], textColor=colors.HexColor(COR_PRINCIPAL), fontSize=20)
    s_h = ParagraphStyle("h", parent=base["Heading2"], textColor=colors.HexColor(COR_PRINCIPAL),
                         spaceBefore=12, spaceAfter=6)
    s_p = ParagraphStyle("p", parent=base["BodyText"], fontSize=9.5, leading=13.5,
                         alignment=TA_JUSTIFY, spaceAfter=5)
    s_meta = ParagraphStyle("m", parent=s_p, textColor=colors.grey, fontSize=8.5)

    graf_por_secao = {"3.": ["scree"], "4.": ["circulo"], "5.": ["ranking"], "6.": ["scores"]}

    story = [Paragraph("Relatório de Análise de Componentes Principais", s_titulo),
             Paragraph(_pdf_texto(f"{titulo_dados} — gerado em {datetime.now():%d/%m/%Y %H:%M}"), s_meta),
             Spacer(1, 8)]
    for s in secoes:
        bloco = [Paragraph(_pdf_texto(s["titulo"]), s_h)]
        bloco += [Paragraph(_pdf_texto(par), s_p) for par in s["paragrafos"]]
        story.append(KeepTogether(bloco[:2]))
        story += bloco[2:]
        for chave in graf_por_secao.get(s["titulo"][:2], []):
            if chave in figs:
                figs[chave].seek(0)
                story += [Spacer(1, 4), _imagem(figs[chave], largura * (0.62 if chave == "circulo" else 0.85)),
                          Spacer(1, 6)]
        if s.get("tabela") is not None:
            story += [Spacer(1, 4), _tabela_pdf(s["tabela"], largura), Spacer(1, 6)]

    def rodape(canvas, documento):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Página {documento.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=rodape, onLaterPages=rodape)
    return buffer.getvalue()
