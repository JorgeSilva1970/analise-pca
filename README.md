# Análise PCA — app web em Streamlit

App para correr uma Análise de Componentes Principais sobre dados carregados de CSV/Excel
ou inseridos manualmente, escolhendo 2 ou mais variáveis. Identifica as variáveis mais
relevantes, mostra a matemática passo a passo e gera um relatório (PDF/Markdown) que
explica os resultados e as decisões.

## Estrutura
```
pca-app/
├── app.py                 # interface Streamlit (8 separadores)
├── pca_core.py            # cálculos: PCA, KMO, Bartlett, análise paralela, ranking, projeções
├── relatorio.py           # relatório automático: texto, gráficos matplotlib e PDF (ReportLab)
├── requirements.txt
└── .streamlit/config.toml # tema e limite de upload
```

## Correr localmente (VS Code)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
python -m streamlit run app.py
```

## GitHub
```bash
git init
git add .
git commit -m "App de análise PCA"
git branch -M main
git remote add origin https://github.com/JorgeSilva1970/analise-pca.git
git push -u origin main
```

## Publicar no Render (Web Service)
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- Variável de ambiente: `PYTHON_VERSION=3.12.4`

Alternativa: Streamlit Community Cloud (share.streamlit.io), escolhendo o repositório e `app.py`.

## Formato dos dados
Uma linha por observação e uma coluna por variável. Colunas de texto podem servir como
grupo (para colorir gráficos) ou identificação. CSV com `;` ou `,` é detetado
automaticamente; escolha o separador decimal na barra lateral.
