# Previsão de Popularidade de Músicas no Spotify

**Grupo 4 · Engenharia de Software · Introdução à IA e Análise de Dados**  
**Prof. Wesley Andrade · Faculdade Donaduzzi**

---

## Objetivo

Prever o score de popularidade (0-100) de faixas musicais no Spotify com base em atributos de audio (danceability, energy, loudness, etc.) e genero musical, utilizando tecnicas de Machine Learning.

- **Tipo de tarefa:** Regressao
- **Metrica principal:** RMSE (Root Mean Squared Error)
- **Resultado final:** RMSE = 16.60 | R² = 0.446

## Dataset

- **Nome:** Spotify Tracks Dataset
- **Fonte:** [Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- **Tamanho:** 114.000 faixas, 22 colunas, 114 generos musicais
- **Target:** `popularity` (0-100)

## Estrutura do Repositorio

```
PREVISAO-DE-POPULARIDADE-SPOTIFY/
├── dataset/
│   ├── spotify-tracks-dataset.csv          # Dataset original (bruto)
│   └── dados-processados/
│       ├── modelo_projeto.pkl              # Modelo final treinado
│       ├── preprocessor_sprint2.pkl        # Pipeline de pre-processamento
│       ├── X_train_sprint2.csv             # Features de treino (processadas)
│       ├── X_test_sprint2.csv              # Features de teste (processadas)
│       ├── y_train_sprint2.csv             # Target de treino
│       └── y_test_sprint2.csv              # Target de teste
├── Sprint1_EDA.ipynb                       # Sprint 1: Analise Exploratoria
├── Sprint2.ipynb                           # Sprint 2: Pre-processamento
├── Sprint3.ipynb                           # Sprint 3: Modelagem
├── Sprint4.ipynb                           # Sprint 4: Interpretacao e conclusoes
└── README.md                              # Este arquivo
```

## Sprints

### Sprint 1 — Definicao do Problema e EDA

Analise exploratoria completa do dataset Spotify com 114.000 faixas musicais. Identificamos que a popularidade varia enormemente entre generos (de 2.2 a 59.3 pontos de media) enquanto as correlacoes lineares entre features de audio e o target sao fracas (todas < 0.10). Formulamos 3 hipoteses para guiar as sprints seguintes.

### Sprint 2 — Pre-processamento e Feature Engineering

Tratamento de outliers (duration_ms=0, tempo=0, winsorizacao), filtragem de conteudo nao-musical (podcasts), Target Encoding para `track_genre` (114 categorias) e criacao de 8 features engenheiradas. Pipeline reproduzivel com scikit-learn, sem vazamento de dados.

### Sprint 3 — Modelagem e Ajuste de Hiperparametros

Comparacao de 3 modelos (Ridge, RandomForest, HistGradientBoosting) com cross-validation (cv=5). Ajuste de hiperparametros com RandomizedSearchCV. Modelo final: HistGradientBoostingRegressor com RMSE = 16.60 e R² = 0.446. Conjunto de teste usado uma unica vez.

### Sprint 4 — Interpretacao e Conclusoes

Analise de erros por faixa de popularidade e por genero. SHAP values para interpretabilidade. Revisao das hipoteses: o genero musical e o preditor dominante (confirmando H1), faixas instrumentais sao menos populares (confirmando H2). O modelo acerta bem para generos homogeneos mas falha em prever hits virais.

## Resultados

| Modelo | RMSE (CV) | R² (CV) | RMSE (Teste) | R² (Teste) |
|---|---|---|---|---|
| Baseline (media) | 22.31 | 0.000 | — | — |
| Ridge | 19.20 | 0.259 | — | — |
| RandomForest | 18.22 | 0.333 | — | — |
| **HistGradientBoosting (final)** | **16.59** | **0.447** | **16.60** | **0.446** |

### Feature mais importante

`genre_pop_mean` (Target Encoding do genero) — SHAP medio 6x maior que qualquer feature de audio. Isso confirma que o *mercado* (publico e playlists) define popularidade mais do que os atributos sonoros da musica.

## Como Reproduzir

```bash
# 1. Instalar dependencias
pip install pandas numpy scikit-learn matplotlib seaborn shap joblib

# 2. Executar os notebooks na ordem
# Sprint1_EDA.ipynb → Sprint2.ipynb → Sprint3.ipynb → Sprint4.ipynb
# Os notebooks devem estar na mesma pasta que a pasta dataset/

# 3. Ou usar o modelo salvo diretamente
import joblib
modelo = joblib.load('dataset/dados-processados/modelo_projeto.pkl')
previsao = modelo.predict(X_novo_processado)
```

## Tecnologias

- Python 3.12
- pandas, numpy, matplotlib, seaborn
- scikit-learn (Pipeline, ColumnTransformer, HistGradientBoostingRegressor)
- SHAP (interpretabilidade)
- joblib (persistencia de modelos)
