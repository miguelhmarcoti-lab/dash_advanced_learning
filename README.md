# War Risk Monitor — Advanced Learning Dashboard

Dashboard Flask para visualização de risco de conflito armado por país.

## Estrutura
```
dash_advanced_learning/
├── data/
│   └── panel_final.csv          # base de dados (UCDP + World Bank + Polity)
├── model/
│   ├── model.py                 # ← EDITE AQUI para mudar o modelo
│   ├── nn_model.keras           # gerado após treino
│   ├── scaler.joblib            # gerado após treino
│   ├── predictions.json         # probabilidades por país
│   ├── baseline_results.json    # métricas dos modelos clássicos
│   ├── nn_results.json          # métricas + histórico da rede neural
│   ├── all_results.json         # todos os resultados combinados
│   └── model_config.json        # arquitetura e parâmetros
└── app/
    ├── app.py                   # Flask app
    └── templates/
        ├── base.html
        ├── map.html             # Página 1: mapa-múndi
        ├── table.html           # Página 2: tabela de países
        ├── model.html           # Página 3: arquitetura do modelo
        └── evaluation.html      # Página 4: comparação de modelos
```

## Setup

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Treinar os modelos (gera todos os artefatos em model/)
cd model
python model.py

# 3. Rodar o dashboard
cd ../app
python app.py
# Acesse: http://localhost:5050
```

## Como atualizar o modelo

Edite `model/model.py`. Os pontos principais:

- **`NN_CONFIG`** — dicionário com todos os hiperparâmetros da rede
- **`build_nn_model(input_dim)`** — função que constrói a arquitetura Keras
- **`build_features_nn(df, target)`** — quais colunas entram como features
- **`BASELINE_MODELS`** — modelos clássicos para comparação

Após editar, basta rodar `python model.py` novamente. O dashboard lê automaticamente os novos JSONs.

## Páginas

| URL | Descrição |
|-----|-----------|
| `/map` | Mapa-múndi com probabilidades em color code gradual |
| `/table` | Tabela de países com bandeiras e probabilidades |
| `/model` | Arquitetura, parâmetros e resultados da Rede Neural |
| `/evaluation` | Comparação de todos os modelos (baseline + rede neural) |
# dash_advanced_learning
