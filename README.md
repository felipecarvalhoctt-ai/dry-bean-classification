# Classificação de variedades de feijão seco (Dry Bean)

Projeto de classificação multiclasse com dados do repositório UCI. A esteira gera estatísticas descritivas, trata duplicatas, divide os dados em treino/validação/teste, compara dois modelos, avalia o selecionado e salva um pipeline para inferência.

> Os resultados abaixo foram gerados executando o pipeline. Eles podem ser reproduzidos pelos comandos indicados nesta página.

## Pergunta do projeto

É possível classificar uma variedade de feijão seco a partir de atributos morfológicos extraídos de imagens?

## Base de dados

- Nome: Dry Bean
- Origem: [UCI Machine Learning Repository — Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean)
- DOI: [10.24432/C50S4B](https://doi.org/10.24432/C50S4B)
- Licença: CC BY 4.0; cite a fonte ao redistribuir/adaptar os dados.
- A descrição da UCI informa 13.611 grãos, 16 atributos numéricos, sete classes e ausência de valores ausentes.
- A lista de classes documentada pela UCI: Seker, Barbunya, Bombay, Cali, Dermosan, Horoz e Sira.
- A base original é baixada via pacote `ucimlrepo` na primeira execução e salva em `data/raw/` (ignorado pelo Git).

## Requisitos atendidos

| Etapa | Implementação |
|---|---|
| 2. Estatísticas descritivas | Dimensões, tipos, ausentes, duplicatas, distribuição de classes e `describe()` das variáveis numéricas. |
| 3. Transformação de colunas | `StandardScaler` para atributos numéricos. É ajustado dentro do pipeline somente no treino (e depois no treino+validação para o ajuste final). |
| 4. Transformação de linhas | Remoção de linhas completamente duplicadas antes da partição; contagem registrada em `outputs/divisao_dados.json`. |
| 5. Partições | Divisão estratificada 70% treino, 15% validação, 15% teste, semente fixa 42. |
| 6. Treinamento | Regressão Logística e Random Forest como candidatos. |
| 7. Avaliação | Seleção por F1 macro na validação; ajuste final com treino+validação; teste reservado para avaliação final. |
| 8. Métricas | Acurácia, F1 macro, relatório por classe, matriz de confusão CSV e PNG. |
| 9. Predição | Exemplo de predição em uma observação do teste e pipeline persistido em `models/`. |

### Decisões metodológicas

- A estratificação mantém proporções de classe semelhantes nas três partições.
- A semente aleatória é `42` para permitir repetição dos resultados.
- Duplicatas exatas são removidas antes da divisão para reduzir o risco de cópias idênticas caírem em conjuntos diferentes. O resumo descritivo é calculado sobre o conjunto original; a quantidade removida é relatada separadamente.
- A padronização não é pré-calculada em todos os dados. O pipeline ajusta o scaler apenas nos dados usados para treinar o modelo, prevenindo vazamento de informação.
- A seleção acontece na validação; o conjunto de teste fica separado até a medição final.
- O exemplo de predição usa uma linha do conjunto de teste para demonstrar a chamada do modelo. A classe verdadeira dessa linha fica disponível no CSV original de partições e pode ser exibida lado a lado em uma apresentação.

## Resultados desta execução

Executado em Python 3.12 com a semente 42, após remover 68 linhas duplicadas exatas:

- Partições: 9.480 treino, 2.031 validação e 2.032 teste.
- Seleção na validação: Random Forest, F1 macro 0,9319 (acurácia 0,9197); Regressão Logística, F1 macro 0,9289 (acurácia 0,9168).
- Teste final do Random Forest: acurácia **0,9183** e F1 macro **0,9317**.
- Predição demonstrativa: classe **DERMASON**, confiança estimada **1,00** para essa classe.
- A classe BOMBAY teve 78 acertos em 78 exemplos de teste; SIRA teve o menor F1 (0,8611), em especial pelas confusões com DERMASON (40 casos) e HOROZ (9 casos).

A confiança é a probabilidade estimada pelo classificador para uma única amostra; não deve ser interpretada como garantia de acerto. Os arquivos detalhados ficam em `outputs/`.
## Como executar

Requer Python 3.10 ou superior.

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.pipeline
```

Se o PowerShell bloquear a ativação, execute o programa pelo caminho direto: `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` e `.\.venv\Scripts\python.exe -m src.pipeline`.

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.pipeline
```

A primeira execução precisa de internet para obter os dados da UCI. Depois do download, a cópia local em `data/raw/dry_bean.csv` será usada em execuções seguintes.

## Notebook

Abra `notebooks/analise_dry_bean.ipynb` após instalar as dependências:

```bash
jupyter lab
```

O notebook executa a mesma esteira do módulo Python e exibe os resultados gerados.

## Saídas geradas

- `outputs/resumo_base.json`: visão geral da base
- `outputs/estatisticas_descritivas.csv`: estatísticas dos atributos
- `outputs/distribuicao_classes.csv`: frequência e proporção das classes
- `outputs/divisao_dados.json`: tamanhos das partições e duplicatas removidas
- `outputs/comparacao_validacao.csv`: comparação dos candidatos na validação
- `outputs/metricas_teste.json`: acurácia e F1 macro no teste
- `outputs/metricas_por_classe.csv`: precisão, revocação, F1 e suporte por classe
- `outputs/matriz_confusao.csv` e `outputs/matriz_confusao.png`
- `outputs/exemplo_predicao.json`: entrada, classe prevista e confiança
- `models/dry_bean_pipeline.joblib`: pipeline serializado para inferência
- `data/processed/dry_bean_particoes.csv`: observações e partição de cada linha

Os arquivos gerados, dados e modelo estão no `.gitignore` para manter o repositório leve e evitar incluir artefatos produzidos localmente sem revisão. O código recria esses arquivos.

## Usar o modelo para uma nova predição

Depois de rodar a esteira, use as 16 colunas de atributos (sem `Class`):

```python
import joblib
import pandas as pd

modelo = joblib.load("models/dry_bean_pipeline.joblib")
novo_grao = pd.DataFrame([{
    # Preencha com os mesmos nomes e unidades das colunas em data/raw/dry_bean.csv
}])
classe = modelo.predict(novo_grao)[0]
probabilidades = modelo.predict_proba(novo_grao)[0]
print(classe)
```

Para um exemplo pronto que funciona imediatamente após a execução, veja `outputs/exemplo_predicao.json` e a seção de predição do notebook.

## Conteúdo do repositório

```text
.
├── data/                 # dados baixados/derivados (ignorados pelo Git)
├── models/               # pipeline serializado (ignorado pelo Git)
├── notebooks/            # análise demonstrativa
├── outputs/              # tabelas, métricas e gráfico (ignorados pelo Git)
├── src/pipeline.py       # esteira reprodutível
├── requirements.txt
└── roteiro_video.md
```

## Citação

Koklu, M. & Ozkan, I. A. (2020). *Dry Bean* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50S4B


