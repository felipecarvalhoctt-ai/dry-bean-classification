# Roteiro para vídeo (6 a 8 minutos)

## 1. Apresentação (0:00–0:40)

“Olá! Neste projeto vou construir uma esteira de aprendizado de máquina para classificar sete variedades de feijão seco com base em características morfológicas extraídas de imagens.”

Na tela: título do projeto e pergunta de negócio.

## 2. Origem e contexto da base (0:40–1:30)

“Escolhi a base Dry Bean, do UCI Machine Learning Repository. A página informa 13.611 grãos, 16 atributos e sete classes. A licença é CC BY 4.0; a referência está no README.”

Na tela: página do UCI, em seguida a tabela resumo gerada em `outputs/resumo_base.json`.

## 3. Exploração e preparação (1:30–2:45)

“Começo pelas dimensões, tipos, ausentes, duplicatas e estatísticas descritivas. A base tem 13.611 linhas, 16 atributos, zero valores ausentes e 68 linhas duplicadas exatas, que removo antes das partições. Para as colunas, uso padronização. O scaler fica dentro do pipeline para aprender as escalas apenas no treinamento, sem vazamento.”

Na tela: `outputs/estatisticas_descritivas.csv`, contagem de classes e trecho do pipeline.

## 4. Divisão e treinamento (2:45–4:00)

“Faço uma divisão estratificada de 70% para treino, 15% para validação e 15% para teste, com semente 42. Comparo regressão logística e Random Forest usando F1 macro na validação. Depois, refaço o treinamento selecionado em treino mais validação.”

Na tela: `outputs/divisao_dados.json` e `outputs/comparacao_validacao.csv`.

## 5. Avaliação (4:00–5:30)

“Só agora avalio no conjunto de teste. A acurácia foi 91,83% e o F1 macro foi 93,17%, conforme `outputs/metricas_teste.json`. A matriz de confusão mostra quais variedades o modelo acerta e onde confunde classes.”

Na tela: matriz de confusão PNG e métricas por classe. A matriz mostra 78 acertos em 78 observações BOMBAY; SIRA foi a classe com menor F1, de 86,11%.

## 6. Predição demonstrativa (5:30–6:30)

“Por fim, carrego o pipeline salvo e mostro uma predição de uma observação reservada do teste. O modelo prevê DERMASON para o exemplo demonstrativo, com confiança estimada de 100%.”

Na tela: `outputs/exemplo_predicao.json` e arquivo `models/dry_bean_pipeline.joblib`.

## 7. Encerramento (6:30–7:00)

“O projeto está organizado para reprodução: o README explica a instalação, o código baixa a base e a esteira recria tabelas, gráfico, métricas, modelo e exemplo de inferência. O link do repositório estará na descrição.”

## Antes de gravar

- Instale Python e siga o README do início ao fim.
- Rode `python -m src.pipeline` e confirme que os arquivos listados no README foram criados.
- Se você alterar a base, a semente, o tratamento ou os modelos, atualize os números do roteiro com os arquivos de sua execução.
- Abra o gráfico da matriz de confusão e aumente o zoom para que os nomes das classes fiquem legíveis.
- Não mostre segredos, credenciais ou diretórios pessoais na gravação.




