# Meta-Learning para Recomendação de Algoritmos IQA

Repositório dedicado ao projeto da disciplina **IN1097 - Tópicos Avançados em Agentes Inteligentes 2** (Mestrado Acadêmico - CIn UFPE).

## Sobre o Projeto

A avaliação da qualidade de imagem (*Image Quality Assessment* - IQA) é um problema onde nenhum algoritmo isolado (como SSIM, LPIPS ou BRISQUE) tem o melhor desempenho para todos os tipos de distorção. 

Este projeto constrói um **Meta-Dataset de IQA** consolidando fatias de 4 benchmarks consagrados (TID2013, CSIQ, KADID-10k e ChallengeDB). A partir dessas bases, utilizamos abordagens de **Meta-Learning (*Learning to Rank*)** — como Meta-Regressores e *HARRIS Forests* — para prever a ordem de eficácia dos algoritmos de IQA em novas(inéditas) bases de imagens.

### Estrutura Principal
- `src/`: Códigos-fonte do pipeline, extratores de meta-características, e modelos de Meta-Learning.
- `data/`: Base de dados agregada, contendo as bases de dados processadas.
- `main.py`: Orquestrador principal que executa a geração de features, treino e validação Leave-One-Dataset-Out.
- `notebooks/`: Análises experimentais e geração dos gráficos de resultados (Curvas de Perda e Diagrama CD).
- `scripts/`: Script .sh para execução em um cluster de IA
