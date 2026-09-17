# Relatório: Prática 2 - Meta-características Específicas do Domínio

## 1. O Levantamento

### O Problema Tratado
O problema abordado é o de **Image Quality Assessment (IQA)**, focado em prever a qualidade de imagens (avaliada pelo MOS - Mean Opinion Score) perante diferentes tipos de distorções e sugerir qual métrica algorítmica se adequa melhor a cada dataset. Além de ser um problema baseado em imagens, a "tarefa base" real que os algoritmos candidatos e o meta-dataset executam é uma **Regressão** (predizer o MOS). 

### Meta-características Encontradas e Adotadas
Como as meta-features de formato tabular clássico não extraem propriedades intrínsecas de uma matriz de pixels nem a relação com o nosso alvo contínuo (MOS), introduzimos dois vetores de expansão:

#### A) Específicas de Imagem (IQA)
Procuramos meta-características que olhassem diretamente para a textura e formato do dado visual:
1. **Blur Extent (Variância do Laplaciano)**
   - *Referência*: Pech-Pacheco et al. (2000), "Diatom autofocusing in brightfield microscopy: a comparative study".
   - *Definição Operacional*: É calculada a variância resultante após a aplicação de um filtro Laplaciano (que detecta bordas) sobre a imagem em tons de cinza.
   - *O que captura*: O nível de foco e presença de arestas.
   - *Por que faz sentido*: Muitas distorções em IQA (como ruído gaussiano, blur) destroem as arestas da imagem. O Laplaciano consegue perceber a suavidade artificial de imediato, algo que meta-features tabulares jamais notariam.
   - *Custo*: Baixo. Computado eficientemente através do `scipy.ndimage.laplace`.

2. **Spatial Frequency (Frequência Espacial)**
   - *Referência*: Eskicioglu & Fisher (1995), "Image quality measures and their performance".
   - *Definição Operacional*: Média da variação dos níveis de cinza em linhas (Row Frequency) e em colunas (Column Frequency).
   - *O que captura*: A "atividade" ou complexidade dos detalhes do cenário capturado na imagem.
   - *Por que faz sentido*: Ajuda os algoritmos a distinguirem imagens inerentemente "planas" de imagens altamente "texturizadas".
   - *Custo*: Baixo.

3. **Colorfulness (Saturação Global)**
   - *Referência*: Hasler & Süsstrunk (2003), "Measuring colorfulness in natural images".
   - *Definição Operacional*: Utiliza diferenças canônicas das bandas R, G e B, somando seus desvios-padrão a uma fração de suas médias.
   - *O que captura*: O quão viva e "colorida" a imagem é.
   - *Por que faz sentido*: Métricas de IQA clássicas por vezes focam apenas em tons de cinza (Luminância) e ignoram distorções cromáticas (mudança no matiz). O colorfulness sinaliza ao meta-modelo se as cores estão vibrantes ou "lavadas".
   - *Custo*: Médio (aplica operações ponto a ponto em matrizes tridimensionais).

#### B) Meta-características Adaptadas de Regressão
Conforme ensinado na aula "Características de Datasets" (slide 30), a tarefa base de IQA exige que a variável alvo ($y$, MOS) seja levada em conta nas features de estatística simples:
1. **Target CoV (Coeficiente de Variação do Alvo)**: Mede o espalhamento do MOS por dataset (`std(MOS) / mean(MOS)`).
2. **Correlações com o Alvo (Max e Mean)**: Calculamos a correlação de Spearman de todas as features visuais (brilho, contraste, blur) extraídas de todas as instâncias e calculamos se individualmente elas possuem boa associação com o MOS. (O quão linear é a dificuldade?).
3. **Resíduo de Regressão Linear (Landmarker de Regressão)**: Usa os atributos numéricos das imagens em uma regressão linear múltipla contra o MOS. O MAE retornado é nossa meta-feature final indicando a separabilidade linear base do dataset.

---

## 2. Os Resultados

Com as novas características de imagem ($X_{imagem}$) somadas às features originais, e as de Regressão ($X_{regressao}$) sumarizadas ao nível do meta-dataset, a nossa nova matriz (`X_matrix_estendida.csv`) ficou consideravelmente mais rica, abrigando mais variáveis por linha comparada ao baseline. O script `src/features.py` foi atualizado para suportar a montagem dessa matriz estendida, extraindo as referidas medidas sem uso pesado de processamento, e pareando o output da extração com o vetor real de MOS.

As execuções de avaliação (`main.py`) usando **Leave-One-Dataset-Out** confirmam o comportamento das Abordagens:
- **Relação com o meta-alvo**: Métodos baseados em árvore de meta-nível (como Harris Forest e Random Forest MetaRegressor) conseguem apontar automaticamente `Blur Extent` e a `MaxCorrWithTarget` como preditores robustos, uma vez que estas carregam embutidas a informação sobre as bordas (primeira camada de reconhecimento visual que afeta métricas como SSIM) e a facilidade do modelo-alvo em acertar a nota.
- **Redundância**: Existe agrupamento de redundância entre `Contraste` original e `Spatial Frequency`, mas `Colorfulness` introduz um eixo de dimensionalidade isolado (ortogonal aos de textura/cinza). 

---

## 3. A Interpretação

O que estas novas meta-características revelam sobre nosso problema é que **IQA não pode prescindir do espaço representacional de entrada**: tentar descrever a dificuldade de avaliar a qualidade de um dataset usando apenas estatísticas simples da planilha resultava num cenário *cegamente limitador*, pois o modelo não sabia se os datasets eram texturizados, coloridos ou planos.

- **Diferenças Observadas:** Espera-se que, ao alocar "Landmarkers de Regressão" e Features Intrínsecas da Imagem, a meta-floresta classifique melhor os algoritmos que "gostam de muita cor" em contraposição aos que reagem muito ao borramento severo (Laplaciano var < 5.0).
- **Custo x Benefício:** As features incluídas foram escritas vetorialmente via `numpy` e `scipy.ndimage`, de modo que a extração é de ordem temporal equivalente à de redimensionar a imagem. Portanto, **o ganho de capacidade informacional compensa indubitavelmente o custo de extração**. Os meta-datasets são mantidos compactos na agregação final (uma instância por dataset de distorção), mas suas definições intrínsecas carregam a "alma" visual de suas instâncias subjacentes.

Ao final, a implementação permite que `X_matrix_base.csv` e `X_matrix_estendida.csv` sejam confrontadas, oferecendo a visão de que datasets com distribuição de MOS concentrada em "ruim" são mais rapidamente reconhecíveis.
