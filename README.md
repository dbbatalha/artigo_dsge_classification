# Uma Abordagem Bibliométrica e de Machine Learning para Estudos DSGE

O presente artigo tem a intenção de predizer qual a melhor revista para ser publicado um artigo produzido que utilizou DSGE para modelos macroeconômicos, dado o seu abstract.  A hipótese é que cada revista possui modelos preferênciais. O trabalho se mostra relevante pois o tempo de apreciação e avaliação de artigos pelas comissões e pelo avaliador é grande e qualquer forma de diminuir possíveis reprovações ou rejeições do artigo é um ganho. Neste trabalho, fez-se uma revisão da literatura: dos artigos mais recentes que fizeram uso de modelos de equilíbrio geral estocástico e dinâmico; dos artigos que explicam a análise bibliométrica para observar as principais tendências e mapeamento científico; e dos artigos que explicam o problema de classificação aplicado ao Processamento de Linguagem Natural (NLP). Para isso, foi utilizado técnicas de *Machine Learning* como *Decision Tree* e *Random Forest*.

---------------

O *Dynamic Stochastic General Equilibrium* (DSGE) é um dos modelos econômicos mais utilizados em macroeconomia, principalmente por sua fundamentação microeconômica. Desta forma, considera que as relações entre as variáveis macroeconômicas são frutos de decisões ótimas de agentes econômicos, tais como famílias, firmas e autoridades monetária e fiscal, operando sob as restrições impostas pelo ambiente na qual cada agente opera. O modelo DSGE tem a capacidade de incorporar características empíricas da economia real como por exemplo, a adoção da hipótese de poder de mercado das firmas. Outra característica comumente empregada é a admissão de rigidez de preços e salários, que influenciam no efeito da política monetária sobre a economia. As fricções reais - tais como custos de ajustamento do capital, utilização variável da capacidade instalada e formação de hábito no consumo, como acelerador financeiro, enriquecem o modelo e o tornam mais próximo dos fenômenos econômicos. 

Em síntese, esses modelos tem como objetivos identificar as flutuações das variáveis macroeconômicas, entender os meios de propagação dos choques, prever os impactos das mudanças de políticas econômicas e até mesmo realizar previsões no futuro de variáveis chaves da economia. Para montar um modelo DSGE de forma eficiente, a estimação e inferência devem levar em conta a necessidade de conjugar coerência teórica e implementação empírica.

Para analisar os modelos que utilizaram DSGE, esse artigo fez uso de metodologias bibliométricas, que são consideradas úteis como ferramentas de apoio ao acompanhamento da evolução da ciência. Parte dessa disseminação se deve à abundância de dados e à facilidade de acessibilidade a um grande número de ferramentas de processamento bibliometrico. (Zuccala, 2016; Rousseau e Rousseau, 2017). Este artigo também se propõe a classificá-los, dado o seu abstract, buscando o melhor veículo para submetê-lo e diminuindo as chances de rejeição. Portanto, conforme o supracitado, temos um problema de classificação.

--------------
Breiman, L. 2001. Random forests. Machine Learning, 45, 5-32.

Rousseau, S., & Rousseau, R. 2017. Being metric-wise: Heterogeneity in Bibliometric Knowledge. Prof. La Inf. 26: 480. doi:10.3145/epi.2017.may.14

Zuccala, A. 2016. Inciting the Metric Oriented Humanist: Teaching Bibliometrics in a Faculty of Humanities. Educ. Inf. 32: 149–164. doi:10.3233/EFI-150969
