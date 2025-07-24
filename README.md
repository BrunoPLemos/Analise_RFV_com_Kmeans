# Analise_RFV_com_Kmeans

https://analise-rfv-com-kmeans.onrender.com

Projeto de Segmentação de Clientes com Análise RFV e Clustering K-Means
Este projeto consiste em uma aplicação interativa desenvolvida com Streamlit para realizar análise de segmentação de clientes com base na metodologia RFV (Recência, Frequência e Valor). A ferramenta permite ao usuário:

Fazer upload de uma base de dados de compras;

 -Calcular métricas de Recência (dias desde a última compra), Frequência (número de compras) e Valor (valor total gasto);

 -Classificar os clientes com base em quartis e gerar um score RFV (como AAA, DDD, etc.);

 -Visualizar a distribuição dos clientes por segmento e sugerir ações de marketing personalizadas para cada grupo;

 -Aplicar o algoritmo de agrupamento K-Means, com apoio do Método do Cotovelo para definição do número ideal de clusters;

 -Visualizar os clusters em 2D usando PCA (Análise de Componentes Principais);

 -Exportar os dados segmentados com os clusters em planilhas Excel diretamente pela interface.

 Essa solução pode ser usada por áreas de marketing, CRM ou vendas para orientar estratégias de retenção, fidelização e reativação de clientes com base em dados.


 ## Principais Etapas Relacionadas a Dados
 
Importação e Pré-processamento dos Dados
  Upload de arquivo CSV/XLSX pelo usuário via Streamlit.

 -Leitura e interpretação de datas com parse_dates.

 -Validação da estrutura dos dados esperados (colunas como ID_cliente, DiaCompra, CodigoCompra, ValorTotal).

Cálculo das Métricas RFV
 -Recência (R): Dias desde a última compra de cada cliente.

 -Frequência (F): Número de compras feitas por cliente.

 -Valor (V): Total gasto por cliente no período.

 -Os dados são agregados por cliente e combinados em um único dataframe df_RFV.

Classificação RFV por Quartis
 -Uso de quartis estatísticos para segmentar os clientes em categorias (A, B, C, D) para cada métrica.

 -Criação de um RFV_Score combinando as letras (ex: “AAA”, “BCD”) para representar perfis de comportamento.

Segmentação Avançada com K-Means
 -Padronização dos dados (scaling) com StandardScaler.

 -Análise do Cotovelo para determinar o número ideal de clusters.

 -Agrupamento com KMeans, e visualização dos clusters usando PCA (redução de dimensionalidade).

Análise e Interpretação dos Clusters
 -Perfil médio de Recência, Frequência e Valor por cluster.

 -Visualização dos grupos formados em um gráfico bidimensional.

Mapeamento de Ações de Marketing
 -Associação de sugestões estratégicas para diferentes perfis RFV (ex: “AAA” → fidelização, “DDD” → churn).

 -Criação de uma nova coluna com as recomendações de ações.

Exportação dos Resultados
 -Geração de arquivos Excel para download com os dados RFV e clusters.

Interface interativa com botão para exportação direta via Streamlit.
