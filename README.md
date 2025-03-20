# Predição de Churn em Telecomunicações

Este repositório apresenta um projeto de ciência de dados focado na predição de churn (evasão de clientes) em uma empresa de telecomunicações. O objetivo é construir um modelo de machine learning capaz de identificar clientes com alta probabilidade de cancelar seus serviços, permitindo que a empresa tome ações proativas de retenção.

## Contexto

A retenção de clientes é crucial para o sucesso de empresas de telecomunicações.  A perda de clientes (churn) impacta diretamente a receita e a lucratividade. Este projeto utiliza dados reais de uma empresa de telecomunicações (disponibilizados pela IBM) para desenvolver um modelo preditivo que auxilia na identificação de clientes em risco de churn.

## Dataset

O dataset utilizado neste projeto é o "Telco Customer Churn" da IBM, disponível publicamente.  Ele contém informações sobre clientes de uma empresa de telecomunicações, incluindo:

*   **Churn:**  Indica se o cliente cancelou o serviço no último mês (variável alvo).
*   **Serviços Contratados:**  Detalhes sobre os serviços que cada cliente assinou (telefone, múltiplas linhas, internet, segurança online, backup online, proteção de dispositivo, suporte técnico, streaming de TV e filmes).
*   **Informações da Conta:**  Tempo de contrato, tipo de contrato, método de pagamento, faturamento eletrônico, cobranças mensais e cobranças totais.
*   **Dados Demográficos:**  Gênero, faixa etária, presença de parceiros e dependentes.

Link para o dataset original (versão atualizada): [https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)

## Estrutura do Repositório

```
telco-churn-prediction/
├── data/
│  ├── raw/                                       <- Dados brutos (originais do dataset)
│  └── processed/                                 <- Dados pré-processados, prontos para o modelo
├── scripts/
│  ├── 01-exploratory-data-analysis.ipynb         <- Análise exploratória dos dados
│  ├── 02-data-preprocessing.ipynb                <- Pré-processamento e feature engineering
│  ├── 03-model-training.ipynb                    <- Treinamento e avaliação de modelos
│  ├── 04-model-interpretation.ipynb              <- Interpretação dos resultados e insights
│  └── 05-deployment-simulation.ipynb (Opcional)  <- Simulação de um ambiente de deploy
├── src/
│  ├── data_processing.py                         <- Funções de pré-processamento
│  ├── modeling.py                                <- Funções de treinamento e avaliação de modelos
│  └── utils.py                                   <- Funções utilitárias
├── models/                                       <- Modelos treinados e serializados (e.g., arquivos .pkl)
├── reports/
│  ├── figures/                                   <- Gráficos e visualizações
│  └── metrics.csv                                <- Métricas de avaliação dos modelos
├── environment.yml                               <- Arquivo de ambiente Conda (ou requirements.txt)
├── README.md                                     <- Este arquivo
```
## Metodologia

1.  **Análise Exploratória de Dados (EDA):**
    *   Análise das distribuições das variáveis.
    *   Identificação de valores ausentes e outliers.
    *   Visualização das relações entre as variáveis e o churn.
    *   Criação de hipóteses sobre os fatores que influenciam o churn.

2.  **Pré-processamento e Feature Engineering:**
    *   Tratamento de valores ausentes (imputação ou remoção).
    *   Codificação de variáveis categóricas (one-hot encoding, label encoding).
    *   Normalização ou padronização de variáveis numéricas.
    *   Criação de novas features (ex: combinar variáveis existentes, criar features baseadas em conhecimento de domínio).
    *   Seleção de features (usando técnicas como análise de correlação, importância de features em modelos de árvore).
    *   Divisão dos dados em conjuntos de treino, validação e teste.

3.  **Treinamento e Avaliação de Modelos:**
    *   Seleção de modelos de classificação adequados para o problema (Regressão Logística, Árvores de Decisão, Random Forest, Gradient Boosting, SVM, etc.).
    *   Treinamento dos modelos com o conjunto de treino.
    *   Ajuste de hiperparâmetros usando validação cruzada (cross-validation) no conjunto de validação.
    *   Avaliação do desempenho dos modelos no conjunto de teste, utilizando métricas como:
        *   Precisão (Precision)
        *   Recall (Revocação)
        *   F1-score
        *   AUC-ROC (Área sob a curva ROC)
        *   Matriz de Confusão
    *   Comparação dos resultados dos diferentes modelos.
    *   Escolha do modelo com melhor desempenho e interpretabilidade.

4. **Interpretação e Insights:**
    *  Análise da importância das features no modelo selecionado.
    *  Identificação dos principais fatores que contribuem para o churn.
    *  Extração de insights acionáveis para a empresa (ex: quais clientes segmentar para ações de retenção).
    *  Sugestões de estratégias de retenção.

5. **(Opcional) Simulação de Deploy:**
   *   Criação de um script (ou API simples) que simula a aplicação do modelo em produção.
   *    Exemplo: Receber dados de um novo cliente e retornar a probabilidade de churn.

## Ferramentas e Bibliotecas

*   **Linguagem:** Python
*   **Bibliotecas:**
    *   Pandas (manipulação de dados)
    *   NumPy (computação numérica)
    *   Scikit-learn (machine learning)
    *   Matplotlib e Seaborn (visualização)
    *   XGBoost/LightGBM/CatBoost (gradient boosting - opcional)
    *   SHAP/LIME (interpretação de modelos - opcional)
    *   Streamlit/Flask (para criar uma API simples - opcional)

## Como Executar

1.  Clone este repositório: `git clone https://github.com/<seu-usuario>/telco-churn-prediction.git`
2.  Crie um ambiente virtual (recomendado): `conda create -n churn-env python=3.9` ou `python -m venv churn-env`
3.  Ative o ambiente: `conda activate churn-env` ou `source churn-env/bin/activate`
4.  Instale as dependências: `pip install -r environment.yml` (ou `pip install -r requirements.txt`)
5.  Execute os notebooks Jupyter na ordem numérica.

## Conclusões e Próximos Passos

Este projeto fornece um modelo de predição de churn e insights valiosos para a empresa de telecomunicações.  Alguns próximos passos possíveis incluem:

*   **Refinar o modelo:**  Experimentar com outros algoritmos, técnicas de feature engineering e ajuste de hiperparâmetros.
*   **Desenvolver um sistema de alerta:**  Integrar o modelo a um sistema que notifique a empresa sobre clientes com alto risco de churn.
*   **Implementar testes A/B:** Testar diferentes estratégias de retenção para avaliar seu impacto real.
*   **Monitorar o desempenho do modelo:**  Acompanhar as métricas do modelo ao longo do tempo e re-treinar o modelo periodicamente.
*  **Incorporar dados externos:** Adicionar informações de outras fontes (redes sociais, dados de mercado) para enriquecer o modelo.
*  **Segmentar clientes:** Criar modelos de churn específicos para diferentes grupos de clientes (por exemplo, por plano ou tempo de contrato).
* **Análise de Sobrevivência**: Em vez de simplesmente prever se um cliente irá cancelar (sim/não), prever *quando* ele irá cancelar.

Este projeto é um ponto de partida para a construção de um sistema completo de prevenção de churn e demonstra as habilidades e conhecimentos do desenvolvedor em ciência de dados e machine learning.
