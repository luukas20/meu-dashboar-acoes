# 💹 Dashboard Interativo de Ações

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lucas-estatistico-acoes.streamlit.app/)

## 📜 Descrição

Este é um aplicativo web interativo construído com **Streamlit** para visualização e análise de dados históricos de ações. O aplicativo utiliza a biblioteca `yfinance` para buscar dados em tempo real da B3 e de outras bolsas de valores globais, e usa `Plotly` para gerar gráficos dinâmicos.

Adicionalmente, o projeto inclui um modelo de Machine Learning (Keras/ARMA-GARCH) para realizar previsões de preços futuros com base nos dados históricos.

---

## 📸 Demonstração

*Você pode adicionar um screenshot ou um GIF animado do seu aplicativo aqui para uma melhor visualização.*

*![Demonstração do App](URL_DO_SEU_SCREENSHOT_OU_GIF)

**Acesse a versão ao vivo do aplicativo aqui:**
[**https://lucas-estatistico-acoes.streamlit.app/**](https://lucas-estatistico-acoes.streamlit.app/)

---

## ✨ Funcionalidades

* **Busca de Dados Dinâmica:** Insira qualquer ticker de ação (ex: `PETR4.SA`, `AAPL`, `MGLU3.SA`) para buscar dados.
* **Seleção de Período:** Escolha o intervalo de datas desejado através de um calendário interativo.
* **Visualização de Preços:** Gráfico interativo com a evolução do preço de fechamento da ação.
* **Análise de Volume:** Gráfico de barras com o volume de negociações diárias.
* **Previsão de Preços:** Utiliza um modelo treinado para prever os preços futuros e os compara com os valores reais.
* **Interface Amigável:** Todos os controles estão na barra lateral para uma experiência de usuário limpa e intuitiva.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3
* **Framework Web:** [Streamlit](https://streamlit.io/)
* **Análise de Dados:** [Pandas](https://pandas.pydata.org/) e [NumPy](https://numpy.org/)
* **Busca de Dados Financeiros:** [yfinance](https://pypi.org/project/yfinance/)
* **Visualização de Dados:** [Plotly](https://plotly.com/python/)
* **Machine Learning (Previsão):** [TensorFlow (Keras)](https://www.tensorflow.org/), [Statsmodels](https://www.statsmodels.org/), [Arch](https://arch.readthedocs.io/en/latest/)

---

## 🚀 Como Rodar o Projeto Localmente

Para executar este projeto na sua máquina, siga os passos abaixo:

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/luukas20/meu-dashboar-acoes.git](https://github.com/luukas20/meu-dashboar-acoes.git)
    cd meu-dashboar-acoes
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    # Criar o ambiente
    python -m venv .venv

    # Ativar no Windows (PowerShell)
    .\.venv\Scripts\Activate.ps1

    # Ativar no Linux/Mac
    source .venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute o aplicativo Streamlit:**
    ```bash
    streamlit run app.py
    ```

O aplicativo abrirá automaticamente no seu navegador padrão.

---

## 📂 Estrutura do Projeto

```
├── app.py                # Script principal do Streamlit
├── requirements.txt      # Lista de dependências Python
├── README.md             # Documentação do projeto
├── .gitignore            # Arquivos a serem ignorados pelo Git
└── models/
    └── meu_modelo.keras  # Modelo de previsão treinado
```

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
*(Dica: se não tiver um arquivo de licença, você pode adicionar um facilmente no GitHub clicando em "Add file" > "Create new file" e digitando `LICENSE` como nome do arquivo. O GitHub oferecerá modelos prontos, como o MIT.)*