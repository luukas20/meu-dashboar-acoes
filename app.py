import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from arch.univariate import ConstantMean, GARCH, Normal, arch_model
from statsmodels.tsa.arima.model import ARIMA

@st.cache_data
def carregar_dados(ticker, data_inicio, data_fim):
    """
    Busca dados históricos e adiciona a última cotação intraday, se for de um novo dia.
    """
    # 1. Cria o objeto Ticker para o ativo desejado
    require = yf.Ticker(ticker)

    # 2. Busca o histórico diário principal
    data = require.history(start=data_inicio, end=data_fim)

    # Se não houver dados históricos, não há o que fazer.
    if data.empty:
        return data

    # 3. Busca dados intraday bem recentes (últimos 2 dias, intervalo de 1 minuto)
    bd = require.history(period="2d", interval='1m')
    
    # Se não houver dados intraday, apenas retorne o histórico principal
    if bd.empty:
        return data

    # 4. Compara as datas e concatena se necessário
    # Checa se o último dia do histórico é diferente do último dia do intraday
    if data.index[-1].date() != bd.index[-1].date():
        # Se forem diferentes, anexa a última cotação intraday ao histórico
        # Usamos pd.concat para garantir que os dataframes sejam unidos corretamente
        data = pd.concat([data, bd.tail(1)])
        
    return data

def modelo_garch(bd):
    # --- 1. Gerar dados simulados (ou você pode carregar seus dados reais) ---
    # Retorno simples
    returns = bd['Close'].pct_change().dropna()
    initial_price = bd['Close'].iloc[-1]

    # --- 2. Ajustar o ARMA(9,10) para a média ---
    # Ajustando um ARMA(9,10) com statsmodels
    arma_order = (9, 0, 10)  # (p, d, q)
    arma_model = ARIMA(returns, order=arma_order)
    arma_model = arma_model.fit()

    # --- 3. Obter os resíduos do modelo ARMA ---
    arma_residuals = arma_model.resid

    # --- 4. Modelar a volatilidade dos resíduos com GARCH(2,1) ---
    # ConstantMean significa que vamos modelar a volatilidade apenas, média constante (resíduo é a parte importante)
    mean_model = ConstantMean(arma_residuals)

    # Definir o modelo de volatilidade GARCH(2,1)
    vol_model = GARCH(p=2, q=1)

    # Distribuição dos resíduos: Normal (pode trocar para 'StudentsT', etc)
    dist = Normal()

    # Setando os componentes no modelo
    mean_model.volatility = vol_model
    mean_model.distribution = dist

    # Ajustar o modelo completo
    garch_res = mean_model.fit()

    # --- 5. Resultados ---
    print(garch_res.summary())

    # --- 6. Fazer previsões ---
    horizon = 10  # Quantidade de passos à frente
    forecast = garch_res.forecast(horizon=horizon)

    # Acesso às previsões
    variance_forecast = forecast.variance.values[-1]  # Variância prevista
    mean_forecast = arma_model.forecast(steps=horizon)  # Previsão da média

    # Média prevista dos retornos
    return_forecast = mean_forecast

    # Desvio padrão previsto (sqrt da variância)
    std_forecast = np.sqrt(variance_forecast)

    # Gerar choques aleatórios baseados no desvio previsto
    # np.random.seed(42)  # Para reprodutibilidade
    random_shocks = np.random.normal(0, variance_forecast)

    # Retornos ajustados = média prevista + choque aleatório
    adjusted_returns = mean_forecast + variance_forecast * random_shocks

    # print("Previsão de retornos:")
    # print(return_forecast)

    # print("\nPrevisão de variâncias (volatilidades ao quadrado):")
    # print(variance_forecast)

    # print("\nPrevisão de desvio padrão (raiz da volatilidades ao quadrado):")
    # print(std_forecast)

    # print("\nPrevisão de choques aleatórios:")
    # print(random_shocks)

    # print("\nPrevisão de retornos ajustados:")
    # print(adjusted_returns)

    # --- 7. Conversão inversa: De retorno para preço ---

    # Convertendo retornos previstos em preços (assumindo preço inicial)
    predicted_prices = [initial_price]
    for r in adjusted_returns:
        next_price = predicted_prices[-1] * (1 + r)
        predicted_prices.append(next_price)

    predicted_prices = np.array(predicted_prices[1:])  # Remover o preço inicial

    # Pegar o fitted (previsão da média)
    arma_fitted = arma_model.fittedvalues

    # Pegar fitted da volatilidade
    garch_fitted_volatility = garch_res.conditional_volatility  # desvio-padrão

    # Gerar choques aleatórios baseados no desvio previsto
    # np.random.seed(42)  # Para reprodutibilidade
    random_shocks = np.random.normal(0, garch_fitted_volatility)

    # Retornos ajustados = média prevista + choque aleatório
    adjusted_returns_fitted = arma_fitted + garch_fitted_volatility*random_shocks

    # Convertendo retornos previstos em preços (assumindo preço inicial)
    fitted_prices = [bd['Close'].iloc[0]]
    # Loop para calcular os próximos valores
    for i in range(len(adjusted_returns_fitted)):
        next_value = bd['Close'].iloc[i] * (1 + adjusted_returns_fitted[i])
        fitted_prices.append(next_value)


    # --- Montar o DataFrame organizado ---

    # Ajustar o comprimento: às vezes o modelo ARMA gera fitteds um pouco menores
    min_len = min(len(returns), len(arma_fitted), len(garch_fitted_volatility))

    base_resultados = pd.DataFrame({'Date': bd.index,
                        'Close': bd['Close'],
                        'Open': bd['Open'],
                        'High': bd['High'],
                        'Low': bd['Low'],
                        'Fitted':  np.array(fitted_prices),
                        'Predict':  np.full(len(bd.index), np.nan)})

    dates = pd.to_datetime(bd.index)
    predict_dates = pd.date_range(list(dates)[-1]+pd.DateOffset(1), periods=10,freq='b').tolist()

    df_predict = pd.DataFrame({'Date': np.array(predict_dates),
                            'Close': np.full(len(predicted_prices), np.nan),
                            'Open':  np.full(len(predicted_prices), np.nan),
                            'High':  np.full(len(predicted_prices), np.nan),
                            'Low':  np.full(len(predicted_prices), np.nan),
                            'Fitted':  np.full(len(predicted_prices), np.nan),
                            'Predict': np.array(predicted_prices)})

    bd_final = pd.concat([base_resultados, df_predict], axis=0)

    bd_final['Date'] = pd.to_datetime(bd_final['Date'], format='%Y-%m-%d')

    return bd_final



# --- Configuração da Página ---
st.set_page_config(
    page_title="Dashboard de Ações",
    page_icon="💹",
    layout="wide"
)

# --- Título e Descrição ---
st.title("💹 Dashboard de Ações")
st.markdown("Use a barra lateral para selecionar a ação e o período desejado.")

# --- Barra Lateral (Sidebar) para Inputs do Usuário ---
st.sidebar.header("Opções")

# Input para o ticker da ação
ticker_symbol = st.sidebar.text_input(
    "Digite o Ticker da Ação",
    "BOVA11.SA" # Valor padrão (Bovespa)
).upper()

# Inputs para a data de início e fim
# Define as datas padrão (últimos 365 dias)
end_date_default = date.today()
start_date_default = '2018-01-01'

start_date = st.sidebar.date_input(
    "Data de Início",
    start_date_default
)

end_date = st.sidebar.date_input(
    "Data de Fim",
    end_date_default
)

# --- Lógica Principal ---
# Adiciona um botão para buscar os dados
if st.sidebar.button("Buscar Dados"):
    # del require
    # del data 
    # del bd
    # Verifica se as datas são válidas
    if start_date > end_date:
        st.error("Erro: A data de início não pode ser posterior à data de fim.")
    else:
        # Adiciona um placeholder de carregamento
        with st.spinner(f"Buscando dados para {ticker_symbol}..."):
            
            try:
                data = carregar_dados(ticker_symbol, start_date, end_date)
                base_resultados = modelo_garch(data)
                base_resultados.sort_values(by='Date', ascending=False, inplace=True)

                base_filtrada = base_resultados.head(360)

                # Validação dos dados
                if data.empty:
                    st.error(f"Nenhum dado encontrado para o ticker '{ticker_symbol}'. Verifique o código da ação (ex: PETR4.SA, MGLU3.SA, AAPL).")
                else:
                    st.success(f"Dados de {ticker_symbol} carregados com sucesso!")

                    # --- Exibição dos Dados ---
                    st.header(f"Dados Históricos para {ticker_symbol}", divider='rainbow')

                    # Exibe o dataframe com os dados brutos
                    st.dataframe(data.sort_index(ascending=False), use_container_width=True)

                    # --- Visualização com Gráficos (Plotly) ---
                    st.subheader("Gráfico de Preço de Fechamento", divider='rainbow')
                    # Cria o gráfico de linha com o Plotly Express
                    fig_close = px.line(
                        data,
                        x=data.index,
                        y='Close',
                        title=f'Preço de Fechamento de {ticker_symbol}',
                        labels={'Close': 'Preço de Fechamento (R$)', 'Date': 'Data'}
                    )
                    # --- LINHA ADICIONADA PARA CUSTOMIZAR O TOOLTIP ---
                    fig_close.update_traces(hovertemplate='<b>Data:</b> %{x|%d/%m/%Y}<br><b>Preço:</b> R$ %{y:,.2f}<extra></extra>')

                    # Atualiza o layout do gráfico (títulos dos eixos)
                    fig_close.update_layout(xaxis_title='Data', yaxis_title='Preço (R$)')

                    # Exibe o gráfico no Streamlit
                    st.plotly_chart(fig_close, use_container_width=True)


                    st.subheader("Gráfico de Previsão vs. Real", divider='rainbow')
                    # A lógica para criar 'base_resultados' deve vir ANTES deste trecho.

                    # Cria uma figura vazia do Plotly
                    fig_previsao = go.Figure()

                    # Adiciona o traço para o 'Close_Real' com seu próprio hovertemplate
                    fig_previsao.add_trace(go.Scatter(
                        x=base_filtrada['Date'],
                        y=base_filtrada['Close'],
                        mode='lines',
                        name='Preço Real',
                        # Define o formato APENAS para o valor Y desta linha
                        hovertemplate='R$ %{y:,.2f}<extra></extra>'
                    ))

                    # Adiciona o traço para o 'Preço Estimado' com seu próprio hovertemplate
                    fig_previsao.add_trace(go.Scatter(
                        x=base_filtrada['Date'],
                        y=base_filtrada['Fitted'],
                        mode='lines',
                        name='Preço Estimado',
                        line=dict(color='orange', dash='dot'),
                        # Define o formato APENAS para o valor Y desta linha
                        hovertemplate='R$ %{y:,.2f}<extra></extra>'
                    ))

                    # Adiciona o traço para o 'Preço Previsto' com seu próprio hovertemplate
                    fig_previsao.add_trace(go.Scatter(
                        x=base_filtrada['Date'],
                        y=base_filtrada['Predict'],
                        mode='lines',
                        name='Preço Previsto',
                        line=dict(color='green', dash='dot'),
                        # Define o formato APENAS para o valor Y desta linha
                        hovertemplate='R$ %{y:,.2f}<extra></extra>'
                    ))


                    # Atualiza o layout do gráfico com as novas configurações de hover
                    fig_previsao.update_layout(
                        title="Previsão de Preço a partir de ARMA(9,10)-GARCH(2,1)",
                        xaxis_title="Data",
                        yaxis_title="Preço",
                        # --- MUDANÇAS PRINCIPAIS AQUI ---
                        hovermode='x unified', # Unifica o tooltip para o eixo X
                        xaxis_hoverformat='%d/%m/%Y' # Formata a data no topo do tooltip unificado
                    )

                    # Renderiza o gráfico de previsão no Streamlit
                    st.plotly_chart(fig_previsao, use_container_width=True)

            except Exception as e:
                st.error(f"Ocorreu um erro ao buscar os dados para {ticker_symbol}: {e}")