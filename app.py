import pandas as pd
import numpy as np
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
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, RNN,SimpleRNN, Dropout,LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import SwapEMAWeights,Callback
from tensorflow.keras.saving import save_model, load_model
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping

@st.cache_data
def carregar_dados(ticker, data_inicio, data_fim):
    """
    Busca dados históricos e adiciona a última cotação intraday, se for de um novo dia.
    """
    # 1. Cria o objeto Ticker para o ativo desejado
    require = yf.Ticker(ticker)
    name_ticker = require.info.get('longName', 'Unknown')

    # 2. Busca o histórico diário principal
    data = require.history(start=data_inicio, end=data_fim)

    # 3. Busca dados intraday bem recentes (últimos 2 dias, intervalo de 1 minuto)
    bd = require.history(period="2d", interval='1m')

    # 4. Compara as datas e concatena se necessário
    # Checa se o último dia do histórico é diferente do último dia do intraday
    if data.index[-1].date() != bd.index[-1].date():
        # Se forem diferentes, anexa a última cotação intraday ao histórico
        # Usamos pd.concat para garantir que os dataframes sejam unidos corretamente
        data = pd.concat([data, bd.tail(1)])
        
    return data, name_ticker

def modelo_garch(bd):
    # --- 1. Gerar dados simulados (ou você pode carregar seus dados reais) ---
    # Retorno simples
    returns = bd['Close'].pct_change().dropna()
    initial_price = bd['Close'].iloc[-1]

    # Definir ponto de corte (ex: últimos 30 dias para teste)
    test_size = 25
    split_date = returns.index[-test_size] # A data onde começa o teste

    # Base de Treino (apenas para aprender os parâmetros)
    train_returns = returns.loc[:split_date].iloc[:-1] # Tudo antes da data de corte
    test_returns = returns.loc[split_date:] # Tudo após a data de corte

    # --- 2. Ajustar o ARMA(9,10) para a média ---
    # Ajustando um ARMA(9,10) com statsmodels
    arma_order = (9, 0, 10)  # (p, d, q)
    arma_model = ARIMA(train_returns, order=arma_order)
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
    garch_model = mean_model.fit()

    # --- 5. Resultados ---
    print(garch_model.summary())

    # --- 6. Fazer previsões de teste ---
    # Média prevista dos retornos
    arima_forecast_test = arma_model.forecast(steps=test_size)
    mean_forecast_test = arima_forecast_test # Série com as previsões da média
    mean_forecast_test.index = test_returns.index

    forecast_test = garch_model.forecast(horizon=test_size)

    # Acesso às previsões
    variance_forecast_test = forecast_test.variance.values[-1]  # Variância prevista
    variance_forecast_test = pd.Series(
        variance_forecast_test,
        index=test_returns.index
    )

    # Desvio padrão previsto (sqrt da variância)
    std_forecast_test = np.sqrt(variance_forecast_test)
    std_forecast_test = pd.Series(
        std_forecast_test,
        index=test_returns.index
    )

    # Alinhar os índices (garantir que ambos têm as mesmas datas)
    common_index_test = mean_forecast_test.index.intersection(std_forecast_test.index)
    mean_forecast_test = mean_forecast_test.loc[common_index_test]
    variance_forecast_test = variance_forecast_test.loc[common_index_test]
    std_forecast_test = std_forecast_test.loc[common_index_test]

    # Gerar choques aleatórios baseados no desvio previsto
    # np.random.seed(42)  # Para reprodutibilidade
    random_shocks_test = np.random.normal(0, variance_forecast_test)

    # Retornos ajustados = média prevista + choque aleatório
    adjusted_returns_test = mean_forecast_test + std_forecast_test * random_shocks_test

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

    # --- 7. Fazer previsões futuras (10 dias) ---

    future_horizon = 10

    # O método .apply() pega os parâmetros (phi, theta) aprendidos no treino
    # e os aplica aos dados novos sem reestimar (muito rápido).
    arma_model_full = arma_model.apply(returns)

    params_garch = garch_model.params

    # Atualizar GARCH (.fix)
    # Criamos um modelo novo com a base CHEIA, mas fixamos os parâmetros do treino
    garch_model_full = ConstantMean(returns)
    garch_model_full.volatility = GARCH(p=1, q=1)
    garch_model_full.distribution = Normal()
    garch_model_full = garch_model_full.fix(params_garch)

    dates = pd.to_datetime(bd.index)
    forecast_dates = pd.date_range(list(dates)[-1]+pd.DateOffset(1), periods=future_horizon,freq='b').tolist()

    # Acesso às previsões
    arma_pred_obj_full = arma_model_full.forecast(steps=future_horizon)
    mean_forecast_full = arma_pred_obj_full # Série com as previsões da média
    mean_forecast_full.index = forecast_dates

    forecast_full = garch_model_full.forecast(horizon=future_horizon)
    variance_forecast_full = forecast_full.variance.values[-1]
    variance_forecast_full = pd.Series(
        variance_forecast_full,
        index=forecast_dates
    )
    std_forecast_full = np.sqrt(variance_forecast_full)
    std_forecast_full = pd.Series(
        std_forecast_full,
        index=forecast_dates
    )

    # Alinhar os índices (garantir que ambos têm as mesmas datas)
    common_index_full = mean_forecast_full.index.intersection(std_forecast_full.index)
    mean_forecast_full = mean_forecast_full.loc[common_index_full]
    variance_forecast_full = variance_forecast_full.loc[common_index_full]
    std_forecast_full = std_forecast_full.loc[common_index_full]

    # Gerar choques aleatórios baseados no desvio previsto
    random_shocks_full = np.random.normal(0, variance_forecast_full)

    # Retornos ajustados = média prevista + choque aleatório
    adjusted_returns_full = mean_forecast_full + std_forecast_full * random_shocks_full

    # --- 8. Conversão inversa: De retorno para preço ---

    # Convertendo retornos previstos em preços (assumindo preço inicial)
    # Pegar preço anterior ao início do teste para basear a primeira previsão
    base_price_test = bd['Close'].loc[:split_date].iloc[-2] 
    prices_test = [base_price_test]
    for r in adjusted_returns_test:
        next_price = prices_test[-1] * (1 + r)
        prices_test.append(next_price)

    prices_test = np.array(prices_test[1:])  # Remover o preço inicial

    # Convertendo retornos previstos em preços (assumindo preço inicial)
    # Pegar preço anterior ao início do teste para basear a primeira previsão
    base_price_future = bd['Close'].iloc[-1] 
    prices_future = [base_price_future]
    for r in adjusted_returns_full:
        next_price = prices_future[-1] * (1 + r)
        prices_future.append(next_price)

    prices_future = np.array(prices_future[1:])  # Remover o preço inicial


    # --- Montar o DataFrame organizado ---

    base_resultados = pd.DataFrame({'Date': bd.loc[:split_date].index,
                        'Close': bd['Close'].loc[:split_date],
                        'Open': bd['Open'].loc[:split_date],
                        'High': bd['High'].loc[:split_date],
                        'Low': bd['Low'].loc[:split_date],
                        'Fitted':  np.full(len(bd.loc[:split_date].index), np.nan),
                        'Predict':  np.full(len(bd.loc[:split_date].index), np.nan)}).iloc[:-1]


    base_test = pd.DataFrame({'Date': bd.loc[common_index_test].index,
                        'Close': bd['Close'].loc[common_index_test],
                        'Open': bd['Open'].loc[common_index_test],
                        'High': bd['High'].loc[common_index_test],
                        'Low': bd['Low'].loc[common_index_test],
                        'Fitted':  prices_test,
                        'Predict':  np.full(len(prices_test), np.nan)})

    df_predict = pd.DataFrame({'Date': np.array(forecast_dates),
                            'Close': np.full(len(forecast_dates), np.nan),
                            'Open':  np.full(len(forecast_dates), np.nan),
                            'High':  np.full(len(forecast_dates), np.nan),
                            'Low':  np.full(len(forecast_dates), np.nan),
                            'Fitted':  np.full(len(forecast_dates), np.nan),
                            'Predict': np.array(prices_future)}, index = forecast_dates)

    bd_final = pd.concat([base_resultados, base_test, df_predict], axis=0)

    bd_final['Date'] = pd.to_datetime(bd_final['Date'], format='%Y-%m-%d')

    print("Fim Do Processo do Modelo GARCH")

    return bd_final

def modelo_lstm(bd):
    me = bd['Close'].mean()
    amp = bd['Close'].max() - bd['Close'].min()
    df_scaled = ((bd['Close'] - me)/amp).to_numpy().reshape(-1, 1)


    def create_df(df,steps=1):
        dataX, dataY=[], []
        for i in range(len(df)-steps):
            a = df[i:(i+steps),0]
            dataX.append(a)
            dataY.append(df[i+steps,0])
        return np.array(dataX), np.array(dataY)


    steps = 15
    X, Y = create_df(df_scaled,steps)
    test_size = 25
    train_size = len(X) - test_size
    Xtrain = X[0:train_size]
    Ytrain = Y[0:train_size]
    Xtest = X[train_size:]
    Ytest = Y[train_size:]

    # Organizar dados de treino e teste
    Xtrain = Xtrain.reshape(Xtrain.shape[0],Xtrain.shape[1],1)
    Xtest = Xtest.reshape(Xtest.shape[0],Xtest.shape[1],1)
    X = X.reshape(X.shape[0],X.shape[1],1)

    # 1. Criar um modelo
    model = Sequential()
    model.add(LSTM(128, activation=LeakyReLU(alpha=0.4), return_sequences=True,input_shape=(steps,1)))
    model.add(Dropout(0.03))
    model.add(LSTM(64, activation=LeakyReLU(alpha=0.3),return_sequences=True))
    model.add(Dropout(0.02))
    model.add(LSTM(32, activation=LeakyReLU(alpha=0.2),return_sequences=False))
    model.add(Dropout(0.01))
    model.add(Dense(1,activation=LeakyReLU(alpha=0.1)))

    model.compile(optimizer='adam',loss='mse')

    # 2. Defina o callback com o argumento restore_best_weights=True
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',       # O que ele está vigiando
        patience=25,              # Quantas épocas esperar
        restore_best_weights=True # <--- O SEGREDO ESTÁ AQUI
    )

    # 3. Treinar o modelo com o callback
    validation = model.fit(
        Xtrain, Ytrain,
        validation_data=(Xtest, Ytest),
        epochs=100,
        batch_size=16,
        callbacks=[early_stopping_callback]
    )

    input_steps_test = Xtest[0]
    input_steps_test = np.array(input_steps_test).reshape(1,-1)

    list_output_steps_test = list(input_steps_test)
    list_output_steps_test = list_output_steps_test[0].tolist()

    pred_test = []
    i=0
    n_future=len(Ytest)
    while(i<n_future):
        if(len(list_output_steps_test)>steps):
            input_steps = np.array(list_output_steps_test[1:])
            input_steps = input_steps.reshape(1,-1)
            input_steps = input_steps.reshape((1,steps,1))
            pred = model.predict(input_steps,verbose=0)
            list_output_steps_test.extend(pred[0].tolist())
            list_output_steps_test=list_output_steps_test[1:]
            pred_test.extend(pred.tolist())
            i=i+1
        else:
            input_steps = input_steps_test.reshape((1,steps,1))
            pred = model.predict(input_steps,verbose=0)
            list_output_steps_test.extend(pred[0].tolist())
            pred_test.extend(pred.tolist())
            i=i+1

    pred_test_np = np.array(pred_test)
    pred_test_np = pred_test_np.reshape(-1)
    prev_test = pred_test_np * amp + me
    prev_test = np.array(prev_test).reshape(1,-1)
    list_prev_test = prev_test[0].tolist()

    input_steps_future = df_scaled[-steps:]
    input_steps_future = np.array(input_steps_future).reshape(1,-1)

    list_output_steps_future = list(input_steps_future)
    list_output_steps_future = list_output_steps_future[0].tolist()

    pred_future = []
    i=0
    n_future=10
    while(i<n_future):
        if(len(list_output_steps_future)>steps):
            input_steps = np.array(list_output_steps_future[1:])
            input_steps = input_steps.reshape(1,-1)
            input_steps = input_steps.reshape((1,steps,1))
            pred = model.predict(input_steps,verbose=0)
            list_output_steps_future.extend(pred[0].tolist())
            list_output_steps_future=list_output_steps_future[1:]
            pred_future.extend(pred.tolist())
            i=i+1
        else:
            input_steps = input_steps_future.reshape((1,steps,1))
            pred = model.predict(input_steps,verbose=0)
            list_output_steps_future.extend(pred[0].tolist())
            pred_future.extend(pred.tolist())
            i=i+1

    pred_future_np = np.array(pred_future)
    pred_future_np = pred_future_np.reshape(-1)
    prev_future = pred_future_np * amp + me
    prev_future = np.array(prev_future).reshape(1,-1)
    list_prev_future = prev_future[0].tolist()    
    # --- Montar o DataFrame organizado ---

    dates = pd.to_datetime(bd.index)
    forecast_dates = pd.date_range(list(dates)[-1]+pd.DateOffset(1), periods=n_future,freq='b').tolist()

    base_resultados = pd.DataFrame({'Date': bd.loc[:split_date].index,
                        'Close': bd['Close'].loc[:split_date],
                        'Open': bd['Open'].loc[:split_date],
                        'High': bd['High'].loc[:split_date],
                        'Low': bd['Low'].loc[:split_date],
                        'Fitted':  np.full(len(bd.loc[:split_date].index), np.nan),
                        'Predict':  np.full(len(bd.loc[:split_date].index), np.nan)}).iloc[:-1]


    base_test = pd.DataFrame({'Date': bd[-test_size:].index,
                        'Close': bd[-test_size:]['Close'],
                        'Open': bd[-test_size:]['Open'],
                        'High': bd[-test_size:]['High'],
                        'Low': bd[-test_size:]['Low'],
                        'Fitted':  list_prev_test,
                        'Predict':  np.full(len(list_prev_test), np.nan)})

    df_predict = pd.DataFrame({'Date': np.array(forecast_dates),
                            'Close': np.full(len(forecast_dates), np.nan),
                            'Open':  np.full(len(forecast_dates), np.nan),
                            'High':  np.full(len(forecast_dates), np.nan),
                            'Low':  np.full(len(forecast_dates), np.nan),
                            'Fitted':  np.full(len(forecast_dates), np.nan),
                            'Predict': np.array(list_prev_future)}, index = forecast_dates)

    bd_final = pd.concat([base_resultados, base_test, df_predict], axis=0)

    bd_final['Date'] = pd.to_datetime(bd_final['Date'], format='%Y-%m-%d')

    print("Fim Do Processo do Modelo LSTM")

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
            data, name_ticker = carregar_dados(ticker_symbol, start_date, end_date)
            
        # Validação dos dados
        if data.empty:
            st.error(f"Nenhum dado encontrado para o ticker '{ticker_symbol}'. Verifique o código da ação (ex: PETR4.SA, MGLU3.SA, AAPL).")
        else:
            st.success(f"Dados de {ticker_symbol} carregados com sucesso!")
            # --- Exibição dos Dados ---
            st.header(f"Dados Históricos para {name_ticker}", divider='rainbow')

            # Exibe o dataframe com os dados brutos
            st.dataframe(data.sort_index(ascending=False), use_container_width=True)

            # --- Visualização com Gráficos (Plotly) ---
            st.subheader("Gráfico de Preço de Fechamento", divider='rainbow')

            # Cria uma figura vazia do Plotly
            fig_close = go.Figure()

            # Adiciona o traço para o 'Close_Real' com seu próprio hovertemplate
            fig_close.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Preço Real',
                # Define o formato APENAS para o valor Y desta linha
                hovertemplate='R$ %{y:,.2f}<extra></extra>'
            ))

            # Atualiza o layout do gráfico (títulos dos eixos)
            fig_close.update_layout(
                xaxis_title='Data', 
                yaxis_title='Preço (R$)',
                hovermode='x unified', # Unifica o tooltip para o eixo X
                xaxis_hoverformat='%d/%m/%Y' # Formata a data no topo do tooltip unificado
            )

            # Exibe o gráfico no Streamlit
            st.plotly_chart(fig_close, use_container_width=True)

            with st.spinner(f"Treinando modelo GARCH..."):
                try:
                    base_garch = modelo_garch(data)
                    base_garch.sort_values(by='Date', ascending=True, inplace=True)
                    df_limpo = base_garch[['Date','Close', 'Fitted']].dropna()
                    mape_garch = mean_absolute_percentage_error(df_limpo['Close'], df_limpo['Fitted']) if not df_limpo.empty else 0
                    base_filtrada_garch = base_garch.tail(95)
                    # Definir ponto de corte (ex: últimos 25 dias para teste)
                    test_size = 35
                    split_date = base_garch.index[-test_size] # A data onde começa o teste
                    base_treino_garch = base_filtrada_garch.loc[:split_date].iloc[:-1]
                    base_teste_garch = base_filtrada_garch.loc[split_date:]

                    st.subheader(f"Gráfico de Previsão vs. Real - GARCH - Accuracy {(1-mape_garch):.2%}", divider='rainbow')
                    # A lógica para criar 'base_resultados' deve vir ANTES deste trecho.

                    fig_previsao_garch = go.Figure()

                    # ================= HISTÓRICO / TREINO =================
                    fig_previsao_garch.add_trace(go.Scatter(
                        x=base_treino_garch['Date'],
                        y=base_treino_garch['Close'],
                        mode='lines',
                        name='Treino (Histórico)',
                        line=dict(color='gray', width=2),
                        opacity=0.5,
                        hovertemplate='R$ %{y:,.2f}<extra></extra>'
                    ))

                    # ================= REAL (TESTE) =================
                    fig_previsao_garch.add_trace(go.Scatter(
                        x=base_teste_garch['Date'],
                        y=base_teste_garch['Close'],
                        mode='lines',
                        name='Real (Teste)',
                        line=dict(color='blue', width=3),
                        hovertemplate='R$ %{y:,.2f}<extra></extra>'
                    ))

                    fig_previsao_garch.add_trace(go.Scatter(
                        x=base_teste_garch['Date'],
                        y=base_teste_garch['Fitted'],
                        mode='lines+markers',
                        name='Previsto (Teste)',
                        line=dict(color='red', width=3),
                        hovertemplate='R$ %{y:,.2f}<extra></extra>'
                    ))

                    # ================= PREVISTO (MODELO) =================
                    fig_previsao_garch.add_trace(go.Scatter(
                        x=base_filtrada_garch['Date'],
                        y=base_filtrada_garch['Predict'],
                        mode='lines+markers',
                        name='Previsto (Futuro)',
                        line=dict(color='green', width=2, dash='dash'),
                        marker=dict(size=6),
                        hovertemplate='R$ %{y:,.2f}<extra></extra>'
                    ))

                    # ================= LAYOUT =================
                    fig_previsao_garch.update_layout(
                        # title=dict(
                        #     text="Backtesting: Treino | Teste | Previsão",
                        #     x=0.5,
                        #     xanchor='center'
                        # ),
                        xaxis_title="Data",
                        yaxis_title="Preço",
                        hovermode='x unified',
                        xaxis_hoverformat='%d/%m/%Y',

                        # Grid parecido com matplotlib
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(0,0,0,0.1)'
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(0,0,0,0.1)'
                        ),

                        # Fundo branco estilo plt
                        # plot_bgcolor='white',
                        # paper_bgcolor='white',

                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=1.02,
                            xanchor='center',
                            x=0.5
                        )
                    )

                    st.plotly_chart(fig_previsao_garch, use_container_width=True)

                except Exception as e:
                    st.error(f"Ocorreu um erro ao treinar o modelo GARCH: {e}")
            
            with st.spinner(f"Treinando o Modelo LSTM..."):
                try:
                    base_lstm = modelo_lstm(data)
                    base_lstm.sort_values(by='Date', ascending=True, inplace=True)
                    df_limpo_lstm = base_lstm[['Date','Close', 'Fitted']].dropna()
                    mape_lstm = mean_absolute_percentage_error(df_limpo_lstm['Close'], df_limpo_lstm['Fitted']) if not df_limpo_lstm.empty else 0
                    base_filtrada_lstm = base_lstm.tail(95)
                    # Definir ponto de corte (ex: últimos 25 dias para teste)
                    test_size = 35
                    split_date = base_lstm.index[-test_size] # A data onde começa o teste
                    base_treino_lstm = base_filtrada_lstm.loc[:split_date].iloc[:-1]
                    base_teste_lstm = base_filtrada_lstm.loc[split_date:]

                    st.subheader(f"Gráfico de Previsão vs. Real - Rede LSTM - Accuracy {(1-mape_lstm):.2%}", divider='rainbow')
                    # A lógica para criar 'base_resultados' deve vir ANTES deste trecho.

                    fig_previsao_lstm = go.Figure()

                    # ================= HISTÓRICO / TREINO =================
                    fig_previsao_lstm.add_trace(go.Scatter(
                        x=base_treino_lstm['Date'],
                        y=base_treino_lstm['Close'],
                        mode='lines',
                        name='Treino (Histórico)',
                        line=dict(color='gray', width=2),
                        opacity=0.5,
                        hovertemplate='R$ %{y:,.2f}<extra></extra>'
                    ))

                    # ================= REAL (TESTE) =================
                    fig_previsao_lstm.add_trace(go.Scatter(
                        x=base_teste_lstm['Date'],
                        y=base_teste_lstm['Close'],
                        mode='lines',
                        name='Real (Teste)',
                        line=dict(color='blue', width=3),
                        hovertemplate='R$ %{y:,.2f}<extra></extra>'
                    ))

                    fig_previsao_lstm.add_trace(go.Scatter(
                        x=base_teste_lstm['Date'],
                        y=base_teste_lstm['Fitted'],
                        mode='lines+markers',
                        name='Previsto (Teste)',
                        line=dict(color='red', width=3),
                        hovertemplate='R$ %{y:,.2f}<extra></extra>'
                    ))

                    # ================= PREVISTO (MODELO) =================
                    fig_previsao_lstm.add_trace(go.Scatter(
                        x=base_filtrada_lstm['Date'],
                        y=base_filtrada_lstm['Predict'],
                        mode='lines+markers',
                        name='Previsto (Futuro)',
                        line=dict(color='green', width=2, dash='dash'),
                        marker=dict(size=6),
                        hovertemplate='R$ %{y:,.2f}<extra></extra>'
                    ))

                    # ================= LAYOUT =================
                    fig_previsao_lstm.update_layout(
                        # title=dict(
                        #     text="Backtesting: Treino | Teste | Previsão",
                        #     x=0.5,
                        #     xanchor='center'
                        # ),
                        xaxis_title="Data",
                        yaxis_title="Preço",
                        hovermode='x unified',
                        xaxis_hoverformat='%d/%m/%Y',

                        # Grid parecido com matplotlib
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(0,0,0,0.1)'
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(0,0,0,0.1)'
                        ),

                        # Fundo branco estilo plt
                        # plot_bgcolor='white',
                        # paper_bgcolor='white',

                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=1.02,
                            xanchor='center',
                            x=0.5
                        )
                    )

                    st.plotly_chart(fig_previsao_lstm, use_container_width=True)

                except Exception as e:
                    st.error(f"Ocorreu um erro ao treinar o modelo LSTM: {e}")