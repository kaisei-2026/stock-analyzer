import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# ヘッダー
st.title('株価分析ツール')

# サイドバー
st.sidebar.header('設定')
ticker = st.sidebar.text_input('銘柄コード（ティッカー）:', 'AAPL')
period = st.sidebar.radio(
    '予測期間:',
    ('1日', '1週間', '2週間', '1か月')
)

# データ取得
if ticker:
    data = yf.download(ticker, period='60d', interval='1d')

    # 簡易予測計算
    if period == '1日':
        window = 1
    elif period == '1週間':
        window = 5
    elif period == '2週間':
        window = 10
    else:  # 1か月の場合
        window = 20

    data['SMA'] = data['Close'].rolling(window=window).mean()

    # 株価チャートの表示
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='株価'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA'], mode='lines', name='簡易予測ライン'))

    # チャートを画面に表示
    st.plotly_chart(fig)

# 免責
st.markdown("**※本ツールは参考情報を提供するものであり、投資は自己責任で行ってください。**")
