import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.title('株価分析ツール')

st.sidebar.header('設定')
ticker_input = st.sidebar.text_input('銘柄コード (例: AAPL や 7203.T):', 'AAPL')
period = st.sidebar.radio('予測期間:', ('1日', '1週間', '2週間', '1か月'))

ticker = ticker_input.strip()
if ticker.isdigit():
    ticker += '.T'

if ticker:
    try:
        # データ取得
        data = yf.download(ticker, period='60d', interval='1d')
        
        # 【修正ポイント】ここでデータを確認し、['Close']を確実に数値として取り出す
        if not data.empty:
            # カラムがマルチインデックスの場合の対策
            if isinstance(data.columns, tuple) or isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # データがSeriesかDataFrameか確認し、数値変換
            close_price = data['Close']
            if isinstance(close_price, (pd.DataFrame, pd.Series)):
                close_price = close_price.squeeze() # DataFrameならSeriesに変換
            
            # 期間に応じた移動平均計算
            window_map = {'1日': 1, '1週間': 5, '2週間': 10, '1か月': 20}
            window = window_map[period]
            sma = close_price.rolling(window=window).mean()

            # チャート作成
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=close_price, name='株価', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=data.index, y=sma, name='移動平均線', line=dict(color='red', dash='dot')))
            
            fig.update_layout(title=f'{ticker} の株価チャート', xaxis_title='日付', yaxis_title='価格')
            
            st.plotly_chart(fig, width='stretch')
            
            # 最新の値を表示（Seriesから値を取り出す）
            st.write(f"直近の終値: {float(close_price.iloc[-1]):.2f}")
        else:
            st.error(f"'{ticker}' のデータが見つかりませんでした。")
    except Exception as e:
        st.error(f"エラー内容: {e}")

st.markdown("---")
st.markdown("**※本ツールは参考情報を提供するものであり、投資は自己責任で行ってください。**")

# import文に足りていたか念のため追加
import pandas as pd
