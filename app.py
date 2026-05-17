import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.title('株価分析ツール')

# サイドバー設定
st.sidebar.header('設定')
ticker_input = st.sidebar.text_input('銘柄コード (例: AAPL や 7203.T):', 'AAPL')
period = st.sidebar.radio('予測期間:', ('1日', '1週間', '2週間', '1か月'))

# 日本株コード（数字4桁のみ）なら .T を自動付与するロジック
ticker = ticker_input.strip()
if ticker.isdigit():
    ticker += '.T'

if ticker:
    try:
        # データ取得
        data = yf.download(ticker, period='60d', interval='1d')
        
        if not data.empty:
            # 期間に応じた移動平均計算
            window_map = {'1日': 1, '1週間': 5, '2週間': 10, '1か月': 20}
            window = window_map[period]
            data['SMA'] = data['Close'].rolling(window=window).mean()

            # チャート作成
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='株価', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA'], name='移動平均線', line=dict(color='red', dash='dot')))
            
            fig.update_layout(title=f'{ticker} の株価チャート', xaxis_title='日付', yaxis_title='価格')
            
            # ログに出ていた修正案を適用（width='stretch'）
            st.plotly_chart(fig, width='stretch')
            
            st.write(f"直近の終値: {data['Close'].iloc[-1]:.2f}")
        else:
            st.error(f"'{ticker}' のデータが見つかりませんでした。正しいコードを入力してください。")
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

st.markdown("---")
st.markdown("**※本ツールは参考情報を提供するものであり、投資は自己責任で行ってください。**")
