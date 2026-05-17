import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
import time
from data_store import get_ai_agent_stats, save_ai_agent_stats
from market_data import fetch_live_close

def render_ai_agent_dashboard():
    st.markdown("## 🤖 AI自動運用ダッシュボード")
    st.write("AIが1時間ごとに銘柄をスキャンし、自動で売買判断を行います。")

    # データの読み込みを先に行う
    agent_stats = get_ai_agent_stats()

    # 自動リロード機能（カウントダウンが0になったらリロード）
    if "last_run_time" not in st.session_state:
        st.session_state.last_run_time = agent_stats.get('last_run')

    # 状態が変わっていたらリロード
    if st.session_state.last_run_time != agent_stats.get('last_run'):
        st.session_state.last_run_time = agent_stats.get('last_run')
        st.rerun()

    # サイドバーに設定
    with st.sidebar:
        st.markdown("### ⚙️ AI運用の設定")
        if st.button("🚀 AIサイクルを強制実行", type="primary", use_container_width=True):
            from ai_agent_engine import run_ai_agent_cycle
            with st.spinner("AIが分析と売買を実行中..."):
                run_ai_agent_cycle()
            st.success("実行完了！")
            st.rerun()
        
        if st.button("♻️ 運用データをリセット", use_container_width=True):
            from data_store import reset_ai_agent_stats
            reset_ai_agent_stats()
            st.success("リセットしました。")
            st.rerun()

    # 上部：次回更新までのカウントダウン
    st.markdown("### ⏰ 次回更新まで")
    
    # スケジューラーの状態を確認
    last_run_str = agent_stats.get('last_run')
    if last_run_str:
        try:
            # "2026-05-17 11:29 UTC" 形式をパース
            last_run_dt = datetime.strptime(last_run_str, "%Y-%m-%d %H:%M %Z").replace(tzinfo=timezone.utc)
            next_run_dt = last_run_dt + timedelta(hours=1)
            now_dt = datetime.now(timezone.utc)
            remaining = next_run_dt - now_dt
            
            if remaining.total_seconds() > 0:
                minutes = int(remaining.total_seconds() // 60)
                seconds = int(remaining.total_seconds() % 60)
                
                # リアルタイムカウントダウン (JavaScript)
                unique_id = int(time.time())
                countdown_html = f"""
                <div id="countdown_{unique_id}" style="font-size: 28px; font-weight: bold; padding: 20px; background-color: #f0f7ff; border-radius: 15px; text-align: center; border: 3px solid #007bff; color: #0056b3; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 10px 0;">
                    ⏳ 残り {minutes}分 {seconds}秒
                </div>
                <script>
                (function() {{
                    const targetId = 'countdown_{unique_id}';
                    const nextRunTime = new Date('{next_run_dt.isoformat()}').getTime();
                    function update() {{
                        const now = new Date().getTime();
                        const diff = nextRunTime - now;
                        const el = document.getElementById(targetId);
                        if (!el) return;
                        if (diff > 0) {{
                            const m = Math.floor(diff / 60000);
                            const s = Math.floor((diff % 60000) / 1000);
                            el.innerHTML = "⏳ 残り " + m + "分 " + s + "秒";
                        }} else {{
                            el.innerHTML = '✅ 実行準備完了！';
                            el.style.backgroundColor = '#e8f5e9';
                            el.style.borderColor = '#4caf50';
                            el.style.color = '#1b5e20';
                            setTimeout(() => {{ window.location.reload(); }}, 3000);
                        }}
                    }}
                    setInterval(update, 1000);
                    update();
                }})();
                </script>
                """
                st.markdown(countdown_html, unsafe_allow_html=True)
            else:
                st.success("✅ 今すぐ実行可能です！")
        except Exception as e:
            st.info(f"最後の実行時刻を確認中... ({e})")
    else:
        st.success("✅ 初回実行可能です！")

    # メインエリア
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 1. 過去の予測成功率
        st.markdown("#### 📊 過去の予測成功率")
        from prediction_manager import validate_past_predictions
        stats = validate_past_predictions()
        
        if stats['total'] > 0:
            c1, c2 = st.columns(2)
            c1.metric("検証済み数", stats['validated_count'])
            c2.metric("総合成功率", f"{stats['success_rate']:.1f}%")
            
            with st.expander("詳細な履歴を表示"):
                if stats.get("history"):
                    st.dataframe(pd.DataFrame(stats["history"]), use_container_width=True, hide_index=True)
        else:
            st.info("まだ検証可能な過去の予測データがありません。運用を続けると自動的に集計されます。")

    with col2:
        # 2. AI運用ステータス
        st.markdown("#### 💰 AI運用ステータス")
        c1, c2, c3 = st.columns(3)
        c1.metric("現在の資産", f"¥{agent_stats['current_cash']:,.0f}")
        c2.metric("保有ポジション数", agent_stats['position_count'])
        c3.metric("リセット回数", agent_stats['reset_count'])

    # 3. 資産推移グラフ
    st.markdown("---")
    st.markdown("#### 📈 資産推移")
    portfolio_history = agent_stats.get('portfolio_history', [])
    if portfolio_history and len(portfolio_history) > 1:
        df_history = pd.DataFrame(portfolio_history)
        # 日付形式を柔軟に処理（ISO8601対応）
        df_history['date'] = pd.to_datetime(df_history['date'], errors='coerce')
        df_history = df_history.dropna(subset=['date'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_history['date'],
            y=df_history['total_value'],
            mode='lines+markers',
            name='総資産',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        fig.add_trace(go.Scatter(
            x=df_history['date'],
            y=df_history['cash'],
            mode='lines',
            name='現金',
            line=dict(color='#ff7f0e', width=1, dash='dot')
        ))
        fig.update_layout(
            height=350,
            margin=dict(t=20, b=20, l=0, r=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 累積リターン
        initial = agent_stats.get('initial_cash', 300000.0)
        current = agent_stats['current_cash'] + agent_stats.get('positions_value', 0)
        change_pct = (current - initial) / initial * 100
        st.metric("累積リターン", f"{change_pct:+.2f}%")
    else:
        st.info("資産推移データはまだ記録されていません。AIサイクルを実行してください。")

    # 4. ポートフォリオ構成
    st.markdown("---")
    st.markdown("#### 📊 ポートフォリオ構成")
    
    if agent_stats['positions']:
        # 円グラフ用のデータ作成
        labels = ['現金']
        values = [agent_stats['current_cash']]
        
        for ticker, pos in agent_stats['positions'].items():
            current_price = fetch_live_close(ticker) or pos['entry_price']
            labels.append(ticker)
            values.append(current_price * pos['shares'])
            
        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        fig_pie.update_layout(
            title="資産構成比",
            height=400,
            margin=dict(t=30, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("#### 📋 保有銘柄一覧")
    if agent_stats['positions']:
        pos_list = []
        for ticker, pos in agent_stats['positions'].items():
            # 最新価格を取得
            current_price = fetch_live_close(ticker) or pos['entry_price']
            profit = (current_price - pos['entry_price']) * pos['shares']
            profit_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100
            
            pos_list.append({
                "銘柄": ticker,
                "保有数": pos['shares'],
                "取得価格": pos['entry_price'],
                "現在価格": current_price,
                "評価額": current_price * pos['shares'],
                "損益": profit,
                "損益率": profit_pct
            })
        
        pos_df = pd.DataFrame(pos_list)
        display_df = pos_df.copy()
        display_df['取得価格'] = display_df['取得価格'].map('¥{:,.1f}'.format)
        display_df['現在価格'] = display_df['現在価格'].map('¥{:,.1f}'.format)
        display_df['評価額'] = display_df['評価額'].map('¥{:,.0f}'.format)
        display_df['損益'] = display_df.apply(lambda x: f"¥{x['損益']:+,.0f} ({x['損益率']:+.1f}%)", axis=1)
        st.dataframe(display_df[['銘柄', '保有数', '取得価格', '現在価格', '評価額', '損益']], use_container_width=True, hide_index=True)

        # 個別銘柄チャート
        st.markdown("#### 🔍 保有銘柄のパフォーマンス比較")
        
        # 複数銘柄の推移を一つのグラフに表示
        fig_comp = go.Figure()
        import yfinance as yf
        
        for _, row in pos_df.iterrows():
            t = row['銘柄']
            p = agent_stats['positions'][t]
            e_date = datetime.fromisoformat(p['entry_date'])
            # 購入日から現在までのデータを取得
            d_comp = yf.download(t, start=e_date.strftime('%Y-%m-%d'), progress=False)
            if not d_comp.empty:
                d_comp.columns = d_comp.columns.droplevel(1) if isinstance(d_comp.columns, pd.MultiIndex) else d_comp.columns
                # 取得価格を100とした指数を表示
                normalized_price = (d_comp['Close'] / p['entry_price']) * 100
                fig_comp.add_trace(go.Scatter(
                    x=d_comp.index, y=normalized_price,
                    mode='lines', name=f"{t} (取得時=100)"
                ))
        
        fig_comp.add_hline(y=100, line_dash="dash", line_color="black", opacity=0.5)
        fig_comp.update_layout(
            title="保有銘柄の相対パフォーマンス（取得価格を100として比較）",
            xaxis_title="日付",
            yaxis_title="パフォーマンス (%)",
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("#### 🔍 個別銘柄の詳細チャート")
        selected_ticker = st.selectbox("詳細を表示する銘柄を選択", options=pos_df['銘柄'].tolist())
        
        if selected_ticker:
            pos_info = agent_stats['positions'][selected_ticker]
            entry_date = datetime.fromisoformat(pos_info['entry_date'])
            
            # 取得日から現在までのチャートを表示
            df_ticker = yf.download(selected_ticker, start=(entry_date - timedelta(days=5)).strftime('%Y-%m-%d'), progress=False)
            if not df_ticker.empty:
                df_ticker.columns = df_ticker.columns.droplevel(1) if isinstance(df_ticker.columns, pd.MultiIndex) else df_ticker.columns
                fig_detail = go.Figure()
                fig_detail.add_trace(go.Scatter(x=df_ticker.index, y=df_ticker['Close'], name='株価', line=dict(color='#1f77b4')))
                
                # 購入地点をマーク
                fig_detail.add_trace(go.Scatter(
                    x=[entry_date], y=[pos_info['entry_price']],
                    mode='markers', name='購入地点',
                    marker=dict(size=15, color='gold', symbol='star', line=dict(width=2, color='darkorange'))
                ))
                # 購入価格の水平線
                fig_detail.add_hline(y=pos_info['entry_price'], line_dash="dot", line_color="orange", annotation_text="購入価格")
                
                fig_detail.update_layout(title=f"{selected_ticker} の値動き（購入時からの推移）", height=400, template='plotly_white')
                st.plotly_chart(fig_detail, use_container_width=True)
    else:
        st.info("現在保有している銘柄はありません。")

    # 5. 実行ログ
    st.markdown("---")
    st.markdown("#### 📜 AI実行ログ")
    if agent_stats.get('history'):
        log_df = pd.DataFrame(agent_stats['history'])
        st.dataframe(log_df[['time', 'action']], use_container_width=True, hide_index=True)
    else:
        st.info("実行ログはまだありません。")
