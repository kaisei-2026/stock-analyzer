
from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from ai_agent_engine import run_ai_agent_cycle, get_ai_agent_stats, get_candidate_stocks
from prediction_manager import validate_past_predictions
from data_store import reset_ai_agent, list_watch_tickers, add_watch_ticker, delete_watch_ticker

def render_ai_agent_dashboard() -> None:
    st.markdown("## 🤖 AI自動運用ダッシュボード")
    st.caption("AIが自動で銘柄を探し、30万円の仮想資金を運用します。APIリミッターにより実行頻度は制限されています。")

    # サイドバー: 監視銘柄設定
    with st.sidebar:
        st.markdown("### ⚙️ 監視銘柄の設定")
        st.write("AIが監視する銘柄を追加・管理します。")
        
        with st.form("add_watch_ticker_form"):
            ticker_input = st.text_input(
                "銘柄コードを入力",
                placeholder="例: 7203 (トヨタ) または 7203.T",
                key="watch_ticker_input"
            )
            name_input = st.text_input(
                "銘柄名（オプション）",
                placeholder="例: トヨタ",
                key="watch_name_input"
            )
            notes_input = st.text_area(
                "メモ（オプション）",
                placeholder="例: 配当利回りが高い、長期保有予定",
                height=60,
                key="watch_notes_input"
            )
            
            if st.form_submit_button("➕ 監視銘柄に追加", use_container_width=True):
                if ticker_input.strip():
                    result = add_watch_ticker(ticker_input.strip(), name_input.strip(), notes_input.strip())
                    if result["ok"]:
                        st.success(result["message"])
                        st.rerun()
                    else:
                        st.error(result["message"])
                else:
                    st.warning("銘柄コードを入力してください。")
        
        st.markdown("#### 📋 登録済み監視銘柄")
        watch_list = list_watch_tickers()
        if watch_list:
            for ticker_data in watch_list:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{ticker_data.get('name', ticker_data['ticker'])}** ({ticker_data['ticker']})")
                    if ticker_data.get('notes'):
                        st.caption(ticker_data['notes'])
                with col2:
                    if st.button("🗑️", key=f"del_{ticker_data['ticker']}", help="削除"):
                        delete_watch_ticker(ticker_data['ticker'])
                        st.rerun()
        else:
            st.info("監視銘柄がまだ登録されていません。")

    # データの読み込みを先に行う
    agent_stats = get_ai_agent_stats()

    # 自動リロード機能（カウントダウンが0になったらリロード）
    if "last_run_time" not in st.session_state:
        st.session_state.last_run_time = agent_stats.get('last_run')

    # 状態が変わっていたらリロード
    if st.session_state.last_run_time != agent_stats.get('last_run'):
        st.session_state.last_run_time = agent_stats.get('last_run')
        st.rerun()

    # メインエリア
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 1. 過去の予測成功率
        st.markdown("#### 📊 過去の予測成功率")
        stats = validate_past_predictions()
        if stats and stats.get("total") and stats["total"] > 0:
            accuracy = (stats["correct"] / stats["total"]) * 100
            col_acc, col_total = st.columns(2)
            col_acc.metric("成功率", f"{accuracy:.1f}%")
            col_total.metric("検証済み予測数", f"{stats['total']}件")
            
            if stats.get("history"):
                fig = go.Figure(data=[
                    go.Bar(x=list(range(len(stats["history"]))), y=[1 if h["is_correct"] else 0 for h in stats["history"]], marker_color=['green' if h["is_correct"] else 'red' for h in stats["history"]])
                ])
                fig.update_layout(height=200, margin=dict(t=0, b=0, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)
                
                if stats.get("history"):
                    st.markdown("**直近の検証履歴**")
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
        
        # 次回更新までの時間（確実に表示されるように修正）
        st.markdown("#### ⏰ 次回更新まで")
        last_run = agent_stats.get('last_run')
        
        if last_run:
            try:
                last_run_dt = datetime.fromisoformat(last_run.replace(' UTC', '+00:00'))
                next_run_dt = last_run_dt + timedelta(minutes=60)
                now_dt = datetime.now(last_run_dt.tzinfo)
                remaining = next_run_dt - now_dt
                
                if remaining.total_seconds() > 0:
                    minutes = int(remaining.total_seconds() // 60)
                    seconds = int(remaining.total_seconds() % 60)
                    st.warning(f"⏳ **{minutes}分 {seconds}秒後** に次のサイクルが自動実行されます。")
                    
                    # リアルタイムカウントダウン（JavaScriptが動かない場合の予備としてテキストも表示）
                    countdown_html = f"""
                    <div id="countdown_v2" style="font-size: 20px; font-weight: bold; padding: 15px; background-color: #e3f2fd; border-radius: 10px; text-align: center; border: 2px solid #2196f3; color: #0d47a1;">
                        残り {minutes}分 {seconds}秒
                    </div>
                    <script>
                    (function() {{
                        const nextRunTime = new Date('{next_run_dt.isoformat()}').getTime();
                        const update = () => {{
                            const now = new Date().getTime();
                            const diff = nextRunTime - now;
                            const el = document.getElementById('countdown_v2');
                            if (!el) return;
                            if (diff > 0) {{
                                const m = Math.floor(diff / 60000);
                                const s = Math.floor((diff % 60000) / 1000);
                                el.innerHTML = `⏳ 残り ${{m}}分 ${{s}}秒`;
                            }} else {{
                                el.innerHTML = '✅ 実行準備完了！';
                                el.style.backgroundColor = '#e8f5e9';
                                el.style.borderColor = '#4caf50';
                                el.style.color = '#1b5e20';
                                // 実行準備完了から少し待ってリロード（バックグラウンド実行を待つ）
                                setTimeout(() => {{ window.location.reload(); }}, 5000);
                            }}
                        }};
                        update();
                        setInterval(update, 1000);
                    }})();
                    </script>
                    """
                    st.markdown(countdown_html, unsafe_allow_html=True)
                else:
                    st.success("✅ 今すぐ実行可能です！")
            except Exception as e:
                st.info("最後の実行時刻を確認中...")
        else:
            st.success("✅ 初回実行可能です！")

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
            line=dict(color='#ff7f0e', width=1, dash='dash'),
            marker=dict(size=3)
        ))
        fig.add_trace(go.Scatter(
            x=df_history['date'],
            y=df_history['positions_value'],
            mode='lines',
            name='ポジション評価額',
            line=dict(color='#2ca02c', width=1, dash='dot'),
            marker=dict(size=3)
        ))
        
        fig.update_layout(
            title="AI自動運用の資産推移",
            xaxis_title="日時",
            yaxis_title="金額（円）",
            hovermode='x unified',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 統計情報
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            max_value = df_history['total_value'].max()
            st.metric("最大資産", f"¥{max_value:,.0f}")
        with stat_col2:
            min_value = df_history['total_value'].min()
            st.metric("最小資産", f"¥{min_value:,.0f}")
        with stat_col3:
            current_value = df_history['total_value'].iloc[-1]
            initial_value = df_history['total_value'].iloc[0]
            change_pct = ((current_value - initial_value) / initial_value * 100) if initial_value > 0 else 0
            st.metric("累積リターン", f"{change_pct:+.2f}%")
    else:
        st.info("資産推移データはまだ記録されていません。AIサイクルを実行してください。")

    # 4. ポートフォリオ
    st.markdown("---")
    st.markdown("#### 📊 ポートフォリオ構成")
    
    if agent_stats['positions']:
        # 円グラフ用のデータ作成
        labels = ['現金']
        values = [agent_stats['current_cash']]
        
        for ticker, pos in agent_stats['positions'].items():
            from market_data import fetch_live_close
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
            from market_data import fetch_live_close
            current_price = fetch_live_close(ticker) or pos['entry_price']
            
            profit = (current_price - pos['entry_price']) * pos['shares']
            profit_pct = ((current_price - pos['entry_price']) / pos['entry_price'] * 100) if pos['entry_price'] > 0 else 0
            pos_list.append({
                '銘柄': ticker,
                '保有数': pos['shares'],
                '取得価格': pos['entry_price'],
                '現在価格': current_price,
                '評価額': current_price * pos['shares'],
                '損益': profit,
                '損益率': profit_pct
            })
        
        pos_df = pd.DataFrame(pos_list)
        # 表示用にフォーマット
        display_df = pos_df.copy()
        display_df['取得価格'] = display_df['取得価格'].map('¥{:,.0f}'.format)
        display_df['現在価格'] = display_df['現在価格'].map('¥{:,.0f}'.format)
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
            import yfinance as yf
            pos_info = agent_stats['positions'][selected_ticker]
            entry_date = datetime.fromisoformat(pos_info['entry_date'])
            
            # 購入日の前日から現在までのデータを取得
            start_date = (entry_date - timedelta(days=5)).strftime('%Y-%m-%d')
            df_ticker = yf.download(selected_ticker, start=start_date, progress=False)
            
            if not df_ticker.empty:
                df_ticker.columns = df_ticker.columns.droplevel(1) if isinstance(df_ticker.columns, pd.MultiIndex) else df_ticker.columns
                
                fig_ticker = go.Figure()
                # 株価チャート
                fig_ticker.add_trace(go.Scatter(
                    x=df_ticker.index, y=df_ticker['Close'],
                    mode='lines', name='株価', line=dict(color='#1f77b4')
                ))
                # 購入ライン
                fig_ticker.add_hline(
                    y=pos_info['entry_price'], 
                    line_dash="dash", line_color="red",
                    annotation_text=f"購入価格: ¥{pos_info['entry_price']:,.0f}",
                    annotation_position="top left"
                )
                # 購入点
                fig_ticker.add_trace(go.Scatter(
                    x=[entry_date], y=[pos_info['entry_price']],
                    mode='markers', name='購入地点',
                    marker=dict(color='red', size=12, symbol='star')
                ))
                
                fig_ticker.update_layout(
                    title=f"{selected_ticker} の値動き（購入日: {entry_date.strftime('%Y-%m-%d')}）",
                    xaxis_title="日付",
                    yaxis_title="株価（円）",
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig_ticker, use_container_width=True)
            else:
                st.warning("チャートデータの取得に失敗しました。")
    else:
        st.info("現在、保有ポジションはありません。")

    # 5. 購入検討リスト
    st.markdown("---")
    st.markdown("#### 🎯 購入検討中の銘柄")
    candidate_stocks = get_candidate_stocks()
    if candidate_stocks:
        candidate_list = []
        for stock in candidate_stocks:
            candidate_list.append({
                '銘柄': stock['ticker'],
                '銘柄名': stock.get('name', ''),
                '買いスコア': f"{stock['buy_score']:.1f}点",
                '現在価格': f"¥{stock['current_price']:,.0f}",
                '推奨理由': stock.get('reason', ''),
            })
        cand_df = pd.DataFrame(candidate_list)
        st.dataframe(cand_df, use_container_width=True, hide_index=True)
    else:
        st.info("現在、購入検討中の銘柄はありません。")

    # 6. AI 操作パネル（目立つように配置）
    st.markdown("---")
    st.markdown("#### 🎮 AI 操作パネル")
    
    # 状態に関わらずボタンを確実に表示
    last_run = agent_stats.get('last_run')
    is_first_run = not last_run
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        btn_label = "🚀 AIサイクルを強制実行（初回リミッター解除）" if is_first_run else "🤖 AIサイクルを手動実行"
        if st.button(btn_label, type="primary", use_container_width=True, key="force_run_btn"):
            with st.spinner("AIが最新データを分析中..."):
                # 初回または前回の実行から時間が経過している場合に実行
                res = run_ai_agent_cycle(force_run=True) 
                if res["ok"]:
                    st.success("✅ 分析完了！最新の市場状況を反映しました。")
                    st.rerun()
                else:
                    st.error(f"実行失敗: {res.get('message', '不明なエラー')}")
    
    with col_btn2:
        if st.button("🔄 AIをリセット（資金をリセット）", use_container_width=True):
            if st.session_state.get("confirm_reset"):
                reset_ai_agent()
                st.success("AIをリセットしました。")
                st.session_state.confirm_reset = False
                st.rerun()
            else:
                st.session_state.confirm_reset = True
                st.warning("本当にリセットしますか？もう一度ボタンを押してください。")

    # 7. 取引履歴
    st.markdown("---")
    with st.expander("📜 AI取引履歴", expanded=False):
        agent_data = agent_stats.get('raw_data', {})
        if agent_data.get('history'):
            trade_df = pd.DataFrame(agent_data['history'][:20])  # 最新20件
            st.dataframe(trade_df, use_container_width=True, hide_index=True)
        else:
            st.info("まだ取引履歴がありません。")
