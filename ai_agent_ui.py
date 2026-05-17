"""AI自動運用ダッシュボードのUI"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from ai_agent_engine import run_ai_agent_cycle, get_ai_agent_stats
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
        
        st.markdown("---")
        st.markdown("### 📋 現在の監視銘柄")
        
        watch_list = list_watch_tickers()
        if watch_list:
            for item in watch_list:
                with st.container(border=True):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{item['ticker']}**")
                        if item.get('name'):
                            st.caption(f"名前: {item['name']}")
                        if item.get('notes'):
                            st.caption(f"📝 {item['notes']}")
                        st.caption(f"追加日: {item['added_date'][:10]}")
                    with col2:
                        if st.button("🗑️", key=f"del_{item['id']}", use_container_width=True):
                            delete_watch_ticker(item['ticker'])
                            st.success(f"{item['ticker']} を削除しました。")
                            st.rerun()
        else:
            st.info("監視銘柄がまだ登録されていません。\n左パネルで追加してください。")

    # メインエリア
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 1. 予測成功率の検証と表示
        with st.expander("📊 過去の予測成功率レポート", expanded=True):
            stats = validate_past_predictions()
            if stats["total"] > 0:
                c1, c2, c3 = st.columns(3)
                c1.metric("検証済み予測数", f"{stats['validated_count']} / {stats['total']}")
                c2.metric("総合成功率", f"{stats['success_rate']:.1f}%")
                
                # 成功率のゲージ
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=stats["success_rate"],
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#10b981"},
                        "steps": [
                            {"range": [0, 50], "color": "#fee2e2"},
                            {"range": [50, 70], "color": "#fef9c3"},
                            {"range": [70, 100], "color": "#dcfce7"},
                        ]
                    }
                ))
                fig.update_layout(height=200, margin=dict(t=0, b=0, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)
                
                if stats.get("history"):
                    st.markdown("**直近の検証履歴**")
                    st.dataframe(pd.DataFrame(stats["history"]), use_container_width=True, hide_index=True)
            else:
                st.info("まだ検証可能な過去の予測データがありません。運用を続けると自動的に集計されます。")

    with col2:
        # 2. AI運用ステータス
        agent_stats = get_ai_agent_stats()
        
        st.markdown("#### 💰 AI運用ステータス")
        c1, c2, c3 = st.columns(3)
        c1.metric("現在の資産", f"¥{agent_stats['current_cash']:,.0f}")
        c2.metric("保有ポジション数", agent_stats['position_count'])
        c3.metric("リセット回数", agent_stats['reset_count'])
        
        st.markdown("#### 📈 ポートフォリオ")
        if agent_stats['positions']:
            pos_list = []
            for ticker, pos in agent_stats['positions'].items():
                current_price = pos.get('current_price', pos['entry_price'])
                profit = (current_price - pos['entry_price']) * pos['shares']
                profit_pct = ((current_price - pos['entry_price']) / pos['entry_price'] * 100) if pos['entry_price'] > 0 else 0
                pos_list.append({
                    '銘柄': ticker,
                    '保有数': f"{pos['shares']}株",
                    '取得価格': f"¥{pos['entry_price']:,.0f}",
                    '現在価格': f"¥{current_price:,.0f}",
                    '評価額': f"¥{current_price * pos['shares']:,.0f}",
                    '損益': f"¥{profit:+,.0f} ({profit_pct:+.1f}%)"
                })
            pos_df = pd.DataFrame(pos_list)
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
        else:
            st.info("現在、保有ポジションはありません。")
        
        st.markdown("#### AI 操作パネル")
        
        # 監視銘柄の状態を確認
        watch_list = list_watch_tickers()
        if not watch_list:
            st.warning("⚠️ 監視銘柄が登録されていません。\n左のパネルで銘柄を追加してください。")
        
        if st.button("🤖 今すぐAIサイクルを実行", type="primary", use_container_width=True):
            with st.spinner("AIが市場を調査中…"):
                res = run_ai_agent_cycle()
                if res["ok"]:
                    st.success("サイクル完了！")
                    st.rerun()
                else:
                    st.warning(res["message"])
        
        if st.button("🔄 AIをリセット（資金をリセット）", use_container_width=True):
            if st.session_state.get("confirm_reset"):
                reset_ai_agent()
                st.success("AIをリセットしました。")
                st.session_state.confirm_reset = False
                st.rerun()
            else:
                st.session_state.confirm_reset = True
                st.warning("本当にリセットしますか？もう一度ボタンを押してください。")

    # 3. 取引履歴
    st.markdown("---")
    with st.expander("📜 AI取引履歴", expanded=False):
        agent_data = agent_stats.get('raw_data', {})
        if agent_data.get('trade_log'):
            trade_df = pd.DataFrame(agent_data['trade_log'])
            st.dataframe(trade_df, use_container_width=True, hide_index=True)
        else:
            st.info("まだ取引履歴がありません。")
