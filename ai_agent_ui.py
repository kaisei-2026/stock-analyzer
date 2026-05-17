"""AI自動運用システムのダッシュボード UI"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from ai_agent_engine import run_ai_agent_cycle, get_ai_agent_stats
from prediction_manager import validate_past_predictions
from data_store import reset_ai_agent

def render_ai_agent_dashboard() -> None:
    st.markdown("## 🤖 AI自動運用ダッシュボード")
    st.caption("AIが自動で銘柄を探し、30万円の仮想資金を運用します。APIリミッターにより実行頻度は制限されています。")

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

    # 2. AI運用ステータス
    agent_stats = get_ai_agent_stats()
    
    st.markdown("### 💰 運用状況")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("現在の総資産", f"{agent_stats['total_value']:,.0f} 円")
    m2.metric("リターン", f"{agent_stats['return_pct']:+.2f} %")
    m3.metric("保有銘柄数", f"{agent_stats['positions_count']}")
    m4.metric("リセット回数", f"{agent_stats['reset_count']} 回")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 保有ポジション")
        if agent_stats["positions"]:
            pos_df = []
            for t, p in agent_stats["positions"].items():
                pos_df.append({
                    "銘柄": t,
                    "株数": p["shares"],
                    "取得単価": f"{p['entry_price']:,.1f}",
                    "取得日": p["entry_date"][:10]
                })
            st.dataframe(pd.DataFrame(pos_df), use_container_width=True, hide_index=True)
        else:
            st.info("現在保有している銘柄はありません。")

        st.markdown("#### 取引履歴")
        if agent_stats["history"]:
            st.dataframe(pd.DataFrame(agent_stats["history"]), use_container_width=True, hide_index=True)
        else:
            st.caption("履歴はまだありません。")

    with col2:
        st.markdown("#### AI 操作パネル")
        if st.button("🤖 今すぐAIサイクルを実行", type="primary", use_container_width=True):
            with st.spinner("AIが市場を調査中…"):
                res = run_ai_agent_cycle()
                if res["ok"]:
                    st.success("サイクル完了！")
                    st.rerun()
                else:
                    st.warning(res["message"])
        
        if st.button("⚠️ 資金を30万円にリセット", use_container_width=True):
            reset_ai_agent(300000.0)
            st.success("リセットしました。")
            st.rerun()

        st.markdown("---")
        st.markdown("**AIの学習状態**")
        st.caption("運用結果は自動的にログに記録され、次回の分析時に特徴量として活用されます。")
