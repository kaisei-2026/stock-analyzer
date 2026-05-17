from __future__ import annotations
"""ワークフロー各ステップの UI（メインタブ用）"""

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ai_predictor import compute_buy_score, predict_direction, predict_price
from backtest_engine import run_backtest, equity_curve_for_plot
from data_store import (
    add_knowledge,
    demo_buy,
    list_knowledge,
    load_demo_account,
)
from market_data import fetch_recommendation_closes
from recommendations import PICKS_FOR_SMALL_CAPITAL, unit_cost_yen


def render_workflow_checklist() -> None:
    st.caption("本番発注・証券API連携は意図的に未実装です。")
    st.markdown(
        """
| ステップ | 状態 |
|---------|------|
| データ収集 | 対応済 |
| ① 投資アイデア | 対応済 |
| ② 分析 & バックテスト | 対応済 |
| ③ デモトレード | 対応済 |
| ④ AI 予測 | 対応済 |
| ⑤ 知見の蓄積 | 対応済 |
"""
    )


def render_investment_ideas(ticker: str = None) -> None:
    st.markdown("### 投資アイデアを探す")
    st.write("資金量に合わせて、おすすめの銘柄を提案します。")

    capital_options = {
        "10万円以下": 100000,
        "30万円以下": 300000,
        "100万円以下": 1000000,
        "無制限": 999999999,
    }
    capital_label = st.select_slider("あなたの投資予算は？", options=list(capital_options.keys()), value="30万円以下")
    max_capital = capital_options[capital_label]

    if st.button("🔍 おすすめ銘柄をスキャン", type="primary"):
        with st.spinner("市場データを取得中..."):
            picks = PICKS_FOR_SMALL_CAPITAL
            closes = fetch_recommendation_closes([p["ticker"] for p in picks])
            
            valid_picks = []
            for p in picks:
                ticker_code = p["ticker"]
                if ticker_code in closes:
                    price = closes[ticker_code]
                    cost = unit_cost_yen(price)
                    if cost <= max_capital:
                        p_copy = p.copy()
                        p_copy["price"] = price
                        p_copy["cost"] = cost
                        valid_picks.append(p_copy)
            
            if not valid_picks:
                st.warning("条件に合う銘柄が見つかりませんでした。予算を上げるか、別のタイミングでお試しください。")
            else:
                st.success(f"{len(valid_picks)}件の銘柄が見つかりました！")


def render_analysis_tab(
    ticker: str,
    ohlcv: pd.DataFrame,
    channel_period: int = 20,
    backtest_cash: float = 1000000,
    commission_pct: float = 0.0,
    **kwargs
) -> None:
    st.markdown(f"## 📊 {ticker} の詳細分析")
    
    # サブタブ
    tab1, tab2 = st.tabs(["⚙ 戦略検証 (バックテスト)", "🤖 AI 予測"])
    
    with tab1:
        st.markdown(f"### 📈 {ticker} の戦略検証 (バックテスト)")
        
        # 戦略の実行
        with st.spinner("バックテストを実行中..."):
            res = run_backtest(
                ohlcv, 
                channel_period=channel_period, 
                cash=backtest_cash, 
                commission=commission_pct/100
            )
        
        if not res.get("ok"):
            st.error(f"バックテストの実行に失敗しました: {res.get('error')}")
        else:
            # 1. 運用成績サマリー
            st.markdown("#### 💰 運用成績サマリー")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("最終資産", f"¥{res['final_strategy']:,.0f}")
            ret_pct = res['return_strategy_pct']
            c2.metric("戦略リターン", f"{ret_pct:+.2f}%", delta=f"{ret_pct:+.2f}%")
            dd_pct = res['max_drawdown_pct']
            c3.metric("最大ドローダウン", f"{dd_pct:.2f}%", delta=f"{dd_pct:.2f}%", delta_color="inverse")
            c4.metric("シャープレシオ", f"{res['sharpe']:.2f}")

            c5, c6, c7, c8 = st.columns(4)
            win_rate = res['win_rate_pct']
            c5.metric("勝率", f"{win_rate:.1f}%")
            c6.metric("プロフィットファクター", f"{res['profit_factor']:.2f}")
            bh_ret = res['return_buy_hold_pct']
            c7.metric("買い持ちリターン", f"{bh_ret:+.2f}%")
            c8.metric("取引回数", int(res['num_trades']))

            # 2. 資産推移チャート
            st.markdown("#### 📈 資産推移（戦略 vs 買い持ち）")
            equity_df = equity_curve_for_plot(res)
            
            if not equity_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df['equity'], name='戦略', line=dict(color='#0369a1', width=2)))
                fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df['buy_hold_equity'], name='買い持ち', line=dict(color='#94a3b8', width=1, dash='dot')))
                fig.update_layout(height=400, margin=dict(t=20, b=20), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)

            # 3. 取引履歴
            st.markdown("#### 📜 取引履歴")
            trades = res['trades']
            if not trades.empty:
                display_trades = trades.copy()
                if 'EntryTime' in display_trades.columns:
                    display_trades['EntryTime'] = display_trades['EntryTime'].dt.strftime('%Y/%m/%d')
                if 'ExitTime' in display_trades.columns:
                    display_trades['ExitTime'] = display_trades['ExitTime'].dt.strftime('%Y/%m/%d')
                if 'ReturnPct' in display_trades.columns:
                    display_trades['損益%'] = display_trades['ReturnPct'].apply(lambda x: f"{x*100:+.2f}%")
                
                cols = [c for c in ['Size', 'EntryPrice', 'ExitPrice', 'EntryTime', 'ExitTime', '損益%'] if c in display_trades.columns]
                st.dataframe(display_trades[cols], use_container_width=True)
            else:
                st.info("期間内に取引はありませんでした。")
        
    with tab2:
        render_ai_prediction_tab(ticker, ohlcv)


def render_ai_prediction_tab(ticker: str, ohlcv: pd.DataFrame) -> None:
    st.markdown(f"### 🤖 {ticker} の AI 予測分析")
    
    ai_tab1, ai_tab2, ai_tab3 = st.tabs(["📈 短期方向予測", "💹 株価数値予測", "🎯 買いスコア"])

    with ai_tab1:
        st.markdown("### AI によるトレンド方向予測")
        if st.button("🔮 方向予測を実行", type="primary", key="run_dir_pred"):
            with st.spinner("AI分析中..."):
                result = predict_direction(ohlcv)
            if result.get("ok", False):
                prob_up = result.get("prob_up", 50)
                signal = result.get("signal", "➡ 方向感なし")
                color = "#15803d" if prob_up > 55 else "#b91c1c" if prob_up < 45 else "#334155"
                st.markdown(f"<h2 style='color: {color}; text-align: center;'>{signal}</h2>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📈 上昇確率", f"{prob_up:.1f}%")
                with col2:
                    st.metric("📉 下落確率", f"{result.get('prob_down', 50):.1f}%")
            else:
                st.error(f"予測エラー: {result.get('error', '不明なエラー')}")

    with ai_tab2:
        st.markdown("### 株価数値予測（何円になる？）")
        forecast_days = st.select_slider(
            "予測期間",
            options=[1, 5, 10, 20, 60],
            value=5,
            key="forecast_days_slider"
        )

        if st.button("📊 株価予測を実行", type="primary", key="run_price_pred"):
            with st.spinner("シミュレーション中..."):
                result = predict_price(ohlcv, forecast_days=forecast_days)
            if result.get("ok", False):
                current_price = result.get("current_price", 0)
                predicted_price = result.get("predicted_price", 0)
                change_pct = ((predicted_price - current_price) / current_price * 100) if current_price > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("現在の株価", f"¥{current_price:.0f}")
                with col2:
                    st.metric(f"{forecast_days}日後の予測", f"¥{predicted_price:.0f}")
                with col3:
                    color = "🟢" if change_pct > 0 else "🔴" if change_pct < 0 else "⚪"
                    st.metric("変化率", f"{color} {change_pct:+.2f}%")
            else:
                st.error(f"予測エラー: {result.get('error', '不明なエラー')}")

    with ai_tab3:
        st.markdown("### 総合買いスコア")
        if st.button("🎯 買いスコアを計算", type="primary", key="run_score"):
            with st.spinner("スコア計算中..."):
                result = compute_buy_score(ohlcv)
            if result.get("ok", False):
                score = result.get("score", 0)
                rating = result.get("rating", "不明")
                color = "#15803d" if score >= 70 else "#f59e0b" if score >= 50 else "#b91c1c"
                st.markdown(f"<h2 style='color: {color}; text-align: center;'>スコア: {score:.0f}/100</h2>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color: {color}; text-align: center;'>{rating}</h3>", unsafe_allow_html=True)
            else:
                st.error(f"スコア計算エラー: {result.get('error', '不明なエラー')}")


def render_demo_trade(ticker: str, planning_cash: float) -> None:
    st.markdown(f"## 🛒 {ticker} のデモトレード")
    account = load_demo_account()
    st.metric("余力現金", f"¥{account['cash']:,.0f}")
    
    # 簡易版：銘柄コードと金額を入力して買う
    col1, col2 = st.columns(2)
    with col1:
        buy_ticker = st.text_input("買う銘柄", value=ticker)
    with col2:
        buy_amount = st.number_input("株数", min_value=1, value=1, step=1)
    
    if st.button("🤝 買い注文", type="primary"):
        # 簡略版：現在価格を仮定
        current_price = 2500.0
        res = demo_buy(buy_ticker, buy_amount, current_price)
        if res["ok"]:
            st.success("購入完了")
            st.rerun()
        else:
            st.error(res["message"])


def render_knowledge(ticker: str = "", metrics: dict = None, ohlcv: pd.DataFrame = None) -> None:
    st.markdown("## ⑤ 知見の蓄積")
    st.write("投資から得た知見を記録します。")
    
    # 知見の入力
    title = st.text_input("タイトル")
    content = st.text_area("内容")
    if st.button("💾 知見を保存"):
        if title and content:
            add_knowledge(title, content)
            st.success("知見を保存しました！")
            st.rerun()
        else:
            st.warning("タイトルと内容を入力してください。")
    
    # 過去の知見を表示
    st.markdown("### 📚 過去の知見")
    knowledges = list_knowledge()
    if knowledges:
        for k in reversed(knowledges):
            st.markdown(f"#### {k['title']}")
            st.write(k['content'])
    else:
        st.info("まだ知見が記録されていません。")


def render_data_collection(ticker: str, ohlcv_data: pd.DataFrame = None, fetch_period: str = "1y") -> None:
    st.markdown("## データ収集")
    st.write("銘柄のOHLCVデータが取得できました。")
    if ohlcv_data is not None and not ohlcv_data.empty:
        st.dataframe(ohlcv_data.head(10), use_container_width=True)
        st.info(f"取得件数: {len(ohlcv_data)} 件")
