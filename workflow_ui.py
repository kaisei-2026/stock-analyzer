"""ワークフロー各ステップの UI（メインタブ用）"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai_predictor import compute_buy_score, predict_direction, predict_price, run_all_predictions
from backtest_engine import LIBRARY_DOCS, LIBRARY_GITHUB, LIBRARY_NAME, equity_curve_for_plot, run_backtest
from data_store import (
    add_idea,
    add_knowledge,
    delete_idea,
    delete_knowledge,
    demo_buy,
    demo_portfolio_value,
    demo_sell,
    list_cached_files,
    list_ideas,
    list_knowledge,
    load_demo_account,
    reset_demo_account,
    save_ohlcv_csv,
)
from market_data import (
    CACHE_TTL_SEC,
    RECOMMENDATIONS_TTL_SEC,
    fetch_live_close,
    fetch_live_closes,
    fetch_recommendation_closes,
)
from recommendations import LOT_SIZE, PICKS_FOR_SMALL_CAPITAL, affordability_label, unit_cost_yen


def render_workflow_checklist() -> None:
    st.caption("本番発注・証券API連携は意図的に未実装です。")
    st.markdown(
        """
| ステップ | 状態 |
|---------|------|
| データ収集 | 対応済 |
| ① 投資アイデア | 対応済 |
| ② 分析 & バックテスト | 対応済 |
| ③ デモトレード | 対応済（自動更新可） |
| ⑤ 知見の蓄積 | 対応済 |
| 本運用 / 取引環境構築 | 未実装 |
        """
    )


def render_data_collection(ticker: str, ohlcv: pd.DataFrame, fetch_period: str) -> None:
    st.markdown(
        f"**ソース:** Yahoo Finance ｜ `{ticker}` ｜ 期間 `{fetch_period}` ｜ **{len(ohlcv)}** 行"
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("CSV をローカル保存", type="primary", key="save_csv"):
            path = save_ohlcv_csv(ticker, ohlcv)
            st.success(f"保存: `{path}`")
    with c2:
        st.download_button(
            "ダウンロード",
            ohlcv.to_csv().encode("utf-8-sig"),
            file_name=f"{ticker.replace('.', '_')}_ohlcv.csv",
            mime="text/csv",
            key="dl_csv",
        )
    cached = list_cached_files()
    if cached:
        st.markdown("**キャッシュ:** " + " / ".join(p.name for p in cached[:6]))
    st.dataframe(ohlcv.tail(12), use_container_width=True, hide_index=True)


def render_investment_ideas(ticker: str) -> None:
    st.caption("仮説・リスクを記録し、分析・デモと照合します。")
    with st.form("idea_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        t = c1.text_input("銘柄", value=ticker.split(".")[0] if ticker else "")
        title = c2.text_input("タイトル", placeholder="例: チャネル上抜け待ち")
        thesis = st.text_area("根拠・仮説")
        risk = st.text_area("リスク")
        status = st.selectbox("ステータス", ["検討中", "分析中", "デモ中", "見送り"])
        if st.form_submit_button("保存", type="primary"):
            if title.strip() and thesis.strip():
                add_idea(t.strip(), title.strip(), thesis.strip(), risk.strip(), status)
                st.success("保存しました。")
                st.rerun()
            else:
                st.warning("タイトルと仮説は必須です。")
    ideas = list_ideas()
    if ideas:
        st.dataframe(pd.DataFrame(ideas), use_container_width=True, hide_index=True)
        del_id = st.text_input("削除 ID", key="del_idea_id")
        if st.button("削除", key="del_idea_btn") and del_id.strip():
            delete_idea(del_id.strip())
            st.rerun()
    else:
        st.info("アイデアはまだありません。")


def _demo_trade_body(
    ticker: str,
    planning_cash: float,
    prices: dict[str, float],
    last_updated: str,
) -> None:
    account = load_demo_account()
    if float(account.get("initial_cash", 0)) != planning_cash:
        if st.button(f"口座を {planning_cash:,.0f} 円でリセット", key="demo_reset"):
            reset_demo_account(planning_cash)
            st.rerun()

    sym = ticker if "." in ticker else f"{ticker}.T" if ticker.isdigit() else ticker
    if sym not in prices and ticker in prices:
        prices[sym] = prices[ticker]

    equity = demo_portfolio_value(account, prices)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("現金", f"{account['cash']:,.0f} 円")
    m2.metric("評価額", f"{equity:,.0f} 円")
    m3.metric("損益", f"{equity - account['initial_cash']:+,.0f} 円")
    m4.metric("株価更新", last_updated)

    ref_px = prices.get(sym) or prices.get(ticker) or 0.0
    with st.form("demo_trade_form"):
        side = st.radio("売買", ["買い", "売り"], horizontal=True)
        tk_in = st.text_input("銘柄", value=ticker.replace(".T", ""))
        shares = st.number_input("株数", min_value=1, value=100, step=100)
        px = st.number_input("約定価格", value=float(ref_px) if ref_px else 0.0, step=1.0)
        if st.form_submit_button("約定を記録", type="primary"):
            norm = tk_in.strip().upper()
            if norm.isdigit() and len(norm) == 4:
                norm = f"{norm}.T"
            if side == "買い":
                _, msg = demo_buy(account, norm, int(shares), float(px))
            else:
                _, msg = demo_sell(account, norm, int(shares), float(px))
            st.toast(msg)
            st.rerun()

    if account["positions"]:
        rows = []
        for tk, pos in account["positions"].items():
            mark = prices.get(tk, pos["avg_price"])
            rows.append(
                {
                    "銘柄": tk,
                    "株数": pos["shares"],
                    "現在値": f"{mark:,.2f}",
                    "評価額": f"{pos['shares'] * mark:,.0f}",
                    "含み損益": f"{pos['shares'] * (mark - pos['avg_price']):+,.0f}",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if account["history"]:
        st.dataframe(pd.DataFrame(account["history"][:15]), use_container_width=True, hide_index=True)


def render_demo_trade(
    ticker: str,
    planning_cash: float,
    auto_refresh: bool,
    refresh_sec: int,
) -> None:
    st.caption(
        f"紙トレード専用。株価は **{refresh_sec} 秒キャッシュ**（実質最大 {60 // refresh_sec or 1} 回/分/銘柄）。"
        f" yfinance 上限回避のため同一銘柄は {CACHE_TTL_SEC} 秒以内は再取得しません。"
    )

    def load_prices() -> tuple[dict[str, float], str]:
        account = load_demo_account()
        symbols = {ticker}
        symbols.update(account["positions"].keys())
        sym_list = tuple(sorted({_normalize_demo_sym(s) for s in symbols if s}))
        prices = fetch_live_closes(sym_list)
        for tk in list(account["positions"].keys()):
            if tk not in prices:
                prices[tk] = account["positions"][tk]["avg_price"]
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        return prices, ts

    if auto_refresh:
        try:
            @st.fragment(run_every=timedelta(seconds=refresh_sec))
            def _live_demo() -> None:
                prices, ts = load_prices()
                _demo_trade_body(ticker, planning_cash, prices, ts)

            _live_demo()
        except TypeError:
            st.warning("自動更新には Streamlit 1.33+ が必要です。`pip install -U streamlit`")
            prices, ts = load_prices()
            _demo_trade_body(ticker, planning_cash, prices, ts)
            if st.button("今すぐ更新"):
                st.rerun()
    else:
        prices, ts = load_prices()
        _demo_trade_body(ticker, planning_cash, prices, ts)
        if st.button("株価を更新"):
            fetch_live_close.clear()
            fetch_live_close.clear()
            fetch_live_closes.clear()
            st.rerun()


def _normalize_demo_sym(ticker: str) -> str:
    t = ticker.strip().upper()
    if t.isdigit() and len(t) == 4:
        return f"{t}.T"
    return t


def render_knowledge(ticker: str, backtest_snapshot: dict | None, ohlcv: pd.DataFrame | None = None) -> None:
    from pattern_learner import detect_market_phase, generate_auto_knowledge, scan_patterns

    k_tab_ai, k_tab_pattern, k_tab_phase, k_tab_manual, k_tab_list = st.tabs([
        "🤖 AI自動学習", "🔍 パターン検索", "📊 フェーズ分析", "✏️ 手動メモ", "📚 知見一覧"
    ])

    # -----------------------------------------------------------------------
    # AI自動学習タブ
    # -----------------------------------------------------------------------
    with k_tab_ai:
        st.markdown("### AI が過去データから自動で知見を生成します")
        st.write(
            "ボタンを押すと：\n"
            "- 現在のチャートと似た過去パターンを検索\n"
            "- 相場フェーズを判定\n"
            "- RSI水準別・出来高急増後・月別のリターン統計\n"
            "をすべて自動で **知見一覧に保存** します。"
        )

        col1, col2 = st.columns(2)
        with col1:
            ai_window  = st.select_slider("パターン比較期間", options=[5, 7, 10, 14, 20], value=10, key="k_window")
        with col2:
            ai_horizon = st.select_slider("何日後を見る", options=[3, 5, 7, 10, 14], value=5, key="k_horizon")

        if st.button("🤖 AIに自動学習させる", type="primary", key="run_auto_learn"):
            if ohlcv is None or ohlcv.empty:
                st.error("チャートデータがありません。先に「② 分析」タブを開いてください。")
            else:
                with st.spinner("AIが過去データを学習中…少し待ってね🔍"):
                    items = generate_auto_knowledge(ticker, ohlcv, window=ai_window, horizon=ai_horizon)

                if not items:
                    st.warning("十分なデータがなく知見を生成できませんでした。期間を長くしてみてください。")
                else:
                    for item in items:
                        add_knowledge(**item)
                    st.success(f"✅ {len(items)}件の知見を自動保存しました！「知見一覧」タブで確認できます。")
                    st.rerun()

        # バックテスト結果も保存
        if backtest_snapshot:
            if st.button("バックテスト結果も保存", key="save_bt_knowledge"):
                body = "\n".join(f"{k}: {v}" for k, v in backtest_snapshot.items())
                add_knowledge(
                    title=f"BT {ticker} {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                    body=body,
                    tags="backtest",
                    ticker=ticker,
                    source="自動",
                )
                st.success("保存しました。")

    # -----------------------------------------------------------------------
    # パターン検索タブ
    # -----------------------------------------------------------------------
    with k_tab_pattern:
        st.markdown("### 現在の形と似た過去パターンを検索")
        st.write("今のチャートの形と似ていた過去の場面を探し、その後どうなったかを統計で表示します。")

        c1, c2, c3 = st.columns(3)
        with c1:
            p_window = st.select_slider("形の長さ（日）", options=[5, 7, 10, 14, 20], value=10, key="p_window")
        with c2:
            p_horizon = st.select_slider("何日後を見る", options=[3, 5, 7, 10, 14], value=5, key="p_horizon")
        with c3:
            p_thresh = st.slider("類似度の閾値", 0.5, 0.95, 0.75, 0.05, key="p_thresh",
                                 help="高いほど「そっくりな形」だけを探す")

        if st.button("🔍 パターンを検索", type="primary", key="run_pattern"):
            if ohlcv is None or ohlcv.empty:
                st.error("データがありません。")
            else:
                with st.spinner("過去データをスキャン中…"):
                    result = scan_patterns(ohlcv, window=p_window, horizon=p_horizon,
                                          similarity_threshold=p_thresh)

                if not result["ok"]:
                    st.error(result.get("error", "スキャン失敗"))
                elif result["total_matches"] == 0:
                    st.warning(result.get("note", "類似パターンが見つかりませんでした。閾値を下げてみてください。"))
                else:
                    n       = result["total_matches"]
                    win     = result["win_rate_pct"]
                    avg     = result["avg_return_pct"]
                    med     = result["median_return_pct"]
                    up      = result["up_count"]
                    down    = result["down_count"]
                    max_g   = result["max_gain_pct"]
                    max_l   = result["max_loss_pct"]

                    # 判定バナー
                    if win >= 65:
                        st.success(f"📈 **上昇優勢** — 過去{n}回中{up}回上昇（勝率 **{win:.1f}%**）")
                    elif win <= 35:
                        st.error(f"📉 **下落優勢** — 過去{n}回中{down}回下落（下落率 **{100-win:.1f}%**）")
                    else:
                        st.info(f"➡ **方向感なし** — 過去{n}回 上昇{up}回 / 下落{down}回（勝率{win:.1f}%）")

                    # 数値メトリクス
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("類似パターン数", f"{n}回")
                    m2.metric(f"勝率（{p_horizon}日後↑）", f"{win:.1f}%")
                    m3.metric("平均リターン", f"{avg:+.2f}%")
                    m4.metric("最大上昇", f"{max_g:+.2f}%")
                    m5.metric("最大下落", f"{max_l:+.2f}%")

                    # リターン分布棒グラフ
                    if result.get("distribution"):
                        dist = result["distribution"]
                        fig = go.Figure(go.Bar(
                            x=[d["range"] for d in dist],
                            y=[d["count"] for d in dist],
                            marker_color=["#15803d" if "+" in d["range"] or d["range"].startswith("0") else "#b91c1c"
                                          for d in dist],
                            text=[str(d["count"]) for d in dist],
                            textposition="outside",
                        ))
                        fig.update_layout(
                            title=f"{p_horizon}日後リターンの分布",
                            height=280,
                            xaxis_title="リターン範囲",
                            yaxis_title="回数",
                            paper_bgcolor="#ffffff",
                            plot_bgcolor="#f8fafc",
                            margin=dict(t=40, b=20),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # 類似パターン一覧
                    with st.expander("類似パターン詳細一覧"):
                        df_m = pd.DataFrame(result["matches"])
                        st.dataframe(df_m, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # フェーズ分析タブ
    # -----------------------------------------------------------------------
    with k_tab_phase:
        st.markdown("### 現在の相場フェーズを分析")
        st.write("移動平均の並び順から現在のトレンド状態を判定し、過去の同じフェーズでの成績を表示します。")

        if st.button("📊 フェーズを分析", type="primary", key="run_phase"):
            if ohlcv is None or ohlcv.empty:
                st.error("データがありません。")
            else:
                with st.spinner("フェーズを判定中…"):
                    result = detect_market_phase(ohlcv)

                if not result["ok"]:
                    st.error(result.get("error"))
                else:
                    st.markdown(f"## {result['phase']}")
                    st.info(result["description"])

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("SMA5", f"{result['sma5']:,.1f}")
                    c2.metric("SMA20", f"{result['sma20']:,.1f}")
                    c3.metric("SMA60", f"{result['sma60']:,.1f}")
                    c4.metric("直近5日", f"{result['ret5d']:+.1f}%")

                    ps = result.get("phase_stats", {})
                    if ps:
                        st.markdown("**過去に同じフェーズだったときの成績**")
                        p1, p2, p3 = st.columns(3)
                        p1.metric("サンプル数", f"{ps['sample_count']}回")
                        p2.metric(f"勝率（{ps['horizon_days']}日後↑）", f"{ps['win_rate_pct']:.1f}%")
                        p3.metric("平均リターン", f"{ps['avg_return_pct']:+.2f}%")

                    # SMA推移チャート
                    sma5_s  = ohlcv["Close"].rolling(5).mean().tail(120)
                    sma20_s = ohlcv["Close"].rolling(20).mean().tail(120)
                    sma60_s = ohlcv["Close"].rolling(60).mean().tail(120)
                    price_s = ohlcv["Close"].tail(120)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=price_s.index, y=price_s, name="終値",
                                             line=dict(color="#0369a1", width=2)))
                    fig.add_trace(go.Scatter(x=sma5_s.index, y=sma5_s, name="SMA5",
                                             line=dict(color="#f59e0b", width=1.5, dash="dot")))
                    fig.add_trace(go.Scatter(x=sma20_s.index, y=sma20_s, name="SMA20",
                                             line=dict(color="#15803d", width=1.5)))
                    fig.add_trace(go.Scatter(x=sma60_s.index, y=sma60_s, name="SMA60",
                                             line=dict(color="#b91c1c", width=1.5, dash="dash")))
                    fig.update_layout(
                        title="移動平均の推移（直近120日）",
                        height=320,
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#f8fafc",
                        margin=dict(t=40, b=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------------
    # 手動メモタブ
    # -----------------------------------------------------------------------
    with k_tab_manual:
        st.markdown("### 自分で気づいたことをメモ")
        with st.form("knowledge_form"):
            title = st.text_input("タイトル")
            body  = st.text_area("メモ内容")
            tags  = st.text_input("タグ（カンマ区切り）")
            tk    = st.text_input("銘柄", value=ticker)
            if st.form_submit_button("保存", type="primary"):
                if title.strip() and body.strip():
                    add_knowledge(title.strip(), body.strip(), tags.strip(), tk.strip())
                    st.success("保存しました。")
                    st.rerun()
                else:
                    st.warning("タイトルとメモは必須です。")

    # -----------------------------------------------------------------------
    # 知見一覧タブ
    # -----------------------------------------------------------------------
    with k_tab_list:
        st.markdown("### 蓄積された知見一覧")
        entries = list_knowledge()
        if entries:
            # フィルター
            sources = list({e.get("source", "—") for e in entries})
            selected_source = st.selectbox("ソースで絞り込み", ["すべて"] + sources, key="k_filter")
            filtered = entries if selected_source == "すべて" else [
                e for e in entries if e.get("source") == selected_source
            ]

            st.caption(f"{len(filtered)} 件表示中")
            st.dataframe(pd.DataFrame(filtered), use_container_width=True, hide_index=True)

            # 詳細表示
            with st.expander("内容を詳しく見る"):
                for entry in filtered[:10]:
                    st.markdown(f"**{entry.get('title','—')}** `{entry.get('created','')}`")
                    st.text(entry.get("body", ""))
                    st.markdown("---")

            # 削除
            del_id = st.text_input("削除する ID", key="del_k_id")
            if st.button("削除", key="del_k_btn") and del_id.strip():
                delete_knowledge(del_id.strip())
                st.rerun()
        else:
            st.info("まだ知見がありません。「AI自動学習」タブでAIに学習させてみましょう！")


def render_backtest_section(
    ticker: str,
    sim_label: str,
    channel_period: int,
    commission_pct: float,
    backtest_cash: float,
    planning_cash: float,
    unit_yen: float,
    sim_ohlcv: pd.DataFrame,
    build_equity_chart,
) -> dict | None:
    commission = float(commission_pct) / 100.0
    port_hist = run_backtest(
        sim_ohlcv, channel_period=channel_period, cash=float(backtest_cash), commission=commission
    )
    port_plan = run_backtest(
        sim_ohlcv, channel_period=channel_period, cash=float(planning_cash), commission=commission
    )
    snapshot = None

    def panel(port: dict) -> None:
        nonlocal snapshot
        if not port.get("ok"):
            st.warning(port.get("error", "データ不足"))
            return
        snapshot = port.get("metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最終資産", f"{port['final_strategy']:,.0f} 円")
        c2.metric("リターン", f"{port['return_strategy_pct']:+.2f}%")
        c3.metric("最大DD", f"{port['max_drawdown_pct']:.2f}%")
        c4.metric("勝率", f"{port['win_rate_pct']:.1f}%")
        plot_df = equity_curve_for_plot(port)
        if plot_df.empty:
            return
        st.plotly_chart(build_equity_chart(plot_df, ticker, sim_label), use_container_width=True)

    t1, t2 = st.tabs(
        [
            f"過去検証（{backtest_cash / 1_000_000:.0f}百万円）" if backtest_cash >= 1_000_000 else f"過去検証",
            f"運用想定（{planning_cash / 10000:.0f}万円）",
        ]
    )
    with t1:
        panel(port_hist)
    with t2:
        st.caption(f"1単元 ≈ {unit_yen:,.0f} 円")
        panel(port_plan)
    return snapshot


def render_ai_prediction_tab(ticker: str, ohlcv: pd.DataFrame, planning_cash: float) -> None:
    """AI予測タブの UI。方向予測・数値予測・買いスコア・おすすめ一括スキャンを表示。"""

    st.caption("⚠️ AI予測は過去データの統計パターンです。将来の利益を保証しません。参考情報としてご利用ください。")

    ai_tab1, ai_tab2, ai_tab3, ai_tab4 = st.tabs([
        "📈 短期方向予測", "💹 株価数値予測", "🎯 買いスコア", "🔍 おすすめ一括スキャン"
    ])

    # -----------------------------------------------------------------------
    # タブ1: 短期方向予測
    # -----------------------------------------------------------------------
    with ai_tab1:
        st.markdown("### 短期方向予測（上がる？下がる？）")
        st.write("機械学習（勾配ブースティング）が過去のテクニカル指標パターンを学習し、数日後の方向を予測します。")

        horizon = st.select_slider(
            "予測期間",
            options=[3, 5, 7, 10, 14],
            value=5,
            format_func=lambda x: f"{x}営業日後",
            key="direction_horizon"
        )

        if st.button("🔮 方向予測を実行", type="primary", key="run_direction"):
            with st.spinner("AIが学習中…"):
                result = predict_direction(ohlcv, horizon_days=horizon)

            if not result["ok"]:
                st.error(result.get("error", "予測失敗"))
            else:
                prob_up = result["prob_up"]
                prob_down = result["prob_down"]
                signal = result["signal"]
                confidence = result["confidence"]

                # メインシグナル表示
                color = "🟢" if prob_up >= 55 else ("🔴" if prob_down >= 55 else "⚪")
                st.markdown(f"## {color} {signal}")

                c1, c2, c3 = st.columns(3)
                c1.metric("上昇確率", f"{prob_up:.1f}%")
                c2.metric("下落確率", f"{prob_down:.1f}%")
                c3.metric("信頼度", confidence)

                # 確率バー
                st.markdown("**上昇 vs 下落**")
                fig = go.Figure(go.Bar(
                    x=["上昇", "下落"],
                    y=[prob_up, prob_down],
                    marker_color=["#15803d" if prob_up >= prob_down else "#94a3b8",
                                  "#b91c1c" if prob_down > prob_up else "#94a3b8"],
                    text=[f"{prob_up:.1f}%", f"{prob_down:.1f}%"],
                    textposition="outside"
                ))
                fig.update_layout(
                    height=280,
                    yaxis=dict(range=[0, 100], title="確率 (%)"),
                    margin=dict(t=20, b=20),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#f8fafc",
                )
                st.plotly_chart(fig, use_container_width=True)

                # 重要特徴量
                if result.get("top_features"):
                    st.markdown("**この予測に影響した指標 TOP5**")
                    feat_df = pd.DataFrame(result["top_features"], columns=["指標", "重要度"])
                    feat_df["重要度"] = feat_df["重要度"].map(lambda x: f"{x:.3f}")
                    st.dataframe(feat_df, use_container_width=True, hide_index=True)

                st.caption(f"学習サンプル: {result['train_samples']}日分 | {result['note']}")

    # -----------------------------------------------------------------------
    # タブ2: 株価数値予測
    # -----------------------------------------------------------------------
    with ai_tab2:
        st.markdown("### 株価数値予測（何円になる？）")
        st.write("回帰モデルが将来の株価水準を予測します。90%信頼区間も表示します。")

        forecast_days = st.select_slider(
            "予測期間",
            options=[5, 10, 20, 40, 60],
            value=20,
            format_func=lambda x: f"{x}営業日後（約{x//5}週間）",
            key="forecast_days"
        )

        if st.button("📊 株価予測を実行", type="primary", key="run_price"):
            with st.spinner("AIが計算中…"):
                result = predict_price(ohlcv, forecast_days=forecast_days)

            if not result["ok"]:
                st.error(result.get("error", "予測失敗"))
            else:
                current = result["current_price"]
                predicted = result["predicted_price"]
                ret_pct = result["predicted_return_pct"]
                lower = result["lower_bound"]
                upper = result["upper_bound"]
                label = result["confidence_label"]

                arrow = "📈" if ret_pct > 0 else "📉"
                st.markdown(f"## {arrow} {label}")

                c1, c2, c3 = st.columns(3)
                c1.metric("現在の株価", f"{current:,.1f} 円")
                c2.metric(
                    f"{forecast_days}日後予測",
                    f"{predicted:,.1f} 円",
                    delta=f"{ret_pct:+.1f}%"
                )
                c3.metric("90%信頼区間", f"{lower:,.0f}〜{upper:,.0f} 円")

                # 予測チャート（最近の株価 + 予測点）
                recent = ohlcv["Close"].tail(60)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(recent))),
                    y=recent.values,
                    name="実際の株価",
                    line=dict(color="#0369a1", width=2)
                ))
                # 予測点
                pred_x = len(recent) + forecast_days
                fig.add_trace(go.Scatter(
                    x=[len(recent) - 1, pred_x],
                    y=[float(recent.iloc[-1]), predicted],
                    name="予測",
                    line=dict(color="#15803d", width=2, dash="dash"),
                    mode="lines+markers",
                    marker=dict(size=10)
                ))
                # 信頼区間
                fig.add_trace(go.Scatter(
                    x=[pred_x, pred_x],
                    y=[lower, upper],
                    name="90%信頼区間",
                    mode="markers",
                    marker=dict(size=12, symbol="line-ns-open", color="#f59e0b", line_width=2)
                ))
                fig.update_layout(
                    height=350,
                    xaxis_title="営業日",
                    yaxis_title="株価（円）",
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#f8fafc",
                    margin=dict(t=20, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(result["note"])

    # -----------------------------------------------------------------------
    # タブ3: 買いスコア
    # -----------------------------------------------------------------------
    with ai_tab3:
        st.markdown("### 総合買いスコア（今が買い時？）")
        st.write("RSI・MACD・ボリンジャーバンド・トレンド・出来高を組み合わせた総合スコアです。")

        if st.button("🎯 買いスコアを計算", type="primary", key="run_score"):
            with st.spinner("指標を計算中…"):
                result = compute_buy_score(ohlcv)

            if not result["ok"]:
                st.error(result.get("error", "計算失敗"))
            else:
                score = result["score"]
                grade = result["grade"]

                st.markdown(f"## {grade}")

                # ゲージチャート
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    number={"suffix": "点", "font": {"size": 36}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#0369a1"},
                        "steps": [
                            {"range": [0, 25],  "color": "#fecaca"},
                            {"range": [25, 40], "color": "#fed7aa"},
                            {"range": [40, 55], "color": "#fef9c3"},
                            {"range": [55, 75], "color": "#bbf7d0"},
                            {"range": [75, 100], "color": "#86efac"},
                        ],
                        "threshold": {
                            "line": {"color": "#1d4ed8", "width": 3},
                            "thickness": 0.75,
                            "value": score
                        }
                    }
                ))
                fig.update_layout(
                    height=300,
                    margin=dict(t=20, b=0, l=20, r=20),
                    paper_bgcolor="#ffffff",
                )
                st.plotly_chart(fig, use_container_width=True)

                # 各指標の詳細
                st.markdown("**各指標の判定**")
                for sig in result["signals"]:
                    icon = sig["icon"]
                    name = sig["name"]
                    detail = sig["detail"]
                    weight = sig["weight"]
                    st.markdown(f"{icon} **{name}** （重み{weight}点） — {detail}")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("RSI", f"{result['rsi']:.1f}")
                c2.metric("MACDヒスト", f"{result['macd_hist']:+.3f}")
                c3.metric("BB位置", f"{result['bb_pos_pct']:.0f}%")
                c4.metric("出来高比", f"{result['vol_ratio']:.2f}倍")

                st.caption(result["note"])

    # -----------------------------------------------------------------------
    # タブ4: おすすめ銘柄一括スキャン
    # -----------------------------------------------------------------------
    with ai_tab4:
        st.markdown("### おすすめ銘柄 AI一括スキャン")
        st.write("固定9銘柄 ＋ 自分で追加した銘柄のAI買いスコアをまとめて確認できます。")

        # 追加銘柄入力
        extra_input = st.text_input(
            "追加銘柄（カンマ区切り）",
            placeholder="例: 7203, 9984, 6758",
            key="extra_tickers"
        )

        scan_tickers = list({p.ticker for p in PICKS_FOR_SMALL_CAPITAL})
        if extra_input.strip():
            for t in extra_input.split(","):
                t = t.strip()
                if t:
                    scan_tickers.append(t if "." in t else (f"{t}.T" if t.isdigit() else t))

        if st.button("🔍 全銘柄をスキャン", type="primary", key="run_scan"):
            results_rows = []
            progress = st.progress(0.0)
            status = st.empty()

            for i, tk in enumerate(scan_tickers):
                status.text(f"スキャン中: {tk} ({i+1}/{len(scan_tickers)})")
                try:
                    import yfinance as yf
                    from backtest_engine import normalize_ohlcv
                    raw = yf.download(tk, period="6mo", interval="1d", progress=False, auto_adjust=False)
                    if raw.empty:
                        raw = yf.Ticker(tk).history(period="6mo", auto_adjust=False)
                    scan_ohlcv = normalize_ohlcv(raw)

                    score_res = compute_buy_score(scan_ohlcv)
                    dir_res   = predict_direction(scan_ohlcv, horizon_days=5)

                    score   = score_res["score"] if score_res["ok"] else None
                    grade   = score_res["grade"] if score_res["ok"] else "—"
                    prob_up = dir_res["prob_up"] if dir_res["ok"] else None
                    signal  = dir_res["signal"] if dir_res["ok"] else "—"
                    close   = float(scan_ohlcv["Close"].iloc[-1])

                    results_rows.append({
                        "コード": tk,
                        "終値": f"{close:,.0f}",
                        "買いスコア": f"{score:.0f}点" if score is not None else "—",
                        "グレード": grade,
                        "上昇確率(5日)": f"{prob_up:.1f}%" if prob_up is not None else "—",
                        "方向シグナル": signal,
                    })
                except Exception as e:
                    results_rows.append({
                        "コード": tk, "終値": "—",
                        "買いスコア": "取得失敗", "グレード": str(e)[:30],
                        "上昇確率(5日)": "—", "方向シグナル": "—",
                    })

                progress.progress((i + 1) / len(scan_tickers))

            status.empty()
            progress.empty()

            if results_rows:
                df_result = pd.DataFrame(results_rows)
                st.dataframe(df_result, use_container_width=True, hide_index=True)
                st.success(f"{len(results_rows)}銘柄のスキャン完了！")

        st.caption("スキャンはAPIリクエストが多く発生するため、頻繁な実行は避けてください。")


def build_recommendations_table(capital: float) -> pd.DataFrame:
    codes = tuple(p.ticker for p in PICKS_FOR_SMALL_CAPITAL)
    closes = fetch_recommendation_closes(codes)
    rows = []
    for pick in PICKS_FOR_SMALL_CAPITAL:
        close = closes.get(pick.ticker)
        unit = unit_cost_yen(close) if close else None
        label = affordability_label(close, capital) if close else "（取得失敗）"
        rows.append(
            {
                "コード": pick.ticker,
                "銘柄": pick.name,
                "分類": pick.category,
                "終値": f"{close:,.0f}" if close else "—",
                f"1単元": f"{unit:,.0f}円" if unit else "—",
                "目安": label,
            }
        )
    return pd.DataFrame(rows)


def render_analysis_tab(
    *,
    ticker: str,
    jp_note: str,
    ohlcv: pd.DataFrame,
    period_label: str,
    sim_label: str,
    channel_period: int,
    chart_days: int,
    planning_cash: float,
    backtest_cash: float,
    commission_pct: float,
    state_df: pd.DataFrame,
    bt: dict,
    sim_ohlcv: pd.DataFrame,
    build_candlestick_chart,
    build_equity_chart,
) -> None:
    st.markdown(f"**{ticker}** {jp_note}")
    latest = ohlcv.iloc[-1]
    in_market = bool(state_df["position_eod"].iloc[-1]) if not state_df.empty else False
    signal_text = bt.get("current_signal") or "—"
    pos_label = "ロング" if in_market else "現金"
    st.info(f"**{pos_label}** ｜ {signal_text} ｜ チャネル {channel_period} 日")

    unit_yen = unit_cost_yen(float(latest["Close"]))
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("終値", f"{latest['Close']:,.2f}")
    c2.metric(f"1単元({LOT_SIZE}株)", f"{unit_yen:,.0f} 円")
    c3.metric("上限", f"{bt.get('last_upper', 0):,.2f}")
    c4.metric("下限", f"{bt.get('last_lower', 0):,.2f}")
    c5.metric("買付", "可" if unit_yen <= planning_cash else "不足")

    sub_chart, sub_stats, sub_bt, sub_ai, sub_rec = st.tabs(
        ["チャート", "統計", "バックテスト", "🤖 AI予測", "おすすめ銘柄"]
    )

    with sub_chart:
        st.plotly_chart(
            build_candlestick_chart(ohlcv, channel_period, chart_days, ticker),
            use_container_width=True,
        )

    with sub_stats:
        if bt.get("p_up") is None:
            st.warning("サンプル不足")
        else:
            a, b, c = st.columns(3)
            a.metric(f"{period_label} 先↑", f"{bt['p_up']:.1f}%")
            b.metric(f"{period_label} 先↓", f"{bt['p_down']:.1f}%")
            c.metric("状態", bt["current_signal"])
            st.progress(bt["p_up"] / 100.0)

    with sub_bt:
        metrics = render_backtest_section(
            ticker,
            sim_label,
            channel_period,
            commission_pct,
            backtest_cash,
            planning_cash,
            unit_yen,
            sim_ohlcv,
            build_equity_chart,
        )
        if metrics:
            st.session_state["last_backtest_metrics"] = metrics

    with sub_ai:
        render_ai_prediction_tab(ticker, ohlcv, planning_cash)

    with sub_rec:
        st.caption(f"おすすめ株価は {RECOMMENDATIONS_TTL_SEC // 60} 分キャッシュ（API 上限対策）。")
        if st.button("おすすめを今すぐ更新", key="refresh_recs"):
            fetch_recommendation_closes.clear()
            fetch_live_close.clear()
        st.dataframe(build_recommendations_table(float(planning_cash)), use_container_width=True, hide_index=True)
