"""ワークフロー各ステップの UI（メインタブ用）"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

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


def render_knowledge(ticker: str, backtest_snapshot: dict | None) -> None:
    if backtest_snapshot:
        if st.button("直近バックテストを知見に保存", key="save_bt_knowledge"):
            body = "\n".join(f"{k}: {v}" for k, v in backtest_snapshot.items())
            add_knowledge(
                title=f"BT {ticker} {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                body=body,
                tags="backtest",
                ticker=ticker,
                source="自動",
            )
            st.success("保存しました。")
    with st.form("knowledge_form"):
        title = st.text_input("タイトル")
        body = st.text_area("メモ")
        tags = st.text_input("タグ")
        tk = st.text_input("銘柄", value=ticker)
        if st.form_submit_button("保存", type="primary"):
            if title.strip() and body.strip():
                add_knowledge(title.strip(), body.strip(), tags.strip(), tk.strip())
                st.success("保存しました。")
                st.rerun()
    entries = list_knowledge()
    if entries:
        st.dataframe(pd.DataFrame(entries), use_container_width=True, hide_index=True)
    else:
        st.info("知見がまだありません。")


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

    sub_chart, sub_stats, sub_bt, sub_rec = st.tabs(
        ["チャート", "統計", "バックテスト", "おすすめ銘柄"]
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

    with sub_rec:
        st.caption(f"おすすめ株価は {RECOMMENDATIONS_TTL_SEC // 60} 分キャッシュ（API 上限対策）。")
        if st.button("おすすめを今すぐ更新", key="refresh_recs"):
            fetch_recommendation_closes.clear()
            fetch_live_close.clear()
        st.dataframe(build_recommendations_table(float(planning_cash)), use_container_width=True, hide_index=True)
