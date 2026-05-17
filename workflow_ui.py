"""ワークフロー各ステップの UI"""

from __future__ import annotations

from datetime import datetime, timezone

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
from recommendations import LOT_SIZE, PICKS_FOR_SMALL_CAPITAL, unit_cost_yen


def render_workflow_checklist() -> None:
    st.markdown(
        """
| ステップ | 状態 |
|---------|------|
| **環境** データ収集 | <span class="workflow-ok">対応済</span> |
| **環境** 取引環境構築（証券API） | <span class="workflow-ng">未実装（意図的にスコープ外）</span> |
| **①** 投資アイデア作成 | <span class="workflow-ok">対応済</span> |
| **②** 分析 & バックテスト | <span class="workflow-ok">対応済</span> |
| **③** デモトレード | <span class="workflow-ok">対応済</span> |
| **④** 本運用 | <span class="workflow-ng">未実装（意図的にスコープ外）</span> |
| **⑤** 知見の蓄積 | <span class="workflow-ok">対応済</span> |
        """,
        unsafe_allow_html=True,
    )


def render_data_collection(ticker: str, ohlcv: pd.DataFrame, fetch_period: str) -> None:
    st.subheader("データの収集")
    st.markdown(
        f"**ソース:** Yahoo Finance（yfinance） ｜ ティッカー `{ticker}` ｜ "
        f"期間 `{fetch_period}` ｜ **{len(ohlcv)}** 行"
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("OHLCV を CSV 保存（ローカルキャッシュ）", type="primary"):
            path = save_ohlcv_csv(ticker, ohlcv)
            st.success(f"保存しました: `{path}`")
    with col2:
        st.download_button(
            "ブラウザにダウンロード",
            ohlcv.to_csv().encode("utf-8-sig"),
            file_name=f"{ticker.replace('.', '_')}_ohlcv.csv",
            mime="text/csv",
        )
    st.caption("保存先フォルダ: プロジェクト内 `data/ohlcv_cache/`（Git には含めません）")
    cached = list_cached_files()
    if cached:
        st.markdown("**キャッシュ一覧（直近）**")
        for p in cached[:8]:
            st.text(f"• {p.name}")
    else:
        st.info("まだキャッシュがありません。上のボタンで保存してください。")
    st.dataframe(ohlcv.tail(15), use_container_width=True)


def render_investment_ideas(ticker: str) -> None:
    st.subheader("① 投資アイデア作成")
    st.caption("仮説・リスクをメモし、あとで分析・デモトレードと照合します。")
    with st.form("idea_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        t = c1.text_input("銘柄", value=ticker.split(".")[0] if ticker else "")
        title = c2.text_input("アイデアタイトル", placeholder="例: チャネル上抜け待ち")
        thesis = st.text_area("根拠・仮説", placeholder="なぜ今注目するか")
        risk = st.text_area("リスク", placeholder="下抜けした場合の損失イメージ")
        status = st.selectbox("ステータス", ["検討中", "分析中", "デモ中", "見送り"])
        if st.form_submit_button("アイデアを保存", type="primary"):
            if title.strip() and thesis.strip():
                add_idea(t.strip(), title.strip(), thesis.strip(), risk.strip(), status)
                st.success("保存しました。")
            else:
                st.warning("タイトルと仮説は必須です。")
    ideas = list_ideas()
    if ideas:
        st.dataframe(pd.DataFrame(ideas), use_container_width=True, hide_index=True)
        del_id = st.text_input("削除する ID（8文字）")
        if st.button("選択 ID を削除"):
            if del_id.strip():
                delete_idea(del_id.strip())
                st.rerun()
    else:
        st.info("まだアイデアがありません。")


def render_demo_trade(ticker: str, last_close: float, planning_cash: float) -> None:
    st.subheader("③ デモトレード（紙トレード）")
    st.caption("実際の証券口座には接続しません。練習用の仮想口座です。")
    account = load_demo_account()
    if float(account.get("initial_cash", 0)) != planning_cash:
        if st.button(f"デモ口座を {planning_cash:,.0f} 円でリセット"):
            reset_demo_account(planning_cash)
            st.rerun()

    prices = {ticker: last_close}
    for tk in account["positions"]:
        if tk not in prices:
            prices[tk] = account["positions"][tk].get("avg_price", 0.0)
    equity = demo_portfolio_value(account, prices)

    m1, m2, m3 = st.columns(3)
    m1.metric("現金", f"{account['cash']:,.0f} 円")
    m2.metric("評価額合計", f"{equity:,.0f} 円")
    m3.metric("損益", f"{equity - account['initial_cash']:+,.0f} 円")

    with st.form("demo_trade"):
        side = st.radio("売買", ["買い", "売り"], horizontal=True)
        tk = st.text_input("銘柄コード", value=ticker.replace(".T", ""))
        shares = st.number_input("株数", min_value=1, value=100, step=100)
        px = st.number_input("約定価格（円）", value=float(last_close), step=1.0)
        norm = tk.strip().upper()
        if norm.isdigit() and len(norm) == 4:
            norm = f"{norm}.T"
        submitted = st.form_submit_button("デモ約定を記録", type="primary")
        if submitted:
            if side == "買い":
                account, msg = demo_buy(account, norm, int(shares), float(px))
            else:
                account, msg = demo_sell(account, norm, int(shares), float(px))
            st.toast(msg)
            st.rerun()

    if account["positions"]:
        st.markdown("**保有ポジション**")
        rows = []
        for tk, pos in account["positions"].items():
            mark = prices.get(tk, pos["avg_price"])
            rows.append(
                {
                    "銘柄": tk,
                    "株数": pos["shares"],
                    "平均取得": f"{pos['avg_price']:,.2f}",
                    "評価": f"{pos['shares'] * mark:,.0f}",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if account["history"]:
        st.markdown("**約定履歴**")
        st.dataframe(pd.DataFrame(account["history"][:20]), use_container_width=True, hide_index=True)


def render_knowledge(ticker: str, backtest_snapshot: dict | None) -> None:
    st.subheader("⑤ 知見の蓄積")
    if backtest_snapshot and st.button("直近バックテスト結果を知見に保存"):
        body = "\n".join(f"{k}: {v}" for k, v in backtest_snapshot.items())
        add_knowledge(
            title=f"BT {ticker} {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            body=body,
            tags="backtest,auto",
            ticker=ticker,
            source="バックテスト自動",
        )
        st.success("知見に保存しました。")

    with st.form("knowledge_form"):
        title = st.text_input("タイトル")
        body = st.text_area("学び・メモ")
        tags = st.text_input("タグ（カンマ区切り）")
        tk = st.text_input("関連銘柄", value=ticker)
        if st.form_submit_button("知見を保存", type="primary"):
            if title.strip() and body.strip():
                add_knowledge(title.strip(), body.strip(), tags.strip(), tk.strip())
                st.success("保存しました。")
            else:
                st.warning("タイトルと本文は必須です。")

    entries = list_knowledge()
    if entries:
        st.dataframe(pd.DataFrame(entries), use_container_width=True, hide_index=True)
        del_id = st.text_input("削除する知見 ID", key="del_knowledge")
        if st.button("知見を削除"):
            if del_id.strip():
                delete_knowledge(del_id.strip())
                st.rerun()
    else:
        st.info("知見がまだありません。分析後にメモを残しましょう。")


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
    """バックテスト UI。知見保存用に metrics を返す。"""
    st.subheader(f"バックテスト（過去{sim_label}）")
    st.caption(
        f"エンジン: [{LIBRARY_NAME}]({LIBRARY_GITHUB}) ｜ "
        f"[ドキュメント]({LIBRARY_DOCS}) ｜ チャネル {channel_period} 日"
    )
    commission = float(commission_pct) / 100.0
    port_hist = run_backtest(sim_ohlcv, channel_period=channel_period, cash=float(backtest_cash), commission=commission)
    port_plan = run_backtest(sim_ohlcv, channel_period=channel_period, cash=float(planning_cash), commission=commission)
    snapshot = None

    def panel(port: dict, label: str) -> None:
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
        if not plot_df.empty:
            st.plotly_chart(build_equity_chart(plot_df, ticker, sim_label), use_container_width=True)

    hist_label = f"過去検証（{backtest_cash/10000:,.0f}万円）" if backtest_cash >= 10_000 else f"過去検証（{backtest_cash:,.0f} 円）"
    plan_label = f"運用想定（{planning_cash/10000:,.0f}万円）"
    t1, t2 = st.tabs([hist_label, plan_label])
    with t1:
        panel(port_hist, "hist")
    with t2:
        st.caption(f"1単元目安: {unit_yen:,.0f} 円")
        panel(port_plan, "plan")
    return snapshot
