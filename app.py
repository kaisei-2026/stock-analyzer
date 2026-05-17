"""株価分析 Web アプリ — ローソク足 × チャネルブレイクアウト"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

from backtest_engine import normalize_ohlcv
from recommendations import (
    DEFAULT_BACKTEST_CASH,
    DEFAULT_PLANNING_CASH,
    LOT_SIZE,
    PICKS_FOR_SMALL_CAPITAL,
    REFERENCE_LINKS,
    affordability_label,
    unit_cost_yen,
)
from theme import COLORS, inject_theme, style_figure
from workflow_ui import (
    render_backtest_section,
    render_data_collection,
    render_demo_trade,
    render_investment_ideas,
    render_knowledge,
    render_workflow_checklist,
)

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

DATA_SOURCE = "Yahoo Finance（yfinance 経由・無料）"
DISCLAIMER = (
    "※本ツールは過去データに基づく統計的参考情報です。"
    "将来の利益を保証しません。投資は自己責任で行ってください。"
)

PERIOD_OPTIONS = ("1日", "1週間", "2週間", "1か月")
PERIOD_CONFIG: dict[str, dict[str, int | str]] = {
    "1日": {"fetch_period": "3mo", "chart_days": 40, "channel_period": 5, "horizon_days": 1},
    "1週間": {"fetch_period": "6mo", "chart_days": 70, "channel_period": 10, "horizon_days": 5},
    "2週間": {"fetch_period": "1y", "chart_days": 100, "channel_period": 20, "horizon_days": 10},
    "1か月": {"fetch_period": "2y", "chart_days": 160, "channel_period": 20, "horizon_days": 20},
}

JP_TICKER_PATTERN = re.compile(r"^\d{4}$")
SIMULATION_OPTIONS = ("1年", "2年")
SIMULATION_FETCH = {"1年": "1y", "2年": "2y"}
SIMULATION_TRADING_DAYS = {"1年": 252, "2年": 504}

WORKFLOW_PAGES = (
    "② 分析＆バックテスト",
    "① 投資アイデア",
    "③ デモトレード",
    "⑤ 知見の蓄積",
    "データ収集",
)


# ---------------------------------------------------------------------------
# データ
# ---------------------------------------------------------------------------


def normalize_ticker(raw: str) -> str:
    s = raw.strip().upper()
    if JP_TICKER_PATTERN.fullmatch(s):
        return f"{s}.T"
    return s


def fetch_price_history(ticker: str, fetch_period: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=fetch_period,
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if df.empty:
        df = yf.Ticker(ticker).history(period=fetch_period, interval="1d", auto_adjust=False)
    return df


def extract_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV を正規化（列名は大小文字不問 → backtest_engine.normalize_ohlcv）。"""
    if df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    try:
        return normalize_ohlcv(df)
    except ValueError:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def trim_ohlcv(ohlcv: pd.DataFrame, trading_days: int) -> pd.DataFrame:
    return ohlcv.tail(min(trading_days, len(ohlcv)))


# ---------------------------------------------------------------------------
# チャネルブレイクアウト（ドンチャン）
# ---------------------------------------------------------------------------


def compute_donchian_channel(ohlcv: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    前日までの N 日間の高値・安値でチャネルを構成（先読みなし）。
    upper: 当日を除く過去 period 日の最高値
    lower: 当日を除く過去 period 日の最安値
    """
    upper = ohlcv["High"].rolling(period, min_periods=period).max().shift(1)
    lower = ohlcv["Low"].rolling(period, min_periods=period).min().shift(1)
    mid = (upper + lower) / 2
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": mid}, index=ohlcv.index)


def run_channel_state_machine(ohlcv: pd.DataFrame, channel_period: int) -> pd.DataFrame:
    """
    エントリー: 終値がチャネル上限を上抜け → ロング
    イグジット: 終値がチャネル下限を下抜け → 現金
  当日始値時点のポジションは前日シグナル確定後の状態。
    """
    ch = compute_donchian_channel(ohlcv, channel_period)
    df = ohlcv.join(ch).dropna().copy()
    if df.empty:
        return df

    in_position = False
    positions_start: list[float] = []
    positions_end: list[float] = []
    entries: list[bool] = []
    exits: list[bool] = []

    for _, row in df.iterrows():
        pos_start = 1.0 if in_position else 0.0
        positions_start.append(pos_start)

        entry = (not in_position) and (row["Close"] > row["upper"])
        exit_ = in_position and (row["Close"] < row["lower"])
        entries.append(entry)
        exits.append(exit_)

        if entry:
            in_position = True
        elif exit_:
            in_position = False

        positions_end.append(1.0 if in_position else 0.0)

    df["position"] = positions_start
    df["position_eod"] = positions_end
    df["entry_signal"] = entries
    df["exit_signal"] = exits
    df["daily_ret"] = df["Close"].pct_change()
    df["strategy_ret"] = df["position"] * df["daily_ret"]
    df.loc[df.index[0], "strategy_ret"] = 0.0
    return df


def run_breakout_backtest(ohlcv: pd.DataFrame, channel_period: int, horizon_days: int) -> dict:
    """ブレイクアウト / ブレイクダウン後の先読みリターン統計。"""
    ch = compute_donchian_channel(ohlcv, channel_period)
    df = ohlcv.join(ch).dropna()
    df["forward_return"] = df["Close"].shift(-horizon_days) / df["Close"] - 1.0
    df = df.dropna(subset=["forward_return"])
    if df.empty:
        return {"sample_size": 0, "p_up": None, "p_down": None, "current_signal": None}

    df["breakout"] = df["Close"] > df["upper"]
    df["breakdown"] = df["Close"] < df["lower"]

    def stats(mask: pd.Series) -> dict | None:
        sub = df.loc[mask, "forward_return"]
        if sub.empty:
            return None
        return {
            "n": int(len(sub)),
            "p_up": float((sub > 0).mean() * 100),
            "avg_return_pct": float(sub.mean() * 100),
        }

    bo = stats(df["breakout"])
    bd = stats(df["breakdown"])
    last = df.iloc[-1]
    if last["Close"] > last["upper"]:
        current = "ブレイクアウト（上限突破）"
        p_up = bo["p_up"] if bo else None
    elif last["Close"] < last["lower"]:
        current = "ブレイクダウン（下限割れ）"
        p_up = bd["p_up"] if bd else None
    else:
        current = "チャネル内"
        p_up = float((df["forward_return"] > 0).mean() * 100)

    return {
        "sample_size": int(len(df)),
        "p_up": p_up,
        "p_down": (100.0 - p_up) if p_up is not None else None,
        "breakout": bo,
        "breakdown": bd,
        "current_signal": current,
        "last_upper": float(last["upper"]),
        "last_lower": float(last["lower"]),
        "dist_to_upper_pct": float((last["upper"] - last["Close"]) / last["Close"] * 100),
        "dist_to_lower_pct": float((last["Close"] - last["lower"]) / last["Close"] * 100),
    }


# ---------------------------------------------------------------------------
# チャート
# ---------------------------------------------------------------------------


def build_candlestick_chart(
    ohlcv: pd.DataFrame,
    channel_period: int,
    chart_days: int,
    ticker: str,
) -> go.Figure:
    df = run_channel_state_machine(ohlcv, channel_period).tail(chart_days)
    if df.empty:
        return go.Figure()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="ローソク足",
            increasing_line_color=COLORS["up"],
            increasing_fillcolor=COLORS["up"],
            decreasing_line_color=COLORS["down"],
            decreasing_fillcolor=COLORS["down"],
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["upper"],
            name=f"チャネル上限 ({channel_period}日高値)",
            line=dict(color=COLORS["channel_hi"], width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["lower"],
            name=f"チャネル下限 ({channel_period}日安値)",
            line=dict(color=COLORS["channel_lo"], width=1.5),
            fill="tonexty",
            fillcolor=COLORS["channel_fill"],
        ),
        row=1,
        col=1,
    )

    buy_df = df[df["entry_signal"]]
    sell_df = df[df["exit_signal"]]
    if not buy_df.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_df.index,
                y=buy_df["Close"],
                mode="markers",
                name="エントリー（上抜け）",
                marker=dict(symbol="triangle-up", size=14, color=COLORS["buy"], line=dict(width=1, color="#ffffff")),
            ),
            row=1,
            col=1,
        )
    if not sell_df.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_df.index,
                y=sell_df["Close"],
                mode="markers",
                name="イグジット（下抜け）",
                marker=dict(symbol="triangle-down", size=14, color=COLORS["sell"], line=dict(width=1, color="#ffffff")),
            ),
            row=1,
            col=1,
        )

    vol_colors = [
        COLORS["up"] if c >= o else COLORS["down"]
        for o, c in zip(df["Open"], df["Close"])
    ]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="出来高", marker_color=vol_colors, opacity=0.55),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=dict(text=f"{ticker} ｜ チャネルブレイクアウト", x=0.02, font=dict(size=18)),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1),
    )
    fig.update_yaxes(title_text="価格", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return style_figure(fig, height=620)


def build_equity_chart(df: pd.DataFrame, ticker: str, sim_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["equity"],
            name="ブレイクアウト戦略",
            line=dict(color=COLORS["equity"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(3, 105, 161, 0.08)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["buy_hold_equity"],
            name="買い持ち",
            line=dict(color=COLORS["benchmark"], width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title=dict(text=f"{ticker} ｜ 過去{sim_label} 総資産シミュレーション", x=0.02),
        xaxis_title="日付",
        yaxis_title="総資産（円）",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1),
    )
    return style_figure(fig, height=400)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def render_footer() -> None:
    st.markdown("---")
    st.caption(DISCLAIMER)


def fmt_date(idx) -> str:
    return idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_last_closes_jp(tickers: tuple[str, ...]) -> dict[str, float]:
    """おすすめ銘柄の直近終値を取得（銘柄ごとに取得して精度を確保）。"""
    closes: dict[str, float] = {}
    for code in tickers:
        sym = f"{code}.T" if not str(code).endswith(".T") else str(code)
        try:
            raw = yf.download(sym, period="5d", interval="1d", progress=False, auto_adjust=False)
            if raw.empty:
                raw = yf.Ticker(sym).history(period="5d", auto_adjust=False)
            ohlcv = extract_ohlcv(raw)
            if not ohlcv.empty:
                closes[code] = float(ohlcv["Close"].iloc[-1])
        except Exception:
            continue
    return closes


def build_recommendations_table(capital: float) -> pd.DataFrame:
    codes = tuple(p.ticker for p in PICKS_FOR_SMALL_CAPITAL)
    closes = fetch_last_closes_jp(codes)
    rows = []
    for pick in PICKS_FOR_SMALL_CAPITAL:
        close = closes.get(pick.ticker)
        if close is None:
            unit = None
            label = "（株価取得失敗）"
        else:
            unit = unit_cost_yen(close)
            label = affordability_label(close, capital)
        rows.append(
            {
                "コード": pick.ticker,
                "銘柄": pick.name,
                "分類": pick.category,
                "直近終値": f"{close:,.0f}" if close else "—",
                f"1単元({LOT_SIZE}株)": f"{unit:,.0f}円" if unit else "—",
                f"{capital/10000:.0f}万円での目安": label,
                "メモ": pick.note,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(
        page_title="Breakout Trader",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()
    st.title("Breakout Trader")
    st.caption("投資ワークフロー: アイデア → 分析 → デモトレード → 知見（本番取引は非対応）")

    with st.sidebar:
        page = st.radio("ワークフロー", WORKFLOW_PAGES, index=0)
        st.markdown("---")
        with st.expander("🚀 起動方法", expanded=True):
            st.markdown(
                """
1. ターミナルでこのフォルダへ移動  
2. `pip install -r requirements.txt`  
3. **`streamlit run app.py`**  
4. ブラウザで `http://localhost:8501` を開く  

CLI のみ: `python run_backtest.py --ticker 7203.T --period 1y`
                """
            )

        st.header("⚙ トレード設定")

        pick_labels = ["（自分で入力）"] + [
            f"{p.ticker} {p.name}" for p in PICKS_FOR_SMALL_CAPITAL
        ]
        pick_choice = st.selectbox("おすすめから選ぶ（10万円向け）", pick_labels, index=0)
        default_code = "8306" if pick_choice == "（自分で入力）" else pick_choice.split()[0]
        ticker_input = st.text_input(
            "銘柄コード",
            value=default_code,
            help="4桁数字 → 自動で .T ／ ETF は 1306 など",
        )

        period_label = st.radio("チャネル期間（分析）", PERIOD_OPTIONS, index=2)
        sim_label = st.radio("バックテスト期間", SIMULATION_OPTIONS, index=0)

        st.markdown("**資金の使い分け**")
        backtest_cash = st.number_input(
            "過去バックテスト（検証用・円）",
            min_value=100,
            max_value=10_000_000,
            value=DEFAULT_BACKTEST_CASH,
            step=100,
            format="%d",
            help="成績比較用。1,000円など小さくても%は同じ目安になります。",
        )
        planning_cash = st.number_input(
            "これから投資する想定（円）",
            min_value=10_000,
            max_value=10_000_000,
            value=DEFAULT_PLANNING_CASH,
            step=10_000,
            format="%d",
            help="10万円など、実際に使える資金でシミュレーションします。",
        )
        commission_pct = st.number_input(
            "片道手数料（%）",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            format="%.3f",
            help="backtesting.py の commission（例: 0.1% → 0.001）",
        )
        st.markdown("---")
        st.markdown(
            f"**戦略**\n\n"
            f"過去 N 日の**高値**を上限・**安値**を下限とするチャネル。\n\n"
            f"• 終値が上限を**上抜け** → エントリー\n"
            f"• 終値が下限を**下抜け** → イグジット\n\n"
            f"**データ:** {DATA_SOURCE}"
        )
        with st.expander("📚 参考リンク（Qiita・公式）"):
            for ref in REFERENCE_LINKS:
                st.markdown(f"- [{ref['title']}]({ref['url']})（{ref['site']}）")

    ticker = normalize_ticker(ticker_input)
    cfg = PERIOD_CONFIG[period_label]
    fetch_period = str(cfg["fetch_period"])
    chart_days = int(cfg["chart_days"])
    channel_period = int(cfg["channel_period"])
    horizon_days = int(cfg["horizon_days"])

    if not ticker:
        st.warning("銘柄コードを入力してください。")
        render_footer()
        return

    jp_note = "（4桁 → .T 付与）" if JP_TICKER_PATTERN.fullmatch(ticker_input.strip()) else ""

    with st.expander("ワークフロー対応状況", expanded=False):
        render_workflow_checklist()

    if page != "② 分析＆バックテスト":
        st.markdown(f"**対象銘柄:** `{ticker}` {jp_note}")

    try:
        with st.spinner("マーケットデータ取得中…"):
            raw = fetch_price_history(ticker, fetch_period)
            ohlcv = extract_ohlcv(raw)
    except Exception as exc:
        st.error(f"データ取得失敗: {exc}")
        render_footer()
        return

    if ohlcv.empty:
        st.error(f"「{ticker}」のデータがありません。")
        render_footer()
        return

    if page == "データ収集":
        render_data_collection(ticker, ohlcv, fetch_period)
        render_footer()
        return

    if page == "① 投資アイデア":
        render_investment_ideas(ticker)
        render_footer()
        return

    if page == "③ デモトレード":
        render_demo_trade(ticker, float(ohlcv["Close"].iloc[-1]), float(planning_cash))
        render_footer()
        return

    if page == "⑤ 知見の蓄積":
        snap = st.session_state.get("last_backtest_metrics")
        render_knowledge(ticker, snap)
        render_footer()
        return

    # --- ② 分析＆バックテスト ---
    st.markdown(f"**{ticker}** {jp_note}")
    state_df = run_channel_state_machine(ohlcv, channel_period)
    bt = run_breakout_backtest(ohlcv, channel_period, horizon_days)
    latest = ohlcv.iloc[-1]
    in_market = bool(state_df["position_eod"].iloc[-1]) if not state_df.empty else False

    # --- シグナルバナー ---
    signal_text = bt.get("current_signal") or "—"
    pos_label = "🟢 ロング保有中" if in_market else "⚪ 現金待機"
    st.markdown(
        f'<div class="signal-banner">'
        f"<strong>{pos_label}</strong> ｜ 状態: {signal_text} ｜ "
        f"チャネル {channel_period} 日</div>",
        unsafe_allow_html=True,
    )

    unit_yen = unit_cost_yen(float(latest["Close"]))
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("直近終値", f"{latest['Close']:,.2f}")
    c2.metric(f"1単元({LOT_SIZE}株)", f"{unit_yen:,.0f} 円")
    c3.metric(
        f"{planning_cash/10000:.0f}万円で",
        "買付可" if unit_yen <= planning_cash else "単元不足",
    )
    c4.metric("チャネル上限", f"{bt.get('last_upper', 0):,.2f}")
    c5.metric("チャネル下限", f"{bt.get('last_lower', 0):,.2f}")
    c6.metric("データ日", fmt_date(ohlcv.index[-1]))

    # --- おすすめ銘柄（10万円前後） ---
    st.subheader(f"💡 おすすめ銘柄（{planning_cash/10000:.0f}万円・1単元目安）")
    st.caption(
        "流動性と単元の大きさを踏まえた**候補**です。推奨売買ではありません。"
        " 最新株価で「1単元」が資金内か確認してから検討してください。"
    )
    with st.spinner("おすすめ銘柄の株価を確認中…"):
        rec_df = build_recommendations_table(float(planning_cash))
    st.dataframe(rec_df, use_container_width=True, hide_index=True)
    affordable = rec_df[rec_df[f"{planning_cash/10000:.0f}万円での目安"].str.startswith("✅")]
    if not affordable.empty:
        st.success(
            "資金内で1単元が買える候補: "
            + "、".join(f"{r['コード']} {r['銘柄']}" for _, r in affordable.iterrows())
        )

    # --- ローソク足チャート ---
    st.subheader("📈 ローソク足 & チャネル")
    st.plotly_chart(
        build_candlestick_chart(ohlcv, channel_period, chart_days, ticker),
        use_container_width=True,
        config={"displayModeBar": True, "scrollZoom": True},
    )

    # --- 方向性統計 ---
    st.subheader(f"🎯 ブレイクアウト統計（{period_label} 先）")
    st.markdown(
        f"過去 **{channel_period} 日**の高値・安値チャネルで、"
        f"上抜け / 下抜け後 **{horizon_days} 営業日**の騰落率を集計。"
    )

    if bt.get("p_up") is None or bt["sample_size"] < 10:
        st.warning("統計サンプルが不足しています。期間を長くするか別銘柄を試してください。")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{period_label} 先 上昇確率", f"{bt['p_up']:.1f}%")
        m2.metric(f"{period_label} 先 下落確率", f"{bt['p_down']:.1f}%")
        m3.metric("現在の状態", bt["current_signal"])

        st.progress(bt["p_up"] / 100.0)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**上抜け（ブレイクアウト）後**")
            bo = bt.get("breakout")
            if bo:
                st.write(f"サンプル {bo['n']} ｜ 上昇 {bo['p_up']:.1f}% ｜ 平均 {bo['avg_return_pct']:+.2f}%")
            else:
                st.write("データ不足")
        with r2:
            st.markdown("**下抜け（ブレイクダウン）後**")
            bd = bt.get("breakdown")
            if bd:
                st.write(f"サンプル {bd['n']} ｜ 上昇 {bd['p_up']:.1f}% ｜ 平均 {bd['avg_return_pct']:+.2f}%")
            else:
                st.write("データ不足")

    try:
        sim_raw = fetch_price_history(ticker, SIMULATION_FETCH[sim_label])
        sim_ohlcv = trim_ohlcv(extract_ohlcv(sim_raw), SIMULATION_TRADING_DAYS[sim_label])
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
    except Exception as exc:
        st.error(f"バックテスト失敗: {exc}")

    # --- 直近トレードログ ---
    with st.expander("📋 直近のエントリー / イグジット", expanded=False):
        if not state_df.empty:
            log = state_df[state_df["entry_signal"] | state_df["exit_signal"]][
                ["Close", "upper", "lower", "entry_signal", "exit_signal"]
            ].tail(12)
            log["種別"] = log.apply(
                lambda r: "🟢 ENTRY" if r["entry_signal"] else "🔴 EXIT", axis=1
            )
            st.dataframe(
                log[["種別", "Close", "upper", "lower"]].rename(
                    columns={"Close": "終値", "upper": "上限", "lower": "下限"}
                ),
                use_container_width=True,
            )

    with st.expander("🔍 データ透明性", expanded=False):
        st.markdown(
            f"| 項目 | 値 |\n|---|---|\n"
            f"| ソース | {DATA_SOURCE} |\n"
            f"| ティッカー | `{ticker}` |\n"
            f"| チャネル期間 | {channel_period} 日（前日までの高安） |\n"
            f"| 取得 | `{fetch_period}` / {len(ohlcv)} 行 |\n"
            f"| UTC | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} |"
        )
        st.dataframe(ohlcv.tail(8), use_container_width=True)

    render_footer()


if __name__ == "__main__":
    main()
