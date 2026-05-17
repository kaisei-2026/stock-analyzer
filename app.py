"""株価分析 Web アプリ（Streamlit + yfinance）"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# 定数・設定
# ---------------------------------------------------------------------------

DATA_SOURCE = "Yahoo Finance（yfinance 経由・無料）"
DISCLAIMER = (
    "※本ツールは過去データに基づく統計的参考情報を表示するものであり、"
    "将来の株価や投資成果を保証するものではありません。投資判断は自己責任で行ってください。"
)

PERIOD_OPTIONS = ("1日", "1週間", "2週間", "1か月")

PERIOD_CONFIG: dict[str, dict[str, int | str]] = {
    "1日": {"fetch_period": "3mo", "chart_days": 30, "sma_window": 1, "horizon_days": 1},
    "1週間": {"fetch_period": "6mo", "chart_days": 60, "sma_window": 5, "horizon_days": 5},
    "2週間": {"fetch_period": "1y", "chart_days": 90, "sma_window": 10, "horizon_days": 10},
    "1か月": {"fetch_period": "2y", "chart_days": 180, "sma_window": 20, "horizon_days": 20},
}

JP_TICKER_PATTERN = re.compile(r"^\d{4}$")

SIMULATION_OPTIONS = ("1年", "2年")
SIMULATION_FETCH = {"1年": "1y", "2年": "2y"}
SIMULATION_TRADING_DAYS = {"1年": 252, "2年": 504}


# ---------------------------------------------------------------------------
# データ処理
# ---------------------------------------------------------------------------


def normalize_ticker(raw: str) -> str:
    """日本株（4桁数字のみ）には .T を付与する。"""
    s = raw.strip().upper()
    if JP_TICKER_PATTERN.fullmatch(s):
        return f"{s}.T"
    return s


def fetch_price_history(ticker: str, fetch_period: str) -> pd.DataFrame:
    """日足終値を取得。空の場合は Ticker.history にフォールバック。"""
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


def extract_close(df: pd.DataFrame) -> pd.Series:
    """マルチインデックス列を考慮し、終値を数値 Series として返す。"""
    if df.empty:
        return pd.Series(dtype=float)

    work = df.copy()
    if isinstance(work.columns, pd.MultiIndex):
        level0 = work.columns.get_level_values(0)
        if "Close" in level0:
            close = work["Close"]
        elif "Adj Close" in level0:
            close = work["Adj Close"]
        else:
            raise KeyError("終値（Close）列が見つかりません。")
    elif "Close" in work.columns:
        close = work["Close"]
    elif "Adj Close" in work.columns:
        close = work["Adj Close"]
    else:
        raise KeyError("終値（Close）列が見つかりません。")

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce")
    close = close.dropna()
    close.name = "Close"
    return close


def compute_sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window=window, min_periods=window).mean()


def run_backtest(close: pd.Series, sma_window: int, horizon_days: int) -> dict:
    """
    シグナル: 終値 > SMA → 上昇シグナル、それ以外 → 下落シグナル。
    各シグナル時点から horizon_days 後の騰落率を集計し、上昇確率を算出。
    """
    sma = compute_sma(close, sma_window)
    aligned = pd.DataFrame({"close": close, "sma": sma}).dropna()
    aligned["forward_return"] = aligned["close"].shift(-horizon_days) / aligned["close"] - 1.0
    aligned = aligned.dropna(subset=["forward_return"])

    if aligned.empty:
        return {
            "sample_size": 0,
            "p_up": None,
            "p_down": None,
            "avg_return_pct": None,
            "bull": None,
            "bear": None,
            "current_bullish": None,
        }

    aligned["bullish"] = aligned["close"] > aligned["sma"]

    def regime_stats(mask: pd.Series) -> dict | None:
        subset = aligned.loc[mask, "forward_return"]
        if subset.empty:
            return None
        return {
            "n": int(len(subset)),
            "p_up": float((subset > 0).mean() * 100),
            "avg_return_pct": float(subset.mean() * 100),
        }

    bull = regime_stats(aligned["bullish"])
    bear = regime_stats(~aligned["bullish"])
    current_bullish = bool(close.iloc[-1] > sma.iloc[-1]) if pd.notna(sma.iloc[-1]) else None

    if current_bullish is True and bull:
        p_up = bull["p_up"]
    elif current_bullish is False and bear:
        p_up = bear["p_up"]
    else:
        p_up = float((aligned["forward_return"] > 0).mean() * 100)

    return {
        "sample_size": int(len(aligned)),
        "p_up": p_up,
        "p_down": 100.0 - p_up,
        "avg_return_pct": float(aligned["forward_return"].mean() * 100),
        "bull": bull,
        "bear": bear,
        "current_bullish": current_bullish,
    }


def trim_trading_days(close: pd.Series, trading_days: int) -> pd.Series:
    """直近 N 営業日分に切り詰める。"""
    return close.tail(min(trading_days, len(close)))


def simulate_portfolio(
    close: pd.Series,
    sma_window: int,
    initial_capital: float,
) -> dict:
    """
    SMA シグナルに従ったロングオンリー・バックテスト。

    ルール（前日終値で判定、当日の騰落を反映）:
      - 前日終値 > SMA → 当日は株式を保有（日次リターンを享受）
      - 前日終値 ≤ SMA → 当日は現金（リターン 0）
    取引コスト・税金・スリッページは含みません。
    """
    sma = compute_sma(close, sma_window)
    df = pd.DataFrame({"close": close, "sma": sma}).dropna()
    if len(df) < 2:
        return {"ok": False}

    df["signal"] = df["close"] > df["sma"]
    df["position"] = df["signal"].shift(1).eq(True).astype(float)
    df["daily_ret"] = df["close"].pct_change()
    df["strategy_ret"] = df["position"] * df["daily_ret"]
    df.loc[df.index[0], "strategy_ret"] = 0.0

    df["equity"] = initial_capital * (1 + df["strategy_ret"]).cumprod()
    first_close = float(df["close"].iloc[0])
    df["buy_hold_equity"] = initial_capital * (df["close"] / first_close)

    final_strategy = float(df["equity"].iloc[-1])
    final_buy_hold = float(df["buy_hold_equity"].iloc[-1])
    peak = df["equity"].cummax()
    max_drawdown_pct = float(((df["equity"] - peak) / peak).min() * 100)
    trades = int((df["position"].diff().fillna(0).abs() > 0).sum())
    in_market_pct = float(df["position"].mean() * 100)

    return {
        "ok": True,
        "df": df,
        "initial_capital": initial_capital,
        "final_strategy": final_strategy,
        "final_buy_hold": final_buy_hold,
        "pnl_strategy": final_strategy - initial_capital,
        "pnl_buy_hold": final_buy_hold - initial_capital,
        "return_strategy_pct": (final_strategy / initial_capital - 1) * 100,
        "return_buy_hold_pct": (final_buy_hold / initial_capital - 1) * 100,
        "max_drawdown_pct": max_drawdown_pct,
        "trades": trades,
        "in_market_pct": in_market_pct,
        "start_date": df.index[0],
        "end_date": df.index[-1],
    }


def build_equity_chart(df: pd.DataFrame, ticker: str, sim_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["equity"],
            name="シグナル運用（総資産）",
            line=dict(color="#059669", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["buy_hold_equity"],
            name="買い持ち（比較）",
            line=dict(color="#6b7280", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title=f"{ticker} — 過去{sim_label}の総資産シミュレーション",
        xaxis_title="日付",
        yaxis_title="総資産",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def build_chart(
    close: pd.Series,
    sma: pd.Series,
    ticker: str,
    sma_window: int,
    chart_days: int,
) -> go.Figure:
    plot_close = close.tail(chart_days)
    plot_sma = sma.reindex(plot_close.index)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_close.index,
            y=plot_close.values,
            name="株価（終値）",
            line=dict(color="#2563eb", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_sma.index,
            y=plot_sma.values,
            name=f"移動平均（SMA {sma_window}）",
            line=dict(color="#dc2626", width=2, dash="dot"),
        )
    )
    fig.update_layout(
        title=f"{ticker} — 株価と移動平均線",
        xaxis_title="日付",
        yaxis_title="価格",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def render_footer() -> None:
    st.markdown("---")
    st.caption(DISCLAIMER)


def main() -> None:
    st.set_page_config(page_title="株価分析ツール", page_icon="📈", layout="wide")
    st.title("株価分析ツール")
    st.caption("終値トレンド・移動平均・過去シグナルに基づく統計的な方向性参考")

    with st.sidebar:
        st.header("設定")
        ticker_input = st.text_input(
            "銘柄コード",
            value="7203",
            help="例: AAPL（米国株）、7203 または 7203.T（日本株・4桁は自動で .T 付与）",
        )
        period_label = st.radio("分析期間", PERIOD_OPTIONS, index=2)
        sim_label = st.radio("総資産シミュレーション", SIMULATION_OPTIONS, index=0)
        initial_capital = st.number_input(
            "初期資金（円）",
            min_value=10_000,
            max_value=1_000_000_000,
            value=1_000_000,
            step=100_000,
            format="%d",
        )
        st.markdown("---")
        st.markdown(
            f"**データ源**\n\n{DATA_SOURCE}\n\n"
            "予測・シミュレーションは「終値が移動平均より上か下か」のルールに基づきます。"
        )

    ticker = normalize_ticker(ticker_input)
    cfg = PERIOD_CONFIG[period_label]
    fetch_period = str(cfg["fetch_period"])
    chart_days = int(cfg["chart_days"])
    sma_window = int(cfg["sma_window"])
    horizon_days = int(cfg["horizon_days"])

    if not ticker:
        st.warning("銘柄コードを入力してください。")
        render_footer()
        return

    st.info(f"照会ティッカー: **{ticker}**" + ("（4桁コードから .T を付与）" if JP_TICKER_PATTERN.fullmatch(ticker_input.strip()) else ""))

    try:
        with st.spinner("株価データを取得しています…"):
            raw_df = fetch_price_history(ticker, fetch_period)
            close = extract_close(raw_df)
    except Exception as exc:
        st.error(f"データ取得に失敗しました: {exc}")
        render_footer()
        return

    if close.empty:
        st.error(f"「{ticker}」のデータが見つかりませんでした。銘柄コードを確認してください。")
        render_footer()
        return

    sma = compute_sma(close, sma_window)
    latest_close = float(close.iloc[-1])
    latest_sma = float(sma.iloc[-1]) if pd.notna(sma.iloc[-1]) else None
    latest_date = close.index[-1]
    if hasattr(latest_date, "strftime"):
        latest_date_str = latest_date.strftime("%Y-%m-%d")
    else:
        latest_date_str = str(latest_date)[:10]

    backtest = run_backtest(close, sma_window, horizon_days)

    # --- メトリクス ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("直近の終値", f"{latest_close:,.2f}")
    col2.metric("直近 SMA", f"{latest_sma:,.2f}" if latest_sma is not None else "—")
    col3.metric("データ最終日", latest_date_str)
    col4.metric(
        "現在のトレンド",
        "終値 > SMA" if backtest.get("current_bullish") else "終値 ≤ SMA",
    )

    # --- チャート ---
    st.subheader("株価チャート")
    st.plotly_chart(
        build_chart(close, sma, ticker, sma_window, chart_days),
        use_container_width=True,
    )

    # --- 方向性（バックテスト根拠付き） ---
    st.subheader(f"方向性参考（{period_label} 先の過去実績ベース）")
    st.markdown(
        f"**ルール**: 終値が SMA（{sma_window} 日）より上なら上昇シグナル、下なら下落シグナル。"
        f" 各シグナルから **{horizon_days} 営業日後** の騰落率を集計しています。"
    )

    if backtest["p_up"] is None or backtest["sample_size"] < 10:
        st.warning(
            "バックテストに必要なデータが不足しています。"
            " 期間を長めに選ぶか、別銘柄をお試しください。"
        )
    else:
        p_up = backtest["p_up"]
        p_down = backtest["p_down"]
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{period_label} 先に上昇した確率（参考）", f"{p_up:.1f}%")
        m2.metric(f"{period_label} 先に下落した確率（参考）", f"{p_down:.1f}%")
        m3.metric("バックテスト平均リターン", f"{backtest['avg_return_pct']:+.2f}%")

        st.progress(p_up / 100.0, text=f"上昇 {p_up:.1f}% / 下落 {p_down:.1f}%")

        regime_cols = st.columns(2)
        with regime_cols[0]:
            st.markdown("**上昇シグナル時（終値 > SMA）**")
            bull = backtest["bull"]
            if bull:
                st.write(f"- サンプル数: {bull['n']}")
                st.write(f"- {horizon_days} 日後に上昇した割合: **{bull['p_up']:.1f}%**")
                st.write(f"- 平均リターン: {bull['avg_return_pct']:+.2f}%")
            else:
                st.write("データ不足")

        with regime_cols[1]:
            st.markdown("**下落シグナル時（終値 ≤ SMA）**")
            bear = backtest["bear"]
            if bear:
                st.write(f"- サンプル数: {bear['n']}")
                st.write(f"- {horizon_days} 日後に上昇した割合: **{bear['p_up']:.1f}%**")
                st.write(f"- 平均リターン: {bear['avg_return_pct']:+.2f}%")
            else:
                st.write("データ不足")

        st.caption(
            f"現在のシグナルに該当する過去パターンから上昇確率を表示しています。"
            f" 全サンプル数: {backtest['sample_size']} 件（重複なしの営業日ベース）。"
        )

    # --- 総資産シミュレーション（1年 / 2年） ---
    st.subheader(f"総資産シミュレーション（過去{sim_label}）")
    st.markdown(
        f"**初期資金 {initial_capital:,.0f} 円** で、SMA（{sma_window} 日）シグナルに従い取引した場合の総資産推移です。"
        " 終値 > SMA の翌営業日から保有、終値 ≤ SMA の翌営業日から現金化（ロングオンリー）。"
        " 手数料・税金は未考慮です。"
    )

    sim_fetch = SIMULATION_FETCH[sim_label]
    sim_days = SIMULATION_TRADING_DAYS[sim_label]

    try:
        sim_raw = fetch_price_history(ticker, sim_fetch)
        sim_close = trim_trading_days(extract_close(sim_raw), sim_days)
        portfolio = simulate_portfolio(sim_close, sma_window, float(initial_capital))
    except Exception as exc:
        st.error(f"シミュレーションの計算に失敗しました: {exc}")
        portfolio = {"ok": False}

    if not portfolio.get("ok"):
        st.warning("シミュレーションに必要なデータが不足しています。")
    else:
        sim_df = portfolio["df"]
        start_s = portfolio["start_date"].strftime("%Y-%m-%d") if hasattr(portfolio["start_date"], "strftime") else str(portfolio["start_date"])[:10]
        end_s = portfolio["end_date"].strftime("%Y-%m-%d") if hasattr(portfolio["end_date"], "strftime") else str(portfolio["end_date"])[:10]

        s1, s2, s3, s4 = st.columns(4)
        s1.metric(
            f"シグナル運用後の総資産（{sim_label}）",
            f"{portfolio['final_strategy']:,.0f} 円",
            f"{portfolio['pnl_strategy']:+,.0f} 円",
        )
        s2.metric("リターン（シグナル運用）", f"{portfolio['return_strategy_pct']:+.2f}%")
        s3.metric(
            "買い持ち後の総資産（比較）",
            f"{portfolio['final_buy_hold']:,.0f} 円",
            f"{portfolio['pnl_buy_hold']:+,.0f} 円",
        )
        s4.metric("リターン（買い持ち）", f"{portfolio['return_buy_hold_pct']:+.2f}%")

        diff = portfolio["final_strategy"] - portfolio["final_buy_hold"]
        st.caption(
            f"期間: {start_s} 〜 {end_s} ／ "
            f"売買回数（ポジション切替）: {portfolio['trades']} 回 ／ "
            f"株式保有率: {portfolio['in_market_pct']:.1f}% ／ "
            f"最大ドローダウン（シグナル運用）: {portfolio['max_drawdown_pct']:.2f}% ／ "
            f"買い持ちとの差額: {diff:+,.0f} 円"
        )

        st.plotly_chart(
            build_equity_chart(sim_df, ticker, sim_label),
            use_container_width=True,
        )

    # --- データ透明性 ---
    with st.expander("データ源・取得内容の詳細", expanded=False):
        st.markdown(
            f"""
| 項目 | 内容 |
|------|------|
| データプロバイダ | {DATA_SOURCE} |
| 照会ティッカー | `{ticker}` |
| 取得期間（API） | `{fetch_period}` |
| 取得行数 | {len(close)} 行 |
| チャート表示 | 直近 {chart_days} 営業日 |
| SMA 窓 | {sma_window} 日 |
| バックテスト先読み | {horizon_days} 営業日 |
| 取得日時（UTC） | {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")} |
            """
        )
        st.dataframe(
            pd.DataFrame({"終値": close, f"SMA{sma_window}": sma}).tail(10),
            use_container_width=True,
        )

    render_footer()


if __name__ == "__main__":
    main()
