"""
OHLCV バックテストエンジン — backtesting.py ラッパー

採用ライブラリ: backtesting.py (kernc)
  - 公式: https://kernc.github.io/backtesting.py/
  - GitHub: https://github.com/kernc/backtesting.py
  - OHLCV DataFrame を入力とし、Strategy.init / next で検証済みの API を使用

列名は open/OPEN など大文字小文字を問わず正規化します。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

# 正規化後の標準列名（backtesting.py 必須）
REQUIRED_COLS = ("Open", "High", "Low", "Close")
OPTIONAL_COL = "Volume"

# エイリアス → 標準名（小文字キー）
_COLUMN_ALIASES: dict[str, str] = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "o": "Open",
    "h": "High",
    "l": "Low",
    "c": "Close",
    "v": "Volume",
    "adj close": "Close",
    "adj_close": "Close",
}

LIBRARY_NAME = "backtesting.py"
LIBRARY_DOCS = "https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html"
LIBRARY_GITHUB = "https://github.com/kernc/backtesting.py"


# ---------------------------------------------------------------------------
# OHLCV 正規化
# ---------------------------------------------------------------------------


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance 等の MultiIndex 列を単一レベルにする。"""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _pick_series(work: pd.DataFrame, keys: tuple[str, ...]) -> pd.Series | None:
    """列名（大小文字不問）の優先順で 1 列だけ選ぶ。"""
    for want in keys:
        for col in work.columns:
            if str(col).strip().lower() == want:
                s = work[col]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                return pd.to_numeric(s, errors="coerce")
    return None


def normalize_ohlcv(df: pd.DataFrame, *, require_volume: bool = False) -> pd.DataFrame:
    """
    任意の OHLCV DataFrame を backtesting.py 用に正規化する。

    - 列名は大文字小文字を区別しない（open, OPEN, Open など可）
    - Close は 'close' を優先（'adj close' は close が無いときのみ）
    - 出力列: Open, High, Low, Close, Volume（Volume は無ければ 0）
    """
    if df is None or df.empty:
        raise ValueError("OHLCV データが空です。")

    work = _flatten_columns(df.copy())
    open_ = _pick_series(work, ("open", "o"))
    high = _pick_series(work, ("high", "h"))
    low = _pick_series(work, ("low", "l"))
    close = _pick_series(work, ("close", "c"))
    if close is None:
        close = _pick_series(work, ("adj close", "adj_close"))
    volume = _pick_series(work, ("volume", "v"))

    if any(s is None for s in (open_, high, low, close)):
        raise ValueError(
            "Open/High/Low/Close に該当する列が見つかりません。"
            f" 現在の列: {list(work.columns)}"
        )

    ohlcv = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close},
        index=work.index,
    )

    if volume is not None:
        ohlcv[OPTIONAL_COL] = volume.fillna(0)
    else:
        ohlcv[OPTIONAL_COL] = 0.0

    ohlcv = ohlcv.dropna(subset=list(REQUIRED_COLS))
    if ohlcv.empty:
        raise ValueError("有効な OHLCV 行がありません。")

    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        try:
            ohlcv.index = pd.to_datetime(ohlcv.index)
        except (TypeError, ValueError):
            pass

    ohlcv = ohlcv.sort_index()
    if require_volume and (ohlcv[OPTIONAL_COL] == 0).all():
        raise ValueError("Volume 列がすべて 0 です。")

    return ohlcv


# ---------------------------------------------------------------------------
# ドンチャン・チャネルブレイクアウト戦略
# ---------------------------------------------------------------------------


def _rolling_max_shifted(values: np.ndarray, period: int) -> np.ndarray:
    """前日までの period 日間の最大値（shift(1) で先読み防止）。"""
    return (
        pd.Series(values, dtype=float)
        .rolling(period, min_periods=period)
        .max()
        .shift(1)
        .to_numpy()
    )


def _rolling_min_shifted(values: np.ndarray, period: int) -> np.ndarray:
    """前日までの period 日間の最小値。"""
    return (
        pd.Series(values, dtype=float)
        .rolling(period, min_periods=period)
        .min()
        .shift(1)
        .to_numpy()
    )


def _make_donchian_upper(period: int):
    def indicator(values: np.ndarray) -> np.ndarray:
        return _rolling_max_shifted(values, period)

    return indicator


def _make_donchian_lower(period: int):
    def indicator(values: np.ndarray) -> np.ndarray:
        return _rolling_min_shifted(values, period)

    return indicator


class DonchianChannelBreakout(Strategy):
    """
    チャネルブレイクアウト（ドンチャン）— ロングオンリー。

    - 終値 > 上限 → 買い（または保有継続）
    - 保有中に終値 < 下限 → 決済
    注文はデフォルトで翌足始値約定（backtesting.py 標準動作）。
    """

    channel_period: int = 20

    def init(self) -> None:
        period = int(self.channel_period)
        self.upper = self.I(_make_donchian_upper(period), self.data.High)
        self.lower = self.I(_make_donchian_lower(period), self.data.Low)

    def next(self) -> None:
        upper = self.upper[-1]
        lower = self.lower[-1]
        if np.isnan(upper) or np.isnan(lower):
            return

        price = self.data.Close[-1]
        if not self.position:
            if price > upper:
                self.buy()
        elif price < lower:
            self.position.close()


# ---------------------------------------------------------------------------
# バックテスト実行
# ---------------------------------------------------------------------------

# stats から UI 表示する主要キー（backtesting.py が算出）
DISPLAY_STAT_KEYS = [
    "Start",
    "End",
    "Duration",
    "Exposure Time [%]",
    "Equity Final [$]",
    "Return [%]",
    "Buy & Hold Return [%]",
    "Return (Ann.) [%]",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Max. Drawdown [%]",
    "# Trades",
    "Win Rate [%]",
    "Profit Factor",
    "Expectancy [%]",
    "Commissions [$]",
]


def run_backtest(
    ohlcv: pd.DataFrame,
    *,
    channel_period: int = 20,
    cash: float = 1_000_000,
    commission: float = 0.0,
    trade_on_close: bool = False,
    finalize_trades: bool = True,
) -> dict[str, Any]:
    """
    backtesting.py でバックテストを実行する。

    Parameters
    ----------
    ohlcv : DataFrame
        OHLCV（列名は大小文字不問）
    channel_period : int
        ドンチャン期間
    cash : float
        初期資金
    commission : float
        片道手数料率（例: 0.001 = 0.1%）。公式 Quick Start では 0.002 を例示
    trade_on_close : bool
        True ならシグナル足の終値で約定

    Returns
    -------
    dict
        ok, stats, metrics, equity_curve, trades, strategy, library
    """
    try:
        data = normalize_ohlcv(ohlcv)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    if len(data) < channel_period + 5:
        return {
            "ok": False,
            "error": f"データ本数が不足しています（{len(data)} 本 < 必要 {channel_period + 5} 本）。",
        }

    strategy_cls = type(
        "DonchianChannelBreakout",
        (DonchianChannelBreakout,),
        {"channel_period": channel_period},
    )

    bt = Backtest(
        data,
        strategy_cls,
        cash=cash,
        commission=commission,
        trade_on_close=trade_on_close,
        exclusive_orders=True,
        finalize_trades=finalize_trades,
    )

    try:
        stats = bt.run()
    except Exception as exc:
        return {"ok": False, "error": f"バックテスト実行エラー: {exc}"}

    equity = stats.get("_equity_curve")
    trades = stats.get("_trades")

    metrics: dict[str, Any] = {}
    for key in DISPLAY_STAT_KEYS:
        if key in stats.index:
            val = stats[key]
            if hasattr(val, "item"):
                try:
                    val = val.item()
                except ValueError:
                    pass
            metrics[key] = val

    equity_df = pd.DataFrame(equity) if equity is not None else pd.DataFrame()
    trades_df = pd.DataFrame(trades) if trades is not None and len(trades) else pd.DataFrame()

    initial = float(cash)
    final_equity = float(metrics.get("Equity Final [$]", initial))
    buy_hold_return = float(metrics.get("Buy & Hold Return [%]", 0.0))
    final_buy_hold = initial * (1 + buy_hold_return / 100.0)

    plot_df = pd.DataFrame()
    if not equity_df.empty and "Equity" in equity_df.columns:
        first_close = float(data["Close"].iloc[0])
        buy_hold_series = initial * (data["Close"] / first_close)
        plot_df = pd.DataFrame(
            {
                "equity": equity_df["Equity"],
                "buy_hold_equity": buy_hold_series.reindex(equity_df.index),
            },
            index=equity_df.index,
        )

    return {
        "ok": True,
        "stats": stats,
        "metrics": metrics,
        "equity_curve": equity_df,
        "plot_df": plot_df,
        "trades": trades_df,
        "backtest": bt,
        "initial_capital": initial,
        "final_strategy": final_equity,
        "final_buy_hold": final_buy_hold,
        "pnl_strategy": final_equity - initial,
        "return_strategy_pct": float(metrics.get("Return [%]", 0.0)),
        "return_buy_hold_pct": buy_hold_return,
        "max_drawdown_pct": float(metrics.get("Max. Drawdown [%]", 0.0)),
        "sharpe": float(metrics.get("Sharpe Ratio", 0.0) or 0.0),
        "win_rate_pct": float(metrics.get("Win Rate [%]", 0.0) or 0.0),
        "num_trades": int(metrics.get("# Trades", 0) or 0),
        "profit_factor": float(metrics.get("Profit Factor", 0.0) or 0.0),
        "library": {
            "name": LIBRARY_NAME,
            "docs": LIBRARY_DOCS,
            "github": LIBRARY_GITHUB,
        },
    }


def equity_curve_for_plot(result: dict[str, Any]) -> pd.DataFrame:
    """Plotly 用の Equity / 買い持ち曲線。"""
    if not result.get("ok"):
        return pd.DataFrame()
    plot = result.get("plot_df")
    return plot if plot is not None and not plot.empty else pd.DataFrame()
