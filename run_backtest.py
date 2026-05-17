#!/usr/bin/env python3
"""
CLI: OHLCV バックテスト（backtesting.py）

使用例:
  python run_backtest.py --ticker 7203.T --period 1y --channel 20
  python run_backtest.py --csv data.csv --cash 1000000 --commission 0.001
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
import yfinance as yf

from backtest_engine import LIBRARY_DOCS, LIBRARY_GITHUB, run_backtest


def load_ohlcv_from_yfinance(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
    if df.empty:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Donchian channel breakout backtest (backtesting.py)")
    parser.add_argument("--ticker", default="7203.T", help="Yahoo Finance ticker")
    parser.add_argument("--csv", help="OHLCV CSV path (columns: open/high/low/close/volume, any case)")
    parser.add_argument("--period", default="2y", help="yfinance period when using --ticker")
    parser.add_argument("--channel", type=int, default=20, help="Donchian channel period")
    parser.add_argument("--cash", type=float, default=1_000_000, help="Initial cash")
    parser.add_argument("--commission", type=float, default=0.0, help="Commission rate per trade side")
    args = parser.parse_args()

    if args.csv:
        raw = pd.read_csv(args.csv, index_col=0, parse_dates=True)
    else:
        print(f"Fetching {args.ticker} ({args.period})…")
        raw = load_ohlcv_from_yfinance(args.ticker, args.period)

    result = run_backtest(
        raw,
        channel_period=args.channel,
        cash=args.cash,
        commission=args.commission,
    )

    if not result["ok"]:
        print("ERROR:", result.get("error"), file=sys.stderr)
        return 1

    print(f"\nEngine: {result['library']['name']}")
    print(f"  Docs:   {LIBRARY_DOCS}")
    print(f"  GitHub: {LIBRARY_GITHUB}\n")
    print("--- Key metrics ---")
    for key, val in result["metrics"].items():
        print(f"  {key}: {val}")

    trades = result["trades"]
    if not trades.empty:
        print(f"\n--- Last 5 trades (of {len(trades)}) ---")
        print(trades.tail().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
