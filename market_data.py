"""
Yahoo Finance 取得（キャッシュでレート制限を抑える）

yfinance は非公式 API のため、同一銘柄は最低 CACHE_TTL_SEC 秒間は再取得しません。
"""

from __future__ import annotations

import streamlit as st
import yfinance as yf

from backtest_engine import normalize_ohlcv

# 実運用で安全側に寄せた値（1分更新 × 保有10銘柄 ≒ 10 req/min）
CACHE_TTL_SEC = 60
RECOMMENDATIONS_TTL_SEC = 3600


def _normalize_symbol(ticker: str) -> str:
    t = ticker.strip().upper()
    if t.isdigit() and len(t) == 4:
        return f"{t}.T"
    if "." not in t and t.isalnum():
        return f"{t}.T" if t.isdigit() else t
    return t


@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner=False)
def fetch_live_close(ticker: str) -> float | None:
    """直近終値（キャッシュ TTL 既定 60 秒）。"""
    sym = _normalize_symbol(ticker)
    try:
        raw = yf.download(sym, period="5d", interval="1d", progress=False, auto_adjust=False)
        if raw.empty:
            raw = yf.Ticker(sym).history(period="5d", auto_adjust=False)
        ohlcv = normalize_ohlcv(raw)
        if ohlcv.empty:
            return None
        return float(ohlcv["Close"].iloc[-1])
    except Exception:
        return None


def fetch_live_closes(tickers: tuple[str, ...]) -> dict[str, float]:
    """複数銘柄の終値（銘柄ごとにキャッシュが効く）。"""
    prices: dict[str, float] = {}
    for tk in tickers:
        if not tk:
            continue
        sym = _normalize_symbol(tk)
        px = fetch_live_close(sym)
        if px is not None:
            prices[sym] = px
            prices[tk] = px
    return prices


@st.cache_data(ttl=RECOMMENDATIONS_TTL_SEC, show_spinner=False)
def fetch_recommendation_closes(tickers: tuple[str, ...]) -> dict[str, float]:
    """おすすめ一覧用（長めキャッシュで API 負荷を抑える）。"""
    return fetch_live_closes(tickers)
