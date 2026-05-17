"""ローカル JSON によるアイデア・デモ口座・知見の保存"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

DATA_DIR = Path(__file__).resolve().parent / "data"
IDEAS_FILE = DATA_DIR / "investment_ideas.json"
KNOWLEDGE_FILE = DATA_DIR / "knowledge.json"
DEMO_ACCOUNT_FILE = DATA_DIR / "demo_account.json"
CACHE_DIR = DATA_DIR / "ohlcv_cache"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default: Any) -> Any:
    _ensure_dirs()
    if not path.exists():
        return default
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _save_json(path: Path, data: Any) -> None:
    _ensure_dirs()
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# --- 投資アイデア ---


def list_ideas() -> list[dict]:
    return _load_json(IDEAS_FILE, [])


def add_idea(
    ticker: str,
    title: str,
    thesis: str,
    risk: str = "",
    status: str = "検討中",
) -> dict:
    item = {
        "id": str(uuid4())[:8],
        "created": _now_iso(),
        "ticker": ticker,
        "title": title,
        "thesis": thesis,
        "risk": risk,
        "status": status,
    }
    items = list_ideas()
    items.insert(0, item)
    _save_json(IDEAS_FILE, items)
    return item


def delete_idea(idea_id: str) -> None:
    items = [i for i in list_ideas() if i.get("id") != idea_id]
    _save_json(IDEAS_FILE, items)


# --- 知見 ---


def list_knowledge() -> list[dict]:
    return _load_json(KNOWLEDGE_FILE, [])


def add_knowledge(
    title: str,
    body: str,
    tags: str = "",
    ticker: str = "",
    source: str = "手動",
) -> dict:
    item = {
        "id": str(uuid4())[:8],
        "created": _now_iso(),
        "title": title,
        "body": body,
        "tags": tags,
        "ticker": ticker,
        "source": source,
    }
    items = list_knowledge()
    items.insert(0, item)
    _save_json(KNOWLEDGE_FILE, items)
    return item


def delete_knowledge(entry_id: str) -> None:
    items = [k for k in list_knowledge() if k.get("id") != entry_id]
    _save_json(KNOWLEDGE_FILE, items)


# --- デモ口座 ---


def default_demo_account(initial_cash: float = 100_000.0) -> dict:
    return {
        "cash": float(initial_cash),
        "initial_cash": float(initial_cash),
        "positions": {},
        "history": [],
        "updated": _now_iso(),
    }


def load_demo_account() -> dict:
    acc = _load_json(DEMO_ACCOUNT_FILE, None)
    if acc is None:
        acc = default_demo_account()
        save_demo_account(acc)
    return acc


def save_demo_account(account: dict) -> None:
    account["updated"] = _now_iso()
    _save_json(DEMO_ACCOUNT_FILE, account)


def reset_demo_account(initial_cash: float = 100_000.0) -> dict:
    acc = default_demo_account(initial_cash)
    save_demo_account(acc)
    return acc


def demo_buy(account: dict, ticker: str, shares: int, price: float) -> tuple[dict, str]:
    cost = shares * price
    if shares <= 0:
        return account, "株数は1以上にしてください。"
    if cost > account["cash"]:
        return account, f"資金不足（必要 {cost:,.0f} 円、残高 {account['cash']:,.0f} 円）"
    account["cash"] -= cost
    pos = account["positions"].get(ticker, {"shares": 0, "avg_price": 0.0})
    total_shares = pos["shares"] + shares
    pos["avg_price"] = (pos["avg_price"] * pos["shares"] + price * shares) / total_shares
    pos["shares"] = total_shares
    account["positions"][ticker] = pos
    account["history"].insert(
        0,
        {
            "time": _now_iso(),
            "side": "BUY",
            "ticker": ticker,
            "shares": shares,
            "price": price,
            "amount": cost,
        },
    )
    save_demo_account(account)
    return account, "デモ買いを記録しました。"


def demo_sell(account: dict, ticker: str, shares: int, price: float) -> tuple[dict, str]:
    pos = account["positions"].get(ticker)
    if not pos or pos["shares"] < shares:
        held = pos["shares"] if pos else 0
        return account, f"保有不足（保有 {held} 株）"
    proceeds = shares * price
    account["cash"] += proceeds
    pos["shares"] -= shares
    if pos["shares"] == 0:
        del account["positions"][ticker]
    else:
        account["positions"][ticker] = pos
    account["history"].insert(
        0,
        {
            "time": _now_iso(),
            "side": "SELL",
            "ticker": ticker,
            "shares": shares,
            "price": price,
            "amount": proceeds,
        },
    )
    save_demo_account(account)
    return account, "デモ売りを記録しました。"


def demo_portfolio_value(account: dict, prices: dict[str, float]) -> float:
    total = float(account["cash"])
    for ticker, pos in account["positions"].items():
        px = prices.get(ticker, pos.get("avg_price", 0.0))
        total += pos["shares"] * px
    return total


# --- OHLCV キャッシュ ---


def save_ohlcv_csv(ticker: str, ohlcv) -> Path:
    _ensure_dirs()
    safe = ticker.replace(".", "_")
    path = CACHE_DIR / f"{safe}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv"
    ohlcv.to_csv(path, encoding="utf-8-sig")
    return path


def list_cached_files() -> list[Path]:
    _ensure_dirs()
    return sorted(CACHE_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
