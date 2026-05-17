"""AIエージェントによる銘柄発掘、自動分析、仮想売買ロジック"""

from __future__ import annotations

import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from data_store import load_ai_agent, save_ai_agent, reset_ai_agent, add_knowledge
from ai_predictor import run_all_predictions
from market_data import fetch_live_close

def run_ai_agent_cycle() -> dict:
    """AIエージェントの1サイクル（発掘・分析・売買）を実行する"""
    agent = load_ai_agent()
    
    # APIリミッター: 1時間に1回以上の実行を制限（手動実行は別）
    now = datetime.now(timezone.utc)
    if agent["last_run"]:
        last_run = datetime.fromisoformat(agent["last_run"].replace(" UTC", ""))
        last_run = last_run.replace(tzinfo=timezone.utc)
        if (now - last_run).total_seconds() < 3600:
            return {"ok": False, "message": "APIリミッター作動中。次回の実行までお待ちください。"}

    # 1. 銘柄発掘（本来は外部検索等を行うが、ここでは注目の日本株リストから選択）
    # ※ 将来的にはGoogle/Yahooニュースのスクレイピング結果をここに統合
    hot_tickers = ["8306.T", "7203.T", "9984.T", "6758.T", "4063.T", "8035.T", "9101.T"]
    
    log_entry = {"time": now.isoformat(), "actions": []}
    
    for ticker in hot_tickers:
        # 2. 分析
        try:
            ohlcv = yf.download(ticker, period="6mo", progress=False)
            if ohlcv.empty: continue
            ohlcv.columns = ohlcv.columns.droplevel(1)
            
            results = run_all_predictions(ohlcv)
            buy_score = results["buy_score"]["score"]
            current_price = fetch_live_close(ticker)
            
            # 3. 売買判断
            # スコア75以上なら購入検討
            if buy_score >= 75 and ticker not in agent["positions"]:
                # 資金の20%までを1銘柄に投入
                budget = agent["cash"] * 0.2
                shares = int(budget // current_price)
                if shares >= 100: # 日本株は100株単位
                    cost = shares * current_price
                    agent["cash"] -= cost
                    agent["positions"][ticker] = {
                        "shares": shares,
                        "entry_price": current_price,
                        "entry_date": now.isoformat()
                    }
                    action = f"BUY {ticker}: {shares}株 @ {current_price:,.1f}円"
                    log_entry["actions"].append(action)
                    agent["history"].insert(0, {"time": now.isoformat(), "action": action, "ticker": ticker})
            
            # スコア40以下なら売却検討
            elif buy_score <= 40 and ticker in agent["positions"]:
                pos = agent["positions"][ticker]
                proceeds = pos["shares"] * current_price
                agent["cash"] += proceeds
                profit = proceeds - (pos["shares"] * pos["entry_price"])
                action = f"SELL {ticker}: {pos['shares']}株 @ {current_price:,.1f}円 (損益: {profit:+,.0f}円)"
                log_entry["actions"].append(action)
                agent["history"].insert(0, {"time": now.isoformat(), "action": action, "ticker": ticker})
                del agent["positions"][ticker]
                
        except Exception as e:
            log_entry["actions"].append(f"Error analyzing {ticker}: {str(e)}")

    # 4. 破産チェック
    total_value = agent["cash"]
    for t, p in agent["positions"].items():
        try:
            total_value += p["shares"] * fetch_live_close(t)
        except:
            total_value += p["shares"] * p["entry_price"]
            
    if total_value < 10000 and not agent["positions"]: # ほぼ0円
        agent = reset_ai_agent(300000.0)
        log_entry["actions"].append("資産が底をついたため、リセットを実行しました。")

    agent["last_run"] = now.strftime("%Y-%m-%d %H:%M UTC")
    agent["learning_log"].append(log_entry)
    if len(agent["learning_log"]) > 50: agent["learning_log"].pop(0)
    
    save_ai_agent(agent)
    return {"ok": True, "agent": agent}

def get_ai_agent_stats() -> dict:
    agent = load_ai_agent()
    # 現在の評価額を計算
    total_value = agent["cash"]
    for t, p in agent["positions"].items():
        try:
            total_value += p["shares"] * fetch_live_close(t)
        except:
            total_value += p["shares"] * p["entry_price"]
            
    return {
        "cash": agent["cash"],
        "total_value": total_value,
        "return_pct": (total_value - agent["initial_cash"]) / agent["initial_cash"] * 100,
        "positions_count": len(agent["positions"]),
        "reset_count": agent.get("reset_count", 0),
        "history": agent["history"][:10],
        "positions": agent["positions"]
    }
