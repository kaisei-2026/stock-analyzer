"""AIエージェントによる銘柄発掘、自動分析、仮想売買ロジック"""

from __future__ import annotations

import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from data_store import load_ai_agent, save_ai_agent, reset_ai_agent, add_knowledge, get_watch_tickers_list
from ai_predictor import run_all_predictions
from market_data import fetch_live_close
from prediction_manager import record_prediction

def run_ai_agent_cycle(force_run: bool = False) -> dict:
    """AIエージェントの1サイクル（発掘・分析・売買）を実行する
    
    Args:
        force_run: Trueの場合、APIリミッターを無視して実行（初回用）
    """
    agent = load_ai_agent()
    
    # APIリミッター: 1時間に1回以上の実行を制限（force_run=Trueの場合は無視）
    if not force_run:
        now = datetime.now(timezone.utc)
        if agent["last_run"]:
            last_run = datetime.fromisoformat(agent["last_run"].replace(" UTC", ""))
            last_run = last_run.replace(tzinfo=timezone.utc)
            if (now - last_run).total_seconds() < 3600:
                return {"ok": False, "message": "APIリミッター作動中。次回の実行までお待ちください。"}

    # 1. 銘柄発掘
    # ユーザーが設定した監視銘柄があればそれを優先、なければデフォルト銘柄を使用
    user_tickers = get_watch_tickers_list()
    if user_tickers:
        hot_tickers = user_tickers
    else:
        # デフォルト銘柄（低価格帯を中心に、将来的に高価格帯も学習）
        # 1,000円～2,000円台の低価格帯銘柄を優先
        hot_tickers = [
            "8306.T",  # 三菱UFJ（低価格帯）
            "1605.T",  # INPEX（低価格帯）
            "1332.T",  # 日本水物（低価格帯）
            "1925.T",  # 大林組（低価格帯）
            "1928.T",  # 佐田建設（低価格帯）
            "2914.T",  # JT（低価格帯）
            "4063.T",  # 信託（低価格帯）
            "7203.T",  # トヨタ（中価格帯）
            "6758.T",  # 索尼（中価格帯）
            "8035.T",  # 東急（中価格帯）
            "9984.T",  # ソフトバンクG（高価格帯・学習用）
            "9101.T",  # 日本貨物（高価格帯・学習用）
        ]
    
    log_entry = {"time": now.isoformat(), "actions": []}
    
    for ticker in hot_tickers:
        # 2. 分析
        try:
            ohlcv = yf.download(ticker, period="6mo", progress=False)
            if ohlcv.empty: continue
            ohlcv.columns = ohlcv.columns.droplevel(1)
            
            results = run_all_predictions(ohlcv)
            # 予測結果を学習データとして記録
            results["direction"]["current_price"] = fetch_live_close(ticker) # 記録用に価格をセット
            record_prediction(ticker, results)
            
            buy_score = results["buy_score"]["score"]
            current_price = results["direction"]["current_price"]
            
            # 3. 売買判断
            # スコア80以上なら強い購入検討
            if buy_score >= 80 and ticker not in agent["positions"]:
                # 資金の20%までを1銘柄に投入
                budget = agent["cash"] * 0.20
                shares = int(budget // current_price)
                if shares >= 1:
                    cost = shares * current_price
                    agent["cash"] -= cost
                    agent["positions"][ticker] = {
                        "shares": shares,
                        "entry_price": current_price,
                        "entry_date": now.isoformat(),
                        "entry_score": buy_score
                    }
                    action = f"BUY {ticker}: {shares}株 @ {current_price:,.1f}円 (スコア: {buy_score:.1f})"
                    log_entry["actions"].append(action)
                    agent["history"].insert(0, {"time": now.isoformat(), "action": action, "ticker": ticker, "side": "BUY", "price": current_price, "shares": shares})
            
            # スコア45以下なら売却検討
            elif buy_score <= 45 and ticker in agent["positions"]:
                pos = agent["positions"][ticker]
                proceeds = pos["shares"] * current_price
                agent["cash"] += proceeds
                profit = proceeds - (pos["shares"] * pos["entry_price"])
                profit_pct = (profit / (pos["shares"] * pos["entry_price"])) * 100 if pos["shares"] * pos["entry_price"] > 0 else 0
                action = f"SELL {ticker}: {pos['shares']}株 @ {current_price:,.1f}円 (損益: {profit:+,.0f}円, スコア: {buy_score:.1f})"
                log_entry["actions"].append(action)
                agent["history"].insert(0, {"time": now.isoformat(), "action": action, "ticker": ticker, "side": "SELL", "price": current_price, "shares": pos["shares"], "profit": profit})
                del agent["positions"][ticker]
            
            # 何もしない場合もログに残す（アクションがない場合のみ）
            else:
                pass
                
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

    # 資産推移を記録
    total_value = agent["cash"]
    positions_value = 0.0
    for t, p in agent["positions"].items():
        try:
            current_price = fetch_live_close(t)
            positions_value += p["shares"] * current_price
        except:
            positions_value += p["shares"] * p["entry_price"]
    total_value += positions_value
    
    portfolio_record = {
        "date": now.isoformat(),
        "total_value": total_value,
        "cash": agent["cash"],
        "positions_value": positions_value,
        "position_count": len(agent["positions"])
    }
    agent["portfolio_history"].append(portfolio_record)
    if len(agent["portfolio_history"]) > 200: agent["portfolio_history"].pop(0)  # 最新200件を保持
    
    agent["last_run"] = now.strftime("%Y-%m-%d %H:%M UTC")
    agent["learning_log"].append(log_entry)
    if len(agent["learning_log"]) > 50: agent["learning_log"].pop(0)
    
    save_ai_agent(agent)
    return {"ok": True, "agent": agent}

def get_candidate_stocks() -> list:
    """購入検討中の銘柄リストを取得（スコア75以上だが未購入の銘柄）"""
    agent = load_ai_agent()
    candidates = []
    
    # デフォルト銘柄またはユーザー設定銘柄をスキャン
    user_tickers = get_watch_tickers_list()
    if user_tickers:
        hot_tickers = user_tickers
    else:
        hot_tickers = [
            "8306.T", "1605.T", "1332.T", "1925.T", "1928.T", "2914.T", "4063.T",
            "7203.T", "6758.T", "8035.T", "9984.T", "9101.T",
        ]
    
    for ticker in hot_tickers:
        if ticker in agent["positions"]:
            continue  # 既に保有している銘柄は除外
        
        try:
            ohlcv = yf.download(ticker, period="6mo", progress=False)
            if ohlcv.empty:
                continue
            ohlcv.columns = ohlcv.columns.droplevel(1)
            
            results = run_all_predictions(ohlcv)
            buy_score = results["buy_score"]["score"]
            
            # スコア75以上の銘柄を候補に追加
            if buy_score >= 75:
                current_price = fetch_live_close(ticker)
                candidates.append({
                    "ticker": ticker,
                    "name": ticker,  # 実際にはティッカーシンボルを表示
                    "buy_score": buy_score,
                    "current_price": current_price,
                    "reason": f"買いスコア: {buy_score:.1f}点"
                })
        except:
            pass
    
    # スコアの高い順にソート
    candidates.sort(key=lambda x: x["buy_score"], reverse=True)
    return candidates[:10]  # 上位10件を返す

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
        "current_cash": agent["cash"],
        "total_value": total_value,
        "return_pct": (total_value - agent["initial_cash"]) / agent["initial_cash"] * 100,
        "position_count": len(agent["positions"]),
        "reset_count": agent.get("reset_count", 0),
        "history": agent["history"][:10],
        "positions": agent["positions"],
        "portfolio_history": agent.get("portfolio_history", []),
        "last_run": agent.get("last_run"),
        "initial_cash": agent.get("initial_cash", 300000),
        "raw_data": agent
    }
