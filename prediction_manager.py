"""予測履歴の保存、検証、成功率の集計を行う"""

from __future__ import annotations

import pandas as pd
from datetime import datetime, timedelta, timezone
from data_store import list_predictions, add_prediction, update_predictions
from market_data import fetch_live_close

def record_prediction(ticker: str, results: dict) -> None:
    """ai_predictor の結果を履歴に保存する"""
    # 短期方向予測の保存
    if results.get("direction", {}).get("ok"):
        d = results["direction"]
        horizon = d["horizon_days"]
        target_date = (datetime.now() + timedelta(days=horizon)).strftime("%Y-%m-%d")
        add_prediction(
            ticker=ticker,
            pred_type="direction",
            target_date=target_date,
            current_price=d.get("current_price", 0), # directionには含まれていない場合がある
            predicted_value=d["prob_up"], # 上昇確率を保存
            confidence=d["prob_up"] if d["prob_up"] > 50 else d["prob_down"]
        )
    
    # 数値予測の保存
    if results.get("price", {}).get("ok"):
        p = results["price"]
        horizon = p["forecast_days"]
        target_date = (datetime.now() + timedelta(days=horizon)).strftime("%Y-%m-%d")
        add_prediction(
            ticker=ticker,
            pred_type="price",
            target_date=target_date,
            current_price=p["current_price"],
            predicted_value=p["predicted_price"],
            confidence=abs(p["predicted_return_pct"])
        )

def validate_past_predictions() -> dict:
    """過去の予測を実際の株価で検証する"""
    preds = list_predictions()
    if not preds:
        return {"total": 0, "success_rate": 0}
    
    now_str = datetime.now().strftime("%Y-%m-%d")
    updated = False
    
    # 検証が必要な銘柄のリストを作成（APIリミッターを考慮）
    to_validate = [p for p in preds if not p["validated"] and p["target_date"] <= now_str]
    
    # 重複を排除して最新株価を取得
    tickers = list(set(p["ticker"] for p in to_validate))
    if not tickers:
        return _summarize_stats(preds)
    
    # APIリミッターを考慮し、一度に多くやりすぎない（最大10銘柄）
    tickers = tickers[:10]
    prices = {}
    for t in tickers:
        try:
            prices[t] = fetch_live_close(t)
        except:
            continue
            
    for p in preds:
        if not p["validated"] and p["target_date"] <= now_str and p["ticker"] in prices:
            actual = prices[p["ticker"]]
            p["actual_end_price"] = actual
            p["validated"] = True
            updated = True
            
            if p["type"] == "direction":
                # 上昇予測（prob_up > 50）で実際に上がったか
                is_up_pred = p["predicted"] > 50
                is_up_actual = actual > p["start_price"]
                p["is_correct"] = (is_up_pred == is_up_actual)
            elif p["type"] == "price":
                # 誤差10%以内なら正解とする
                error = abs(actual - p["predicted"]) / actual
                p["is_correct"] = (error <= 0.10)
                
    if updated:
        update_predictions(preds)
        
    return _summarize_stats(preds)

def _summarize_stats(preds: list[dict]) -> dict:
    validated = [p for p in preds if p["validated"]]
    if not validated:
        return {"total": 0, "validated_count": 0, "success_rate": 0}
    
    correct = [p for p in validated if p["is_correct"]]
    
    # タイプ別の統計
    direction_preds = [p for p in validated if p["type"] == "direction"]
    price_preds = [p for p in validated if p["type"] == "price"]
    
    dir_success = len([p for p in direction_preds if p["is_correct"]]) / len(direction_preds) if direction_preds else 0
    price_success = len([p for p in price_preds if p["is_correct"]]) / len(price_preds) if price_preds else 0
    
    return {
        "total": len(preds),
        "validated_count": len(validated),
        "correct_count": len(correct),
        "success_rate": len(correct) / len(validated) * 100,
        "direction_success_rate": dir_success * 100,
        "price_success_rate": price_success * 100,
        "history": validated[-20:] # 直近20件
    }
