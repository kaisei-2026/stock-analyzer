"""
予測結果の保存・削除・追跡機能を提供するモジュール
ユーザーが保存した予測と実際の株価推移を比較できる
"""

from __future__ import annotations

import json
import os
from datetime import datetime
import yfinance as yf
import pandas as pd
from market_data import fetch_live_close

PREDICTIONS_FILE = os.path.expanduser("~/.stock_analyzer_predictions.json")

def load_predictions() -> list[dict]:
    """保存済みの予測結果を読み込む"""
    if not os.path.exists(PREDICTIONS_FILE):
        return []
    
    try:
        with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []

def save_predictions(predictions: list[dict]) -> None:
    """予測結果をファイルに保存"""
    with open(PREDICTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

def add_prediction(ticker: str, prediction_type: str, predicted_value: float, 
                   forecast_days: int, confidence: float = None) -> dict:
    """新しい予測を保存
    
    Args:
        ticker: 銘柄コード
        prediction_type: 予測タイプ（"price" または "direction"）
        predicted_value: 予測値（株価またはスコア）
        forecast_days: 予測期間（営業日）
        confidence: 信頼度（オプション）
    
    Returns:
        保存された予測のID
    """
    predictions = load_predictions()
    
    current_price = fetch_live_close(ticker)
    
    prediction = {
        "id": len(predictions) + 1,
        "ticker": ticker,
        "type": prediction_type,
        "predicted_value": predicted_value,
        "current_price": current_price,
        "forecast_days": forecast_days,
        "confidence": confidence,
        "saved_at": datetime.now().isoformat(),
        "target_date": None,  # 後で計算
        "actual_price": None,
        "result": None,  # "correct", "incorrect", "pending"
    }
    
    predictions.append(prediction)
    save_predictions(predictions)
    
    return {"ok": True, "id": prediction["id"], "message": "予測を保存しました"}

def delete_prediction(prediction_id: int) -> dict:
    """予測を削除"""
    predictions = load_predictions()
    predictions = [p for p in predictions if p["id"] != prediction_id]
    save_predictions(predictions)
    
    return {"ok": True, "message": "予測を削除しました"}

def get_prediction_by_id(prediction_id: int) -> dict | None:
    """IDで予測を取得"""
    predictions = load_predictions()
    for p in predictions:
        if p["id"] == prediction_id:
            return p
    return None

def get_predictions_by_ticker(ticker: str) -> list[dict]:
    """銘柄別の予測一覧を取得"""
    predictions = load_predictions()
    return [p for p in predictions if p["ticker"] == ticker]

def update_prediction_result(prediction_id: int) -> dict:
    """予測の実績を更新（実際の株価を取得して比較）"""
    predictions = load_predictions()
    prediction = None
    
    for p in predictions:
        if p["id"] == prediction_id:
            prediction = p
            break
    
    if not prediction:
        return {"ok": False, "message": "予測が見つかりません"}
    
    try:
        # 実際の株価を取得
        current_price = fetch_live_close(prediction["ticker"])
        prediction["actual_price"] = current_price
        
        # 結果を判定
        if prediction["type"] == "price":
            # 株価予測の場合
            predicted = prediction["predicted_value"]
            actual = current_price
            original = prediction["current_price"]
            
            # 予測方向が正しいかチェック
            predicted_direction = "up" if predicted > original else "down"
            actual_direction = "up" if actual > original else "down"
            
            prediction["result"] = "correct" if predicted_direction == actual_direction else "incorrect"
        
        prediction["checked_at"] = datetime.now().isoformat()
        save_predictions(predictions)
        
        return {"ok": True, "prediction": prediction}
    except Exception as e:
        return {"ok": False, "message": f"エラー: {str(e)}"}

def get_all_predictions() -> list[dict]:
    """すべての予測を取得"""
    return load_predictions()

def get_accuracy_stats() -> dict:
    """予測精度の統計情報を取得"""
    predictions = load_predictions()
    
    if not predictions:
        return {
            "total": 0,
            "correct": 0,
            "incorrect": 0,
            "pending": 0,
            "accuracy": 0.0
        }
    
    correct = sum(1 for p in predictions if p["result"] == "correct")
    incorrect = sum(1 for p in predictions if p["result"] == "incorrect")
    pending = sum(1 for p in predictions if p["result"] is None or p["result"] == "pending")
    total = len(predictions)
    
    accuracy = (correct / (correct + incorrect) * 100) if (correct + incorrect) > 0 else 0.0
    
    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "pending": pending,
        "accuracy": accuracy
    }

def get_prediction_history_chart(ticker: str, prediction_id: int) -> dict:
    """予測と実績の推移チャート用データを取得"""
    prediction = get_prediction_by_id(prediction_id)
    
    if not prediction:
        return {"ok": False, "message": "予測が見つかりません"}
    
    try:
        # 過去データを取得
        ohlcv = yf.download(ticker, period="3mo", progress=False)
        if isinstance(ohlcv.columns, pd.MultiIndex):
            ohlcv.columns = ohlcv.columns.droplevel(1)
        
        close_prices = ohlcv["Close"]
        dates = close_prices.index.tolist()
        prices = close_prices.values.tolist()
        
        # 予測情報
        saved_at = datetime.fromisoformat(prediction["saved_at"])
        
        return {
            "ok": True,
            "ticker": ticker,
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "prices": prices,
            "saved_date": saved_at.strftime("%Y-%m-%d"),
            "saved_price": prediction["current_price"],
            "predicted_value": prediction["predicted_value"],
            "forecast_days": prediction["forecast_days"],
            "actual_price": prediction["actual_price"],
            "result": prediction["result"]
        }
    except Exception as e:
        return {"ok": False, "message": f"エラー: {str(e)}"}
