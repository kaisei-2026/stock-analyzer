"""
AI 株価予測エンジン

3つの予測モードを提供:
1. 短期方向予測  : 数日〜1週間後の上/下をスコアで判定
2. 数値予測     : 1〜3ヶ月先の株価を線形回帰で予測
3. 買いスコア   : テクニカル指標を組み合わせた総合買いスコア(0〜100)

依存: numpy, pandas, scikit-learn（requirements.txt に追加済み）
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 特徴量エンジニアリング
# ---------------------------------------------------------------------------

def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=1).mean()

def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    fast = _ema(series, 12)
    slow = _ema(series, 26)
    macd_line = fast - slow
    signal = _ema(macd_line, 9)
    return macd_line, signal

def _bollinger(series: pd.Series, n: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = _sma(series, n)
    std = series.rolling(n, min_periods=1).std().fillna(0)
    return mid + 2 * std, mid, mid - 2 * std

def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """OHLCV から ML 特徴量を生成する。"""
    df = ohlcv.copy()
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"] if "Volume" in df.columns else pd.Series(0, index=df.index)

    # トレンド
    df["sma5"]  = _sma(c, 5)
    df["sma20"] = _sma(c, 20)
    df["sma60"] = _sma(c, 60)
    df["ema12"] = _ema(c, 12)
    df["ema26"] = _ema(c, 26)

    # モメンタム
    df["rsi14"] = _rsi(c, 14)
    df["macd"], df["macd_sig"] = _macd(c)
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    # ボラティリティ
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = _bollinger(c, 20)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)
    df["atr14"] = (h - l).rolling(14, min_periods=1).mean()

    # 出来高
    df["vol_sma20"] = _sma(v.astype(float), 20)
    df["vol_ratio"] = (v.astype(float) / df["vol_sma20"].replace(0, np.nan)).fillna(1.0)

    # リターン
    for lag in [1, 2, 3, 5, 10]:
        df[f"ret_{lag}d"] = c.pct_change(lag)

    # 位置
    df["pct_from_sma20"] = (c - df["sma20"]) / df["sma20"].replace(0, np.nan)
    df["pct_from_bb_upper"] = (c - df["bb_upper"]) / df["bb_upper"].replace(0, np.nan)

    return df


# ---------------------------------------------------------------------------
# ① 短期方向予測（上/下スコア）
# ---------------------------------------------------------------------------

def predict_direction(
    ohlcv: pd.DataFrame,
    horizon_days: int = 5,
    min_samples: int = 60,
) -> dict[str, Any]:
    """
    horizon_days 後の終値が現在より上か下かを予測する。

    Returns
    -------
    dict
        ok, prob_up, prob_down, signal, confidence, horizon_days, note
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"ok": False, "error": "scikit-learn が必要です。pip install scikit-learn"}

    df = build_features(ohlcv)
    df["target"] = (df["Close"].shift(-horizon_days) > df["Close"]).astype(int)
    df = df.dropna()

    if len(df) < min_samples:
        return {
            "ok": False,
            "error": f"データ不足（{len(df)} 行 / 必要 {min_samples} 行）",
        }

    feature_cols = [
        "rsi14", "macd_hist", "pct_from_sma20", "pct_from_bb_upper",
        "bb_width", "vol_ratio", "ret_1d", "ret_3d", "ret_5d", "ret_10d",
        "sma5", "sma20",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values
    y = df["target"].values

    # 最後の1行を予測用、残りを学習用
    X_train, y_train = X[:-1], y[:-1]
    X_pred = X[-1:].copy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_pred = scaler.transform(X_pred)

    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
    )
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_pred)[0]
    prob_up = float(prob[1]) * 100
    prob_down = float(prob[0]) * 100

    if prob_up >= 65:
        signal = "📈 上昇優勢"
        confidence = "高"
    elif prob_up >= 55:
        signal = "📈 やや上昇優勢"
        confidence = "中"
    elif prob_down >= 65:
        signal = "📉 下落優勢"
        confidence = "高"
    elif prob_down >= 55:
        signal = "📉 やや下落優勢"
        confidence = "中"
    else:
        signal = "➡ 方向感なし"
        confidence = "低"

    # 特徴量重要度 top5
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:5]
    top_features = [(feature_cols[i], float(importances[i])) for i in top_idx]

    return {
        "ok": True,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "signal": signal,
        "confidence": confidence,
        "horizon_days": horizon_days,
        "top_features": top_features,
        "train_samples": len(X_train),
        "note": "過去データの統計的パターン。将来を保証しません。",
    }


# ---------------------------------------------------------------------------
# ② 数値予測（将来株価）
# ---------------------------------------------------------------------------

def predict_price(
    ohlcv: pd.DataFrame,
    forecast_days: int = 20,
    min_samples: int = 60,
) -> dict[str, Any]:
    """
    forecast_days 後の株価を回帰モデルで予測する。

    Returns
    -------
    dict
        ok, current_price, predicted_price, predicted_return_pct,
        lower_bound, upper_bound, forecast_days, confidence_label
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"ok": False, "error": "scikit-learn が必要です。"}

    df = build_features(ohlcv)
    df["target"] = df["Close"].shift(-forecast_days)
    df = df.dropna()

    if len(df) < min_samples:
        return {
            "ok": False,
            "error": f"データ不足（{len(df)} 行 / 必要 {min_samples} 行）",
        }

    feature_cols = [
        "sma5", "sma20", "sma60", "ema12", "ema26",
        "rsi14", "macd", "macd_hist",
        "bb_width", "atr14", "vol_ratio",
        "ret_1d", "ret_3d", "ret_5d", "ret_10d",
        "pct_from_sma20",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values
    y = df["target"].values

    X_train, y_train = X[:-1], y[:-1]
    X_pred = X[-1:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_pred_s = scaler.transform(X_pred)

    model = GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42
    )
    model.fit(X_train_s, y_train)

    predicted = float(model.predict(X_pred_s)[0])
    current = float(ohlcv["Close"].iloc[-1])
    ret_pct = (predicted - current) / current * 100

    # 残差から信頼区間を推定
    train_preds = model.predict(X_train_s)
    residuals = y_train - train_preds
    std_res = float(np.std(residuals))
    lower = predicted - 1.64 * std_res   # 90%信頼区間
    upper = predicted + 1.64 * std_res

    if abs(ret_pct) < 3:
        confidence_label = "横ばい圏"
    elif ret_pct >= 10:
        confidence_label = "強い上昇予測"
    elif ret_pct >= 3:
        confidence_label = "上昇予測"
    elif ret_pct <= -10:
        confidence_label = "強い下落予測"
    else:
        confidence_label = "下落予測"

    return {
        "ok": True,
        "current_price": current,
        "predicted_price": predicted,
        "predicted_return_pct": ret_pct,
        "lower_bound": lower,
        "upper_bound": upper,
        "forecast_days": forecast_days,
        "confidence_label": confidence_label,
        "note": "統計モデルによる参考値。実際の株価は保証しません。",
    }


# ---------------------------------------------------------------------------
# ③ 買いスコア（0〜100点）
# ---------------------------------------------------------------------------

def calculate_buy_score(ohlcv: pd.DataFrame) -> dict[str, Any]:
    """
    複数のテクニカル指標を組み合わせ、総合的な「買いスコア」を算出する。

    Returns
    -------
    dict
        ok, score(0-100), grade, signals, details
    """
    if ohlcv is None or len(ohlcv) < 30:
        return {"ok": False, "error": "データ不足（30行以上必要）"}

    df = build_features(ohlcv)
    last = df.iloc[-1]
    c = float(last["Close"])

    signals: list[dict] = []
    score = 0.0
    max_score = 0.0

    def add(name: str, weight: float, bullish: bool, detail: str) -> None:
        nonlocal score, max_score
        max_score += weight
        if bullish:
            score += weight
        signals.append({
            "name": name,
            "bullish": bullish,
            "weight": weight,
            "detail": detail,
            "icon": "✅" if bullish else "❌",
        })

    # --- トレンド系 ---
    sma20 = float(last.get("sma20", c))
    sma60 = float(last.get("sma60", c))
    sma5  = float(last.get("sma5", c))

    add("短期トレンド（終値>SMA5）", 10,
        c > sma5,
        f"終値 {c:,.1f} vs SMA5 {sma5:,.1f}")

    add("中期トレンド（終値>SMA20）", 15,
        c > sma20,
        f"終値 {c:,.1f} vs SMA20 {sma20:,.1f}")

    add("長期トレンド（SMA20>SMA60）", 15,
        sma20 > sma60,
        f"SMA20 {sma20:,.1f} vs SMA60 {sma60:,.1f}")

    # --- RSI ---
    rsi = float(last.get("rsi14", 50))
    if rsi < 30:
        add("RSI（売られすぎ反発期待）", 20, True,  f"RSI={rsi:.1f} 売られすぎ水準")
    elif rsi < 50:
        add("RSI（中立〜やや弱め）",     20, False, f"RSI={rsi:.1f} 50未満")
    elif rsi < 70:
        add("RSI（健全な上昇圏）",       20, True,  f"RSI={rsi:.1f} 健全水準")
    else:
        add("RSI（買われすぎ注意）",     20, False, f"RSI={rsi:.1f} 買われすぎ水準")

    # --- MACD ---
    macd_hist = float(last.get("macd_hist", 0))
    add("MACD ヒストグラム（正）", 15,
        macd_hist > 0,
        f"MACDヒスト={macd_hist:+.3f}")

    # --- ボリンジャーバンド ---
    bb_upper = float(last.get("bb_upper", c))
    bb_lower = float(last.get("bb_lower", c))
    bb_mid   = float(last.get("bb_mid", c))
    bb_pos = (c - bb_lower) / max(bb_upper - bb_lower, 1e-8)
    add("BB ポジション（中央〜上半分）", 10,
        0.3 <= bb_pos <= 0.85,
        f"BB位置={bb_pos*100:.0f}% (0%=下限, 100%=上限)")

    # --- 出来高 ---
    vol_ratio = float(last.get("vol_ratio", 1.0))
    add("出来高（平均以上）", 15,
        vol_ratio >= 1.0,
        f"出来高比={vol_ratio:.2f}倍（平均比）")

    normalized = (score / max_score * 100) if max_score > 0 else 0.0

    if normalized >= 75:
        grade = "🟢 強い買いシグナル"
    elif normalized >= 55:
        grade = "🟡 弱い買いシグナル"
    elif normalized >= 40:
        grade = "⚪ 中立"
    elif normalized >= 25:
        grade = "🟠 弱い売りシグナル"
    else:
        grade = "🔴 強い売りシグナル"

    return {
        "ok": True,
        "score": normalized,
        "grade": grade,
        "signals": signals,
        "rsi": rsi,
        "macd_hist": macd_hist,
        "bb_pos_pct": bb_pos * 100,
        "vol_ratio": vol_ratio,
        "note": "テクニカル指標の組み合わせ。投資判断の参考のみ。",
    }


# ---------------------------------------------------------------------------
# 全予測まとめて実行
# ---------------------------------------------------------------------------

def run_all_predictions(
    ohlcv: pd.DataFrame,
    *,
    direction_horizon: int = 5,
    price_forecast_days: int = 20,
) -> dict[str, Any]:
    """3つの予測をまとめて返す。"""
    direction = predict_direction(ohlcv, horizon_days=direction_horizon)
    price = predict_price(ohlcv, forecast_days=price_forecast_days)
    buy_score = calculate_buy_score(ohlcv)

    return {
        "direction": direction,
        "price": price,
        "buy_score": buy_score,
    }
