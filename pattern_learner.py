"""
パターン学習エンジン — 自動知見化

やること:
1. 過去データから「上がる前の形」「下がる前の形」を自動で見つける
2. 今のチャートと似たパターンを過去から検索
3. 「過去○回中○回上昇（勝率○%）/ 平均リターン○%」を算出
4. 重要パターンを自動で knowledge.json に保存

パターンの「形」= 直近N日間の騰落率の正規化した形状（コサイン類似度で比較）
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _normalize_shape(arr: np.ndarray) -> np.ndarray:
    """ゼロ平均・単位分散に正規化（形の比較用）。"""
    std = arr.std()
    if std < 1e-8:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _pct_change_shape(prices: np.ndarray) -> np.ndarray:
    """価格系列 → 日次騰落率の形状ベクトル。"""
    if len(prices) < 2:
        return np.array([])
    pct = np.diff(prices) / prices[:-1]
    return _normalize_shape(pct)


# ---------------------------------------------------------------------------
# パターンスキャン
# ---------------------------------------------------------------------------

def scan_patterns(
    ohlcv: pd.DataFrame,
    *,
    window: int = 10,
    horizon: int = 5,
    top_n: int = 20,
    similarity_threshold: float = 0.75,
) -> dict[str, Any]:
    """
    過去の類似パターンをスキャンし、統計を返す。

    Parameters
    ----------
    ohlcv            : OHLCV DataFrame（Close 必須）
    window           : 比較する形の長さ（日数）
    horizon          : 形の後、何日後のリターンを見るか
    top_n            : 類似上位N件を使って統計を出す
    similarity_threshold : この値以上のコサイン類似度を「似ている」とする

    Returns
    -------
    dict
        ok, win_rate_pct, avg_return_pct, matches, current_shape_label,
        up_count, down_count, total_matches, patterns_detail
    """
    closes = ohlcv["Close"].values.astype(float)

    if len(closes) < window + horizon + 10:
        return {
            "ok": False,
            "error": f"データ不足（{len(closes)}本 / 必要 {window + horizon + 10}本以上）",
        }

    # 現在の形（最新 window 日間）
    current_prices = closes[-(window + 1):]
    current_shape  = _pct_change_shape(current_prices)

    # 過去の全ウィンドウをスキャン（現在を除く）
    matches: list[dict] = []
    scan_end = len(closes) - window - horizon  # 未来に十分なデータがある範囲
    dates = ohlcv.index

    for i in range(window, scan_end):
        past_prices = closes[i - window: i + 1]
        past_shape  = _pct_change_shape(past_prices)
        sim = _cosine_similarity(current_shape, past_shape)

        if sim < similarity_threshold:
            continue

        # horizon 日後のリターン
        future_return = (closes[i + horizon] - closes[i]) / closes[i] * 100.0

        matches.append({
            "date": str(dates[i])[:10],
            "similarity": round(sim, 4),
            "entry_price": round(float(closes[i]), 2),
            "exit_price":  round(float(closes[i + horizon]), 2),
            "return_pct":  round(future_return, 2),
            "direction":   "📈 上昇" if future_return > 0 else "📉 下落",
        })

    if not matches:
        return {
            "ok": True,
            "win_rate_pct": None,
            "avg_return_pct": None,
            "total_matches": 0,
            "up_count": 0,
            "down_count": 0,
            "matches": [],
            "patterns_detail": [],
            "note": f"類似度 {similarity_threshold:.0%} 以上のパターンが見つかりませんでした。閾値を下げてみてください。",
        }

    # 類似度上位 top_n に絞る
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    top_matches = matches[:top_n]

    returns = np.array([m["return_pct"] for m in top_matches])
    up_count   = int((returns > 0).sum())
    down_count = int((returns <= 0).sum())
    win_rate   = up_count / len(returns) * 100
    avg_return = float(returns.mean())
    median_ret = float(np.median(returns))
    max_gain   = float(returns.max())
    max_loss   = float(returns.min())

    # 騰落分布（ヒストグラム用）
    bins = [-20, -10, -5, -3, -1, 0, 1, 3, 5, 10, 20]
    hist_counts, hist_edges = np.histogram(returns, bins=bins)
    distribution = [
        {"range": f"{hist_edges[i]:+.0f}〜{hist_edges[i+1]:+.0f}%", "count": int(hist_counts[i])}
        for i in range(len(hist_counts))
        if hist_counts[i] > 0
    ]

    return {
        "ok": True,
        "win_rate_pct": round(win_rate, 1),
        "avg_return_pct": round(avg_return, 2),
        "median_return_pct": round(median_ret, 2),
        "max_gain_pct": round(max_gain, 2),
        "max_loss_pct": round(max_loss, 2),
        "total_matches": len(top_matches),
        "up_count": up_count,
        "down_count": down_count,
        "matches": top_matches,
        "distribution": distribution,
        "window": window,
        "horizon": horizon,
        "note": f"過去の類似パターン上位{len(top_matches)}件の統計。将来を保証しません。",
    }


# ---------------------------------------------------------------------------
# 相場フェーズ検出
# ---------------------------------------------------------------------------

def detect_market_phase(ohlcv: pd.DataFrame) -> dict[str, Any]:
    """
    現在の相場フェーズを検出する。

    フェーズ:
    - 上昇トレンド   : SMA短期 > SMA中期 > SMA長期
    - 下降トレンド   : SMA短期 < SMA中期 < SMA長期
    - レンジ（もみ合い）: それ以外
    - 急騰           : 直近5日で+5%以上
    - 急落           : 直近5日で-5%以下
    """
    closes = ohlcv["Close"]
    if len(closes) < 60:
        return {"ok": False, "error": "データ不足"}

    sma5  = float(closes.rolling(5).mean().iloc[-1])
    sma20 = float(closes.rolling(20).mean().iloc[-1])
    sma60 = float(closes.rolling(60).mean().iloc[-1])
    c     = float(closes.iloc[-1])
    ret5d = (c - float(closes.iloc[-6])) / float(closes.iloc[-6]) * 100

    if ret5d >= 5:
        phase = "🚀 急騰中"
        phase_key = "surge_up"
        desc = f"直近5日で {ret5d:+.1f}% の急騰。利確タイミングに注意。"
    elif ret5d <= -5:
        phase = "💥 急落中"
        phase_key = "surge_down"
        desc = f"直近5日で {ret5d:+.1f}% の急落。反発狙いか損切りか判断が必要。"
    elif sma5 > sma20 > sma60:
        phase = "📈 上昇トレンド"
        phase_key = "uptrend"
        desc = "短期・中期・長期すべての移動平均が上向き。トレンドフォロー有利。"
    elif sma5 < sma20 < sma60:
        phase = "📉 下降トレンド"
        phase_key = "downtrend"
        desc = "短期・中期・長期すべての移動平均が下向き。空売り or 様子見が基本。"
    elif sma20 > sma60 and sma5 < sma20:
        phase = "🔄 調整中（上昇トレンド内の押し目）"
        phase_key = "pullback"
        desc = "中長期は上昇トレンドだが短期が押し目。押し目買いのチャンスの可能性。"
    else:
        phase = "↔ レンジ（もみ合い）"
        phase_key = "range"
        desc = "方向感なし。ブレイクアウト待ちが基本戦略。"

    # 過去フェーズ別の成績を計算
    phase_stats = _calc_phase_stats(ohlcv, phase_key)

    return {
        "ok": True,
        "phase": phase,
        "phase_key": phase_key,
        "description": desc,
        "ret5d": round(ret5d, 2),
        "sma5": round(sma5, 2),
        "sma20": round(sma20, 2),
        "sma60": round(sma60, 2),
        "phase_stats": phase_stats,
    }


def _calc_phase_stats(ohlcv: pd.DataFrame, phase_key: str) -> dict:
    """同じフェーズだった過去の時期のリターン統計。"""
    closes = ohlcv["Close"].values.astype(float)
    n = len(closes)
    if n < 80:
        return {}

    horizon = 10  # 10日後のリターンを見る
    returns = []

    for i in range(60, n - horizon):
        seg = ohlcv["Close"].iloc[i-60:i]
        s5  = float(seg.rolling(5).mean().iloc[-1])
        s20 = float(seg.rolling(20).mean().iloc[-1])
        s60 = float(seg.rolling(60).mean().iloc[-1])
        r5  = (float(seg.iloc[-1]) - float(seg.iloc[-6])) / float(seg.iloc[-6]) * 100

        if phase_key == "surge_up"   and r5 >= 5:            in_phase = True
        elif phase_key == "surge_down" and r5 <= -5:          in_phase = True
        elif phase_key == "uptrend"  and s5 > s20 > s60:     in_phase = True
        elif phase_key == "downtrend" and s5 < s20 < s60:    in_phase = True
        elif phase_key == "pullback" and s20 > s60 and s5 < s20: in_phase = True
        elif phase_key == "range":
            in_phase = not (s5 > s20 > s60) and not (s5 < s20 < s60)
        else:
            in_phase = False

        if in_phase:
            ret = (closes[i + horizon] - closes[i]) / closes[i] * 100
            returns.append(ret)

    if not returns:
        return {}

    arr = np.array(returns)
    return {
        "sample_count": len(arr),
        "win_rate_pct": round(float((arr > 0).mean() * 100), 1),
        "avg_return_pct": round(float(arr.mean()), 2),
        "horizon_days": horizon,
    }


# ---------------------------------------------------------------------------
# 自動知見生成
# ---------------------------------------------------------------------------

def generate_auto_knowledge(
    ticker: str,
    ohlcv: pd.DataFrame,
    *,
    window: int = 10,
    horizon: int = 5,
) -> list[dict]:
    """
    パターンスキャン＋フェーズ検出の結果から
    knowledge.json に保存すべき知見リストを自動生成する。

    Returns
    -------
    list of dict (title, body, tags, ticker, source)
    """
    knowledge_items: list[dict] = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # --- パターン知見 ---
    pattern_result = scan_patterns(ohlcv, window=window, horizon=horizon)
    if pattern_result["ok"] and pattern_result["total_matches"] > 0:
        win  = pattern_result["win_rate_pct"]
        avg  = pattern_result["avg_return_pct"]
        n    = pattern_result["total_matches"]
        up   = pattern_result["up_count"]
        down = pattern_result["down_count"]

        # 勝率が顕著な場合だけ保存（60%以上 or 40%以下）
        if win is not None and (win >= 60 or win <= 40):
            direction_word = "上昇しやすい" if win >= 60 else "下落しやすい"
            body = (
                f"【自動学習 — パターン分析】\n"
                f"銘柄: {ticker} | 分析日: {today}\n\n"
                f"■ 現在の形と似たパターンが過去 {n} 回見つかりました\n"
                f"  ・上昇: {up}回 / 下落: {down}回\n"
                f"  ・勝率（{horizon}日後↑）: {win:.1f}%\n"
                f"  ・平均リターン: {avg:+.2f}%\n"
                f"  ・中央値リターン: {pattern_result['median_return_pct']:+.2f}%\n"
                f"  ・最大上昇: {pattern_result['max_gain_pct']:+.2f}% / 最大下落: {pattern_result['max_loss_pct']:+.2f}%\n\n"
                f"■ 解釈: 現在のチャートの形は、過去に{direction_word}傾向があります。\n"
                f"  参考情報として使い、必ず他の指標と組み合わせてください。\n\n"
                f"※ 統計的パターンであり、将来の利益を保証しません。"
            )
            knowledge_items.append({
                "title": f"【AI】{ticker} パターン勝率{win:.0f}% ({today})",
                "body": body,
                "tags": "AI自動,パターン,勝率",
                "ticker": ticker,
                "source": "AI自動学習",
            })

    # --- フェーズ知見 ---
    phase_result = detect_market_phase(ohlcv)
    if phase_result["ok"]:
        phase = phase_result["phase"]
        desc  = phase_result["description"]
        ps    = phase_result.get("phase_stats", {})

        body_lines = [
            f"【自動学習 — 相場フェーズ検出】",
            f"銘柄: {ticker} | 分析日: {today}",
            f"",
            f"■ 現在のフェーズ: {phase}",
            f"  {desc}",
            f"",
            f"■ 移動平均の状況:",
            f"  SMA5={phase_result['sma5']:,.1f} / SMA20={phase_result['sma20']:,.1f} / SMA60={phase_result['sma60']:,.1f}",
            f"  直近5日リターン: {phase_result['ret5d']:+.1f}%",
        ]

        if ps:
            body_lines += [
                f"",
                f"■ 過去に同じフェーズだったとき（{ps['sample_count']}回）:",
                f"  ・{ps['horizon_days']}日後の勝率: {ps['win_rate_pct']:.1f}%",
                f"  ・平均リターン: {ps['avg_return_pct']:+.2f}%",
            ]

        body_lines.append("\n※ 統計的パターンであり、将来の利益を保証しません。")

        knowledge_items.append({
            "title": f"【AI】{ticker} フェーズ: {phase} ({today})",
            "body": "\n".join(body_lines),
            "tags": "AI自動,フェーズ,トレンド",
            "ticker": ticker,
            "source": "AI自動学習",
        })

    # --- 上昇/下落タイミング統計 ---
    timing_knowledge = _generate_timing_knowledge(ticker, ohlcv, today)
    if timing_knowledge:
        knowledge_items.append(timing_knowledge)

    return knowledge_items


def _generate_timing_knowledge(
    ticker: str, ohlcv: pd.DataFrame, today: str
) -> dict | None:
    """
    「どのタイミングで上がった/下がったか」の統計知見を生成する。
    - ブレイクアウト後の平均リターン
    - RSI別のその後のリターン
    - 出来高急増後のリターン
    """
    closes = ohlcv["Close"].values.astype(float)
    n = len(closes)
    if n < 100:
        return None

    horizon = 5
    lines = [
        f"【自動学習 — 上昇・下落タイミング統計】",
        f"銘柄: {ticker} | 分析日: {today} | 集計期間: 全{n}日分\n",
    ]

    # --- 1. RSI別リターン ---
    closes_s = ohlcv["Close"]
    delta = closes_s.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    rsi_vals = rsi.values

    rsi_buckets = {
        "RSI<30（売られすぎ）": (rsi_vals < 30),
        "30≤RSI<50（やや弱め）": (rsi_vals >= 30) & (rsi_vals < 50),
        "50≤RSI<70（健全）": (rsi_vals >= 50) & (rsi_vals < 70),
        "RSI≥70（買われすぎ）": (rsi_vals >= 70),
    }

    lines.append("■ RSI水準別 その後5日間のリターン")
    for label, mask in rsi_buckets.items():
        idxs = np.where(mask)[0]
        idxs = idxs[(idxs + horizon) < n]
        if len(idxs) < 5:
            continue
        rets = np.array([(closes[i + horizon] - closes[i]) / closes[i] * 100 for i in idxs])
        win  = (rets > 0).mean() * 100
        avg  = rets.mean()
        lines.append(f"  {label}: {len(rets)}回 | 勝率{win:.0f}% | 平均{avg:+.1f}%")

    # --- 2. 出来高急増後のリターン ---
    if "Volume" in ohlcv.columns:
        vol = ohlcv["Volume"].values.astype(float)
        vol_ma20 = pd.Series(vol).rolling(20).mean().values
        vol_ratio = np.where(vol_ma20 > 0, vol / vol_ma20, 1.0)

        surge_idx = np.where(vol_ratio >= 2.0)[0]
        surge_idx = surge_idx[(surge_idx + horizon) < n]

        if len(surge_idx) >= 5:
            rets = np.array([(closes[i + horizon] - closes[i]) / closes[i] * 100 for i in surge_idx])
            win  = (rets > 0).mean() * 100
            avg  = rets.mean()
            lines.append(
                f"\n■ 出来高急増（平均2倍以上）後5日のリターン\n"
                f"  {len(rets)}回 | 勝率{win:.0f}% | 平均{avg:+.1f}%"
            )

    # --- 3. 月別平均リターン ---
    if hasattr(ohlcv.index, "month"):
        monthly_rets: dict[int, list] = {}
        for i in range(n - 1):
            m = ohlcv.index[i].month
            r = (closes[i + 1] - closes[i]) / closes[i] * 100
            monthly_rets.setdefault(m, []).append(r)

        lines.append("\n■ 月別 平均日次リターン（季節性）")
        month_names = ["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"]
        for m in range(1, 13):
            if m in monthly_rets and len(monthly_rets[m]) >= 5:
                avg_m = np.mean(monthly_rets[m])
                lines.append(f"  {month_names[m-1]}: 平均{avg_m:+.2f}%")

    lines.append("\n※ 統計的パターンであり、将来の利益を保証しません。")

    return {
        "title": f"【AI】{ticker} タイミング統計 ({today})",
        "body": "\n".join(lines),
        "tags": "AI自動,タイミング,RSI,出来高,季節性",
        "ticker": ticker,
        "source": "AI自動学習",
    }
