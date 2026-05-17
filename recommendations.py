"""10万円前後から検討しやすい銘柄・ETF の候補（教育用・推奨売買ではない）"""

from __future__ import annotations

from dataclasses import dataclass

# 日本株は通常 100株単位。終値×100 がおおよその「1単元」必要資金。
LOT_SIZE = 100
DEFAULT_PLANNING_CASH = 100_000
DEFAULT_BACKTEST_CASH = 1_000  # 過去検証の見やすい基準額

# 流動性・知名度を重視した候補（株価は変動するためアプリ側で再計算）
@dataclass(frozen=True)
class Pick:
    ticker: str
    name: str
    category: str
    note: str


PICKS_FOR_SMALL_CAPITAL: tuple[Pick, ...] = (
    Pick("1306", "TOPIX連動型上場投信", "ETF", "日本全体に分散。コスト・流動性のバランスが良い定番。"),
    Pick("1321", "日経225連動型上場投信", "ETF", "日経平均に連動。ボラティリティはTOPIXよりやや大きめ。"),
    Pick("1557", "SPDR S&P500 ETF", "ETF", "米国大型株に分散。為替の影響あり。"),
    Pick("8306", "三菱UFJフィナンシャル・グループ", "銀行", "大型株・出来高が厚い。単元が10万円台になりやすい時期がある。"),
    Pick("9434", "ソフトバンク", "通信", "配当・話題性があり個人投資家にも人気。"),
    Pick("9101", "日本郵船", "海運", "景気・運賃感応度が高い。トレンドが出やすい。"),
    Pick("5411", "JFEホールディングス", "鉄鋼", "景気敏感。チャネルブレイクの練習向き。"),
    Pick("1605", "INPEX", "エネルギー", "資源価格と連動しやすい。"),
    Pick("6758", "ソニーグループ", "電機", "グローバル大手。流動性は高いが単元は10万超になりがち→要確認。"),
)

# Qiita 等（信頼できる技術記事のみ）
REFERENCE_LINKS: tuple[dict[str, str], ...] = (
    {
        "title": "【Python】Backtesting.pyで株売買のバックテスト・最適化",
        "url": "https://qiita.com/Fujinoinvestor/items/f2bdaabb766db443ddc0",
        "site": "Qiita",
    },
    {
        "title": "Pythonで株価バックテストツールを作った【米国・日本株対応】",
        "url": "https://qiita.com/jirachiuwu/items/80840e6bf4ee4cb5e7c8",
        "site": "Qiita",
    },
    {
        "title": "株価分析(SMA) - Backtestingを使ってリターンを計算",
        "url": "https://qiita.com/mahoutsukaino-deshi/items/8907a34986804c58724a",
        "site": "Qiita",
    },
    {
        "title": "TOPIX Core30銘柄に対してバックテスト実施・最適化",
        "url": "https://qiita.com/marumen/items/ec3f70115337b2d6bb4c",
        "site": "Qiita",
    },
    {
        "title": "backtesting.py 公式 Quick Start（英語・ライブラリ本体）",
        "url": "https://kernc.github.io/backtesting.py/doc/examples/Quick%20Start%20User%20Guide.html",
        "site": "公式ドキュメント",
    },
)


def unit_cost_yen(close_price: float, lot: int = LOT_SIZE) -> float:
    """1単元（通常100株）のおおよその必要資金。"""
    return float(close_price) * lot


def fits_capital(close_price: float, capital: float, lot: int = LOT_SIZE) -> bool:
    return unit_cost_yen(close_price, lot) <= capital


def affordability_label(close_price: float, capital: float) -> str:
    cost = unit_cost_yen(close_price)
    if cost <= capital:
        return f"✅ 1単元 約{cost:,.0f}円（資金内）"
    gap = cost - capital
    return f"⚠️ 1単元 約{cost:,.0f}円（あと{gap:,.0f}円不足）"
