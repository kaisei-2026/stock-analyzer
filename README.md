# Breakout Trader（株価分析・バックテスト）

ローソク足チャートとドンチャン・チャネルブレイクアウト戦略の Streamlit アプリです。  
バックテストには [backtesting.py](https://github.com/kernc/backtesting.py) を使用しています。

## 起動方法（ローカル）

### 1. 前提

- Python 3.10 以上を推奨
- インターネット接続（Yahoo Finance から株価取得）

### 2. セットアップ

```powershell
cd c:\Users\81808\Documents\stock-app\stock-analyzer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Web アプリを起動

```powershell
streamlit run app.py
```

ブラウザが開き、`http://localhost:8501` で表示されます。開かない場合はターミナルに表示される URL をクリックしてください。

### 4. バックテストだけ CLI で実行

```powershell
python run_backtest.py --ticker 7203.T --period 2y --channel 20
python run_backtest.py --csv your_ohlcv.csv --cash 100000 --commission 0.1
```

## 資金の考え方（アプリ内）

| 用途 | デフォルト | 説明 |
|------|-----------|------|
| 過去バックテスト | **1,000円** | 成績の比較用。金額はリターン%の理解向け |
| これから投資する想定 | **100,000円** | 1単元が買えるかの目安・シミュレーション向け |

※ 10万円未満で1単元（100株）が買えない銘柄は、証券会社の「単元未満株」サービスを検討してください（SBI・楽天など公式ヘルプ参照）。

## Streamlit Cloud に載せる場合

1. GitHub にリポジトリを push
2. [Streamlit Cloud](https://share.streamlit.io/) で New app
3. Main file: `app.py`、Requirements: `requirements.txt`

## 参考（技術記事）

- [Backtesting.py でバックテスト（Qiita）](https://qiita.com/Fujinoinvestor/items/f2bdaabb766db443ddc0)
- [株価バックテストツール個人開発（Qiita）](https://qiita.com/jirachiuwu/items/80840e6bf4ee4cb5e7c8)

## 免責

本ツールは学習・検証用です。投資助言ではありません。損失のリスクは自己責任でください。
