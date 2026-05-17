import yfinance as yf
import pandas as pd
from backtest_engine import run_backtest

ohlcv = yf.download('8306.T', period='1y', progress=False)
ohlcv.columns = ohlcv.columns.droplevel(1)
result = run_backtest(ohlcv, channel_period=20, cash=1_000_000, commission=0.0)
trades = result.get('trades')
if trades is not None and not trades.empty:
    print('ReturnPct sample:')
    print(trades['ReturnPct'].head())
    print('\nPnL sample:')
    print(trades['PnL'].head())
    print('\nEntryTime sample:')
    print(trades['EntryTime'].head())
    print('\nAll columns:', list(trades.columns))
else:
    print("No trades found")
    print("Result keys:", list(result.keys()))
