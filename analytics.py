import numpy as np
import pandas as pd
import yfinance as yf

def get_benchmark_ticker(ticker: str) -> str:
    upper = ticker.upper()
    if upper.endswith(".NS") or upper.endswith(".BO"):
        return "^NSEI"  # NIFTY 50
    return "^GSPC"      # S&P 500

def get_risk_free_rate(ticker: str) -> float:
    upper = ticker.upper()
    if upper.endswith(".NS") or upper.endswith(".BO"):
        return 0.070  # ~7.0% Indian 10-Yr G-Sec
    return 0.042      # ~4.2% US 10-Yr Treasury

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    if len(returns) < 2:
        return 0.0

    daily_rf = risk_free_rate / 252.0
    excess_returns = returns - daily_rf
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return 0.0

    downside_variance = np.mean(np.square(downside_returns))
    daily_downside_dev = np.sqrt(downside_variance)
    annualized_downside_dev = daily_downside_dev * np.sqrt(252)

    if annualized_downside_dev == 0 or np.isnan(annualized_downside_dev):
        return 0.0

    annualized_return = returns.mean() * 252
    sortino = (annualized_return - risk_free_rate) / annualized_downside_dev
    return round(float(sortino), 2)

def fetch_equity_analytics(ticker: str, period: str = "1y") -> dict:
    clean_ticker = ticker.strip().upper()
    benchmark_symbol = get_benchmark_ticker(clean_ticker)
    rf_rate = get_risk_free_rate(clean_ticker)

    asset_df = yf.download(clean_ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if asset_df.empty or len(asset_df) < 10:
        raise ValueError(f"No sufficient market data retrieved for {clean_ticker}")

    bench_df = yf.download(benchmark_symbol, period=period, interval="1d", auto_adjust=True, progress=False)

    if isinstance(asset_df.columns, pd.MultiIndex):
        asset_df.columns = asset_df.columns.get_level_values(0)
    if isinstance(bench_df.columns, pd.MultiIndex):
        bench_df.columns = bench_df.columns.get_level_values(0)

    candles = []
    for idx, row in asset_df.iterrows():
        candles.append({
            "time": idx.strftime("%Y-%m-%d"),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]) if "Volume" in row and not np.isnan(row["Volume"]) else 0
        })

    close_series = asset_df["Close"].dropna()
    log_returns = np.log(close_series / close_series.shift(1)).dropna()

    daily_vol = float(log_returns.std())
    annualized_vol = round(daily_vol * np.sqrt(252) * 100, 2)

    annualized_return = float(log_returns.mean() * 252)
    annualized_vol_dec = daily_vol * np.sqrt(252)
    sharpe = round((annualized_return - rf_rate) / annualized_vol_dec, 2) if annualized_vol_dec > 0 else 0.0
    sortino = calculate_sortino_ratio(log_returns, rf_rate)

    cumulative_series = np.exp(log_returns.cumsum())
    running_peak = cumulative_series.cummax()
    drawdown_series = (cumulative_series - running_peak) / running_peak
    max_drawdown = round(float(drawdown_series.min() * 100), 2)

    beta = 1.0
    if not bench_df.empty and len(bench_df) > 10:
        bench_close = bench_df["Close"].dropna()
        bench_returns = np.log(bench_close / bench_close.shift(1)).dropna()
        combined = pd.concat([log_returns, bench_returns], axis=1, join="inner").dropna()
        if len(combined) > 10:
            cov_matrix = np.cov(combined.iloc[:, 0], combined.iloc[:, 1])
            cov = cov_matrix[0, 1]
            bench_var = cov_matrix[1, 1]
            if bench_var > 0:
                beta = round(float(cov / bench_var), 2)

    current_price = round(float(close_series.iloc[-1]), 2)
    period_change = round(float(((close_series.iloc[-1] / close_series.iloc[0]) - 1) * 100), 2)

    return {
        "ticker": clean_ticker,
        "benchmark": benchmark_symbol,
        "current_price": current_price,
        "period_change_percent": period_change,
        "annualized_volatility_percent": annualized_vol,
        "beta": beta,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_percent": max_drawdown,
        "risk_free_rate_percent": round(rf_rate * 100, 2),
        "candles": candles
    }