import numpy as np
import pandas as pd

def generate_educational_guide(metrics: dict, benchmark_name: str) -> list:
    guide = []

    # 1. Beta
    beta = metrics["beta_against_benchmark"]
    if beta > 1.2:
        guide.append({
            "topic": f"Market Pace (Beta: {beta})",
            "badge": "Aggressive Swings",
            "badge_color": "amber",
            "text": f"This stock swings wider than the {benchmark_name}. A 1% market move historically translates to roughly a {beta}% move for this stock."
        })
    elif beta < 0.8:
        guide.append({
            "topic": f"Market Pace (Beta: {beta})",
            "badge": "Calmer Mover",
            "badge_color": "emerald",
            "text": f"This stock is steadier than the {benchmark_name}. When broad markets panic, defensive stocks like this historically drop less."
        })
    else:
        guide.append({
            "topic": f"Market Pace (Beta: {beta})",
            "badge": "Market Benchmark",
            "badge_color": "sky",
            "text": f"This stock tends to mirror the pace of {benchmark_name} closely."
        })

    # 2. Sharpe Ratio
    sharpe = metrics["sharpe_ratio"]
    if sharpe >= 1.0:
        guide.append({
            "topic": f"Risk vs Reward (Sharpe: {sharpe})",
            "badge": "Solid Payoff",
            "badge_color": "emerald",
            "text": "The returns comfortably beat safe government bond rates (~6.5%) even after factoring in all price volatility."
        })
    else:
        guide.append({
            "topic": f"Risk vs Reward (Sharpe: {sharpe})",
            "badge": "Subdued Return",
            "badge_color": "rose" if sharpe < 0 else "amber",
            "text": "Returns were thin or negative compared to safe government cash deposits after accounting for the price swings endured."
        })

    # 3. Max Drawdown
    mdd = abs(metrics["max_drawdown_pct"])
    guide.append({
        "topic": f"Worst Historical Drop (-{mdd}%)",
        "badge": "Peak-to-Floor",
        "badge_color": "rose" if mdd > 20 else "sky",
        "text": f"The steepest continuous drop from its highest peak was {mdd}%. This shows the worst holding drop an investor had to endure."
    })

    # 4. Volume Activity
    vols = metrics.get("volume_series", [])
    if len(vols) >= 5:
        recent_vol = vols[-1]["value"]
        avg_vol = np.mean([v["value"] for v in vols[-20:]])
        vol_ratio = round(recent_vol / avg_vol, 1) if avg_vol > 0 else 1.0

        if vol_ratio > 1.5:
            guide.append({
                "topic": "Trading Turnover",
                "badge": f"{vol_ratio}x Above Normal",
                "badge_color": "emerald",
                "text": f"Trading volume was {vol_ratio}x higher than usual today, showing heavy institutional or retail participation."
            })
        else:
            guide.append({
                "topic": "Trading Turnover",
                "badge": "Normal Volume",
                "badge_color": "sky",
                "text": "Daily volume is around regular baseline levels without abnormal institutional spikes."
            })

    return guide


def compute_financial_metrics(stock_df: pd.DataFrame, benchmark_df: pd.DataFrame, benchmark_name: str, risk_free_rate: float = 0.065) -> dict:
    combined = pd.DataFrame({
        'stock_open': stock_df['Open'],
        'stock_high': stock_df['High'],
        'stock_low': stock_df['Low'],
        'stock_close': stock_df['Close'],
        'stock_volume': stock_df['Volume'],
        'bench_close': benchmark_df['Close']
    }).dropna()

    if len(combined) < 15:
        raise ValueError("Insufficient trading days to calculate statistics.")

    # 1. Log Returns
    combined['stock_ret'] = np.log(combined['stock_close'] / combined['stock_close'].shift(1))
    combined['bench_ret'] = np.log(combined['bench_close'] / combined['bench_close'].shift(1))
    returns = combined.dropna()

    stock_returns = returns['stock_ret']
    bench_returns = returns['bench_ret']

    # 2. Cumulative Return
    start_price = float(combined['stock_close'].iloc[0])
    latest_price = float(combined['stock_close'].iloc[-1])
    cumulative_return = (latest_price - start_price) / start_price

    # 3. Volatility
    daily_volatility = float(stock_returns.std())
    annualized_volatility = daily_volatility * np.sqrt(252)

    # 4. Beta
    covariance = np.cov(stock_returns, bench_returns)[0][1]
    bench_variance = np.var(bench_returns)
    beta = float(covariance / bench_variance) if bench_variance != 0 else 1.0

    # 5. Sharpe Ratio
    annualized_stock_return = float(stock_returns.mean()) * 252
    sharpe_ratio = (
        (annualized_stock_return - risk_free_rate) / annualized_volatility
        if annualized_volatility != 0 else 0.0
    )

    # 6. Max Drawdown
    peak = combined['stock_close'].cummax()
    drawdown = (combined['stock_close'] - peak) / peak
    max_drawdown = float(drawdown.min())

    # 7. Moving Averages
    combined['sma_15'] = combined['stock_close'].rolling(window=15).mean()
    combined['ema_15'] = combined['stock_close'].ewm(span=15, adjust=False).mean()

    # 8. Chart Series
    candlesticks = []
    sma_series = []
    ema_series = []
    volume_series = []

    for date, row in combined.tail(90).iterrows():
        date_str = date.strftime("%Y-%m-%d")
        o = round(float(row['stock_open']), 2)
        c = round(float(row['stock_close']), 2)
        h = round(float(row['stock_high']), 2)
        l = round(float(row['stock_low']), 2)
        v = int(row['stock_volume'])

        candlesticks.append({"time": date_str, "open": o, "high": h, "low": l, "close": c})
        vol_color = '#10b981' if c >= o else '#f43f5e'
        volume_series.append({"time": date_str, "value": v, "color": vol_color})

        if not np.isnan(row['sma_15']):
            sma_series.append({"time": date_str, "value": round(float(row['sma_15']), 2)})
        if not np.isnan(row['ema_15']):
            ema_series.append({"time": date_str, "value": round(float(row['ema_15']), 2)})

    core_metrics = {
        "current_price": round(latest_price, 2),
        "cumulative_return_pct": round(cumulative_return * 100, 2),
        "annualized_volatility_pct": round(annualized_volatility * 100, 2),
        "beta_against_benchmark": round(beta, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "candlesticks": candlesticks,
        "sma_15": sma_series,
        "ema_15": ema_series,
        "volume_series": volume_series
    }

    core_metrics["educational_guide"] = generate_educational_guide(core_metrics, benchmark_name)
    return core_metrics