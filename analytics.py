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

def extract_financial_statements(ticker_obj, current_price: float = 0.0) -> dict:
    fundamentals = {
        "multiples": {
            "pe_ratio": "--",
            "pb_ratio": "--",
            "roe": "--",
            "debt_to_equity": "--",
            "operating_margin": "--",
            "profit_margin": "--"
        },
        "annual_trend": []
    }

    try:
        # Load statement DataFrames with broad compatibility
        income = None
        for attr in ['income_stmt', 'financials', 'get_income_stmt']:
            try:
                res = getattr(ticker_obj, attr)
                income = res() if callable(res) else res
                if income is not None and not income.empty:
                    break
            except Exception:
                continue

        cashflow = None
        for attr in ['cash_flow', 'cashflow', 'get_cash_flow']:
            try:
                res = getattr(ticker_obj, attr)
                cashflow = res() if callable(res) else res
                if cashflow is not None and not cashflow.empty:
                    break
            except Exception:
                continue

        balance = None
        for attr in ['balance_sheet', 'get_balance_sheet']:
            try:
                res = getattr(ticker_obj, attr)
                balance = res() if callable(res) else res
                if balance is not None and not balance.empty:
                    break
            except Exception:
                continue

        # Helper to search index dynamically for fuzzy keyword matches
        def find_val(df, keywords, col):
            if df is None or df.empty:
                return 0.0
            for kw in keywords:
                for row_name in df.index:
                    if kw.lower() in str(row_name).lower():
                        val = df.loc[row_name, col]
                        if isinstance(val, (pd.Series, pd.DataFrame)):
                            val = val.iloc[0]
                        if pd.notnull(val) and not np.isnan(val):
                            return float(val)
            return 0.0

        if income is not None and not income.empty:
            years = list(income.columns[:3])
            ticker_str = str(getattr(ticker_obj, 'ticker', '')).upper()
            scale = 1e7 if ticker_str.endswith(('.NS', '.BO')) else 1e6

            for y in years:
                year_str = str(y.year) if hasattr(y, 'year') else str(y)[:4]

                rev = find_val(income, ["Total Revenue", "Operating Revenue", "Revenue"], y)
                op_income = find_val(income, ["Operating Income", "Operating Profit", "EBIT"], y)
                
                # Broadened fuzzy tags for Indian PAT and Net Income
                pat = find_val(income, [
                    "Net Income Common",
                    "Net Income Continuous",
                    "Net Income Including Noncontrolling",
                    "Normalized Income",
                    "Net Income",
                    "PAT"
                ], y)

                # Broadened tags for Operating Cash Flow & CapEx
                cfo = find_val(cashflow, [
                    "Operating Cash Flow",
                    "Cash Flow From Continuing Operating",
                    "Cash Flow From Operating",
                    "Total Cash From Operating",
                    "Cash Provided By Operating"
                ], y)

                capex = find_val(cashflow, [
                    "Capital Expenditure",
                    "Capital Expenditures",
                    "Purchase Of Property Plant And Equipment",
                    "Net PPE Purchase"
                ], y)

                fcf = (cfo + capex) if (cfo != 0.0 or capex != 0.0) else 0.0

                fundamentals["annual_trend"].append({
                    "year": year_str,
                    "revenue": round(rev / scale, 2),
                    "operating_income": round(op_income / scale, 2),
                    "net_income": round(pat / scale, 2),
                    "cfo": round(cfo / scale, 2),
                    "fcf": round(fcf / scale, 2)
                })

            # Multiples direct math calculation from latest fiscal column
            latest_col = income.columns[0]
            latest_pat = find_val(income, ["Net Income Common", "Net Income Continuous", "Normalized Income", "Net Income"], latest_col)
            latest_rev = find_val(income, ["Total Revenue", "Operating Revenue", "Revenue"], latest_col)
            latest_op = find_val(income, ["Operating Income", "Operating Profit", "EBIT"], latest_col)

            equity = find_val(balance, [
                "Stockholders Equity", 
                "Common Stock Equity", 
                "Total Equity", 
                "Total Stockholder Equity"
            ], latest_col)
            
            total_debt = find_val(balance, [
                "Total Debt", 
                "Long Term Debt", 
                "Current Debt",
                "Total Liabilities Net Minority"
            ], latest_col)

            # Try info first
            info = {}
            try:
                info = ticker_obj.info or {}
            except Exception:
                info = {}

            pe = info.get("trailingPE") or info.get("forwardPE")
            pb = info.get("priceToBook")
            roe = info.get("returnOnEquity")
            de = info.get("debtToEquity")

            # Fast Info / Shares outstanding to derive Market Cap
            shares = getattr(getattr(ticker_obj, 'fast_info', None), 'shares', None) or info.get("sharesOutstanding") or 0.0
            market_cap = (shares * current_price) if (shares > 0 and current_price > 0) else (getattr(getattr(ticker_obj, 'fast_info', None), 'market_cap', 0.0) or info.get("marketCap") or 0.0)

            # Mathematical Fallback
            if (not pe or pe == 0) and market_cap > 0 and latest_pat > 0:
                pe = round(market_cap / latest_pat, 2)
            if (not pb or pb == 0) and market_cap > 0 and equity > 0:
                pb = round(market_cap / equity, 2)
            if (not roe or roe == 0) and equity > 0 and latest_pat != 0:
                roe = round((latest_pat / equity) * 100, 2)
            elif roe:
                roe = round(roe * 100, 2) if roe < 2.0 else round(roe, 2)

            if (not de or de == 0) and equity > 0 and total_debt > 0:
                de = round(total_debt / equity, 2)

            fundamentals["multiples"] = {
                "pe_ratio": round(float(pe), 2) if (pe and pe > 0) else "--",
                "pb_ratio": round(float(pb), 2) if (pb and pb > 0) else "--",
                "roe": round(float(roe), 2) if (roe and roe != 0) else "--",
                "debt_to_equity": round(float(de), 2) if (de and de > 0) else "--",
                "operating_margin": round((latest_op / latest_rev) * 100, 2) if (latest_rev and latest_op) else "--",
                "profit_margin": round((latest_pat / latest_rev) * 100, 2) if (latest_rev and latest_pat) else "--"
            }
    except Exception as e:
        print(f"Statement parser error: {e}")

    return fundamentals

def fetch_equity_analytics(ticker: str, period: str = "6mo") -> dict:
    clean_ticker = ticker.strip().upper()
    benchmark_symbol = get_benchmark_ticker(clean_ticker)
    benchmark_name = "NIFTY 50" if benchmark_symbol == "^NSEI" else "S&P 500"
    rf_rate = get_risk_free_rate(clean_ticker)

    ticker_obj = yf.Ticker(clean_ticker)
    asset_df = ticker_obj.history(period=period, auto_adjust=True)
    if asset_df.empty or len(asset_df) < 10:
        raise ValueError(f"No sufficient market data found for symbol '{clean_ticker}'")

    bench_df = yf.download(benchmark_symbol, period=period, interval="1d", auto_adjust=True, progress=False)

    if isinstance(asset_df.columns, pd.MultiIndex):
        asset_df.columns = asset_df.columns.get_level_values(0)
    if isinstance(bench_df.columns, pd.MultiIndex):
        bench_df.columns = bench_df.columns.get_level_values(0)

    # Candlesticks & Volume Series
    candlesticks = []
    volume_series = []
    for idx, row in asset_df.iterrows():
        t = idx.strftime("%Y-%m-%d")
        o = round(float(row["Open"]), 2)
        h = round(float(row["High"]), 2)
        l = round(float(row["Low"]), 2)
        c = round(float(row["Close"]), 2)
        v = int(row["Volume"]) if "Volume" in row and not np.isnan(row["Volume"]) else 0

        candlesticks.append({"time": t, "open": o, "high": h, "low": l, "close": c})
        volume_series.append({
            "time": t,
            "value": v,
            "color": "rgba(16, 185, 129, 0.3)" if c >= o else "rgba(244, 63, 94, 0.3)"
        })

    # SMA & EMA Overlays
    close_series = asset_df["Close"].dropna()
    sma15 = close_series.rolling(window=15).mean().dropna()
    ema15 = close_series.ewm(span=15, adjust=False).mean().dropna()

    sma_series = [{"time": idx.strftime("%Y-%m-%d"), "value": round(float(val), 2)} for idx, val in sma15.items()]
    ema_series = [{"time": idx.strftime("%Y-%m-%d"), "value": round(float(val), 2)} for idx, val in ema15.items()]

    # Continuous Logarithmic Returns & Volatility
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

    # Beta
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
    cumulative_return = round(float(((close_series.iloc[-1] / close_series.iloc[0]) - 1) * 100), 2)

    # Plain English Educational Guides
    if beta > 1.2:
        beta_desc = f"Moves much more wildly than the market ({benchmark_name}). If the market moves 1%, this stock tends to swing by about {beta}%."
    elif beta < 0.8:
        beta_desc = f"Much calmer than the general market ({benchmark_name}). It doesn't jump as high during rallies, but it also doesn't fall as hard during market crashes."
    else:
        beta_desc = f"Moves pretty much hand-in-hand with the overall market ({benchmark_name}), matching its ups and downs almost 1-for-1."

    if sortino > 1.5:
        sortino_desc = "Excellent safety score. It generates great gains while keeping painful down-days and losses to a minimum."
    elif sortino > 0.5:
        sortino_desc = "Decent safety score. The returns are compensating you reasonably well for the bad down-days."
    elif sortino > 0:
        sortino_desc = "Low safety score. You are getting slightly better returns than a safe bank FD, but you are experiencing regular red days."
    else:
        sortino_desc = "Negative safety score. The drops and red days outweigh the gains; keeping money in a risk-free government bond gave a better return."

    guide = [
        {
            "topic": "Sortino Ratio (The 'Bad Volatility' Score)",
            "badge": f"{sortino} Score",
            "badge_color": "emerald" if sortino >= 1.0 else "amber",
            "text": f"{sortino_desc} Unlike basic risk scores that punish a stock just for shooting up fast, Sortino only judges the stock when it actually drops and causes losses."
        },
        {
            "topic": f"Beta (Market Reactivity vs {benchmark_name})",
            "badge": f"{beta}x Pace",
            "badge_color": "rose" if beta > 1.2 else ("emerald" if beta < 0.8 else "sky"),
            "text": beta_desc
        },
        {
            "topic": "Annualized Volatility (Rollercoaster Factor)",
            "badge": f"{annualized_vol}% Swings",
            "badge_color": "amber" if annualized_vol > 25 else "emerald",
            "text": f"How bumpy the ride is over a typical year. A score of {annualized_vol}% means you should expect frequent price swings of this magnitude in either direction."
        },
        {
            "topic": "Max Drawdown (Worst Fall from the Top)",
            "badge": f"{max_drawdown}% Fall",
            "badge_color": "rose" if max_drawdown < -20 else "amber",
            "text": f"If you had the worst luck and bought at the exact peak in the last 6 months, you would have seen your investment drop by {max_drawdown}% before it started recovering."
        }
    ]

    fundamentals = extract_financial_statements(ticker_obj, current_price=current_price)

    return {
        "symbol": clean_ticker,
        "benchmark": benchmark_symbol,
        "metrics": {
            "current_price": current_price,
            "cumulative_return_pct": cumulative_return,
            "annualized_volatility_pct": annualized_vol,
            "beta_against_benchmark": beta,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_pct": max_drawdown,
            "candlesticks": candlesticks,
            "volume_series": volume_series,
            "sma_15": sma_series,
            "ema_15": ema_series,
            "educational_guide": guide,
            "fundamentals": fundamentals
        }
    }