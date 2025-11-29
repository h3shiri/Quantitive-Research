"""
Momentum hypothesis test scaffold using only yfinance (tickers, sectors, prices).

Workflow:
- Fetch S&P 500 tickers via yfinance.tickers_sp500() with fallbacks (Wikipedia or static sample).
- Pull per-ticker sector via yfinance Ticker().info (best-effort; drop tickers with missing sectors).
- Pull adjusted prices via yfinance for stocks and sector ETFs.
- Build momentum signals (12m-1m, 6m, sector-relative, vol-adjusted).
- Form monthly long/short portfolios and compute factor returns.
- Report performance stats to support or reject the hypothesis.

Requirements:
pip install pandas numpy yfinance scipy statsmodels matplotlib
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import yfinance as yf
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

SECTOR_ETFS: Dict[str, str] = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}


@dataclass
class PerfStats:
    cagr: float
    vol: float
    sharpe: float
    max_dd: float
    hit_rate: float
    t_stat: float


def fetch_sp500_constituents(tickers: Iterable[str]) -> pd.DataFrame:
    rows = []
    for symbol in tickers:
        try:
            info = yf.Ticker(symbol).info or {}
            sector = info.get("sector")
        except Exception:
            sector = None
        rows.append({"symbol": symbol, "sector": sector})
    return pd.DataFrame(rows)


def get_sp500_tickers() -> List[str]:
    if hasattr(yf, "tickers_sp500"):
        try:
            tickers = yf.tickers_sp500()
            if tickers:
                return tickers
        except Exception:
            pass
    # Fallback: try Wikipedia
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        if tables:
            data = tables[0]
            if "Symbol" in data.columns:
                return data["Symbol"].astype(str).tolist()
    except Exception:
        pass
    # Last resort: small static list for demo/backtest to still run
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "XOM", "JNJ", "PG", "NVDA"]


def download_prices(
    tickers: Iterable[str],
    start: dt.date,
    end: dt.date,
    interval: str = "1d",
) -> pd.DataFrame:
    tickers = list(tickers)
    if not tickers:
        raise ValueError("No tickers provided for download")
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    closes = data["Close"]
    closes = closes[tickers]  # Preserve column order
    closes = closes.dropna(axis=1, how="all")
    return closes


def compute_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.resample("M").last().dropna(how="all")


def compute_signals(
    stock_monthly: pd.DataFrame,
    sector_monthly: pd.DataFrame,
    sector_map: Dict[str, str],
) -> pd.DataFrame:
    twelve_to_one = (stock_monthly.shift(1) / stock_monthly.shift(12)) - 1
    six_month = stock_monthly.pct_change(6)
    one_month = stock_monthly.pct_change(1)
    vol_12m = one_month.rolling(12).std() * np.sqrt(12)

    sector_six = sector_monthly.pct_change(6)

    frames: List[pd.DataFrame] = []
    for symbol in stock_monthly.columns:
        sector = sector_map.get(symbol)
        sector_etf = SECTOR_ETFS.get(sector, None)
        sector_mom = (
            sector_six[sector_etf] if sector_etf in sector_six.columns else pd.Series(index=six_month.index)
        )
        df = pd.DataFrame(
            {
                "symbol": symbol,
                "sector": sector,
                "mom_12_1": twelve_to_one[symbol],
                "mom_6m": six_month[symbol],
                "sector_rel_6m": six_month[symbol] - sector_mom,
                "vol_adj_12_1": twelve_to_one[symbol] / vol_12m[symbol],
            }
        )
        frames.append(df)
    signals = pd.concat(frames)
    signals = signals.set_index("symbol", append=True).sort_index()
    signals.index.names = ["date", "symbol"]
    return signals


def forward_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    rets = monthly_prices.pct_change()
    return rets.shift(-1)


def decile_portfolio_returns(
    signals: pd.DataFrame,
    fwd_rets: pd.DataFrame,
    signal_col: str = "mom_12_1",
    quantile: float = 0.1,
) -> pd.Series:
    results: List[Tuple[pd.Timestamp, float]] = []
    for date, df in signals.groupby(level=0):
        xsec = df.droplevel(0)
        if xsec[signal_col].dropna().empty:
            continue
        cutoff_low = xsec[signal_col].quantile(quantile)
        cutoff_high = xsec[signal_col].quantile(1 - quantile)
        long_names = xsec.index[xsec[signal_col] >= cutoff_high]
        short_names = xsec.index[xsec[signal_col] <= cutoff_low]

        long_ret = fwd_rets.loc[date, long_names].mean()
        short_ret = fwd_rets.loc[date, short_names].mean()
        results.append((date, long_ret - short_ret))
    return pd.Series(dict(results)).sort_index()


def long_only_returns(
    signals: pd.DataFrame,
    fwd_rets: pd.DataFrame,
    signal_col: str = "mom_12_1",
    top_quantile: float = 0.2,
) -> pd.Series:
    results: List[Tuple[pd.Timestamp, float]] = []
    for date, df in signals.groupby(level=0):
        xsec = df.droplevel(0)
        if xsec[signal_col].dropna().empty:
            continue
        cutoff_high = xsec[signal_col].quantile(1 - top_quantile)
        long_names = xsec.index[xsec[signal_col] >= cutoff_high]
        long_ret = fwd_rets.loc[date, long_names].mean()
        results.append((date, long_ret))
    return pd.Series(dict(results)).sort_index()


def compute_perf_stats(series: pd.Series, periods_per_year: int = 12) -> PerfStats:
    s = series.dropna()
    if s.empty:
        return PerfStats(0, 0, 0, 0, 0, 0)
    cagr = (1 + s).prod() ** (periods_per_year / len(s)) - 1
    vol = s.std() * np.sqrt(periods_per_year)
    sharpe = s.mean() / s.std() * np.sqrt(periods_per_year) if s.std() != 0 else 0
    equity = (1 + s).cumprod()
    peak = equity.cummax()
    dd = (equity / peak - 1).min()
    hit_rate = (s > 0).mean()
    t_stat = stats.ttest_1samp(s, 0.0, nan_policy="omit").statistic
    return PerfStats(
        cagr=cagr,
        vol=vol,
        sharpe=sharpe,
        max_dd=dd,
        hit_rate=hit_rate,
        t_stat=t_stat,
    )


def describe_stats(name: str, stats_: PerfStats) -> str:
    return (
        f"{name}: CAGR {stats_.cagr:.2%}, Vol {stats_.vol:.2%}, "
        f"Sharpe {stats_.sharpe:.2f}, MaxDD {stats_.max_dd:.2%}, "
        f"Hit {stats_.hit_rate:.2%}, t-stat {stats_.t_stat:.2f}"
    )


def plot_equity_curves(
    ls_returns: pd.Series,
    long_only_returns: pd.Series,
    output_path: str = "momentum_pnl.png",
) -> None:
    """Plot cumulative P&L for long-short and long-only legs."""
    curves = pd.DataFrame(
        {
            "Long-Short": (1 + ls_returns.dropna()).cumprod(),
            "Long-Only": (1 + long_only_returns.dropna()).cumprod(),
        }
    ).dropna(how="all")
    if curves.empty:
        return
    plt.figure(figsize=(10, 6))
    for col in curves.columns:
        plt.plot(curves.index, curves[col], label=col, linewidth=2)
    plt.title("Momentum Strategy P&L (Cumulative)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (Gross)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def run_backtest(
    start: dt.date = dt.date(2012, 1, 1),
    end: dt.date | None = None,
    sample_size: int = 150,
) -> None:
    end = end or dt.date.today()
    sp500 = get_sp500_tickers()
    sampled = pd.Series(sp500).sample(min(sample_size, len(sp500)), random_state=42).tolist()

    universe = fetch_sp500_constituents(sampled)
    universe = universe.dropna(subset=["sector"])
    if universe.empty:
        raise RuntimeError("No tickers with sector info were found from yfinance. Try reducing sample_size or using a static sector map.")

    tickers = universe["symbol"].tolist()
    sector_map = dict(zip(tickers, universe["sector"]))
    sector_tickers = sorted(set(SECTOR_ETFS.values()))

    stock_prices = download_prices(tickers, start=start, end=end)
    sector_prices = download_prices(sector_tickers, start=start, end=end)

    stock_monthly = compute_monthly(stock_prices)
    sector_monthly = compute_monthly(sector_prices)

    signals = compute_signals(stock_monthly, sector_monthly, sector_map)
    fwd_rets = forward_returns(stock_monthly)

    ls_returns = decile_portfolio_returns(signals, fwd_rets, signal_col="mom_12_1", quantile=0.1)
    long_only = long_only_returns(signals, fwd_rets, signal_col="mom_12_1", top_quantile=0.2)

    print(describe_stats("Long-Short P10-P90", compute_perf_stats(ls_returns)))
    print(describe_stats("Long-Only Top 20%", compute_perf_stats(long_only)))
    plot_equity_curves(ls_returns, long_only, output_path="momentum_pnl.png")
    print("Saved P&L chart to momentum_pnl.png")


if __name__ == "__main__":
    run_backtest()
