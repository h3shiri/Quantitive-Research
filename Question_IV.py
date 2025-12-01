# 1. Install yfinance if needed:
# pip install yfinance matplotlib pandas
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# --- Parameters ---
tickers = yf.Tickers('QCOM AXON')

end = dt.date.today()
start = end - dt.timedelta(days=365 * 10)  # approx 10 years

# --- Download adjusted close prices (includes splits/dividends) ---
data = yf.download(tickers.symbols, start=start, end=end)

# Keep only price
prices = data["Close"]

# Drop any rows with missing price data
prices = prices.dropna()

# --- Normalize to 100 at the start to show total price return ---
norm = prices / prices.iloc[0] * 100

# --- 6-month drawdowns ---
semi = norm.resample("6M").last()
semi_peak = semi.cummax()
semi_dd = semi / semi_peak - 1
semi_dd_pct = semi_dd.mul(100)
# Label by period for readability
semi_dd_pct.index = semi_dd_pct.index.to_period("6M").astype(str)
# For plotting upward bars, flip sign so drawdown depth is positive
semi_dd_depth = semi_dd_pct.mul(-1)

# --- Plot price ---
plt.figure(figsize=(10, 6))
for col in norm.columns:
    plt.plot(norm.index, norm[col], label=col)

plt.title("QCOM vs AXON – Normalized Price (10 Years, =100 at Start)")
plt.xlabel("Date")
plt.ylabel("Index Level (Start = 100)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- Plot 6-month drawdown depth (positive, grouped bars) ---
plt.figure(figsize=(10, 4))
ax = plt.gca()
semi_dd_depth.plot(kind="bar", width=0.8, ax=ax)
plt.title("6-Month Drawdown Depth (%)")
plt.xlabel("Period")
plt.ylabel("Drawdown Depth (%)")
plt.axhline(0, color="black", linewidth=0.8)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

plt.show()
