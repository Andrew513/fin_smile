import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 1. Load Intel's option data
ticker = yf.Ticker("INTC")

# 2. List all available expiration dates
expirations = ticker.options
print("Available expirations:", expirations[:5])  # show first few

# 3. Pick one expiration (e.g. around 1 month out)
if len(expirations) >= 3:
    target_exp = expirations[2]
else:
    target_exp = expirations[-1]  # fallback

# 4. Fetch option chain
opt = ticker.option_chain(target_exp)
calls = opt.calls
puts = opt.puts

# 5. Clean up data
calls = calls.dropna(subset=["impliedVolatility"])
puts = puts.dropna(subset=["impliedVolatility"])
calls["impliedVolatility"] *= 100
puts["impliedVolatility"] *= 100

# 6. Plot IV vs strike
plt.figure(figsize=(10,6))
plt.plot(calls["strike"], calls["impliedVolatility"], marker="o", label="Calls")
plt.plot(puts["strike"], puts["impliedVolatility"], marker="x", label="Puts")
plt.title(f"INTC Implied Volatility Skew\nExpiration: {target_exp} | {datetime.now():%Y-%m-%d %H:%M}")
plt.xlabel("Strike Price ($)")
plt.ylabel("Implied Volatility (%)")
plt.legend()
plt.grid(True)
plt.show()