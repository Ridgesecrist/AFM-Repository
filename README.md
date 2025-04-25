# AFM-Repository
Advanced Financial Modeling GitHub
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Streamlit setup
st.set_page_config(layout="wide")
st.title("📈 EMA 8/21 Crossover Strategy vs. S&P 500")

# Sidebar inputs
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
benchmark = "^GSPC"
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Download data
data = yf.download([ticker, benchmark], start=start_date, end=end_date)["Close"].dropna()
data = data[[ticker, benchmark]].dropna()

# Calculate EMAs
data["EMA_8"] = data[ticker].ewm(span=8, adjust=False).mean()
data["EMA_21"] = data[ticker].ewm(span=21, adjust=False).mean()

# Generate buy/sell signals
data["Signal"] = 0
data["Signal"].iloc[1:] = np.where(
    (data["EMA_8"].iloc[1:] > data["EMA_21"].iloc[1:]) &
    (data["EMA_8"].shift(1).iloc[1:] <= data["EMA_21"].shift(1).iloc[1:]), 1,
    np.where(
        (data["EMA_8"].iloc[1:] < data["EMA_21"].iloc[1:]) &
        (data["EMA_8"].shift(1).iloc[1:] >= data["EMA_21"].shift(1).iloc[1:]), -1, 0
    )
)

# Simulate trading
initial_capital = 10000
cash = initial_capital
holdings = 0
portfolio_value = []
position = 0  # 0 = no stock, 1 = holding

for i in range(len(data)):
    price = data[ticker].iloc[i]
    signal = data["Signal"].iloc[i]

    if signal == 1 and position == 0:
        holdings = cash / price
        cash = 0
        position = 1
    elif signal == -1 and position == 1:
        cash = holdings * price
        holdings = 0
        position = 0

    total = cash + holdings * price
    portfolio_value.append(total)

data["Total"] = portfolio_value
data["Strategy Return"] = data["Total"].pct_change()

# Benchmark returns
benchmark_returns = data[benchmark].pct_change()
benchmark_cum = (1 + benchmark_returns).cumprod() * initial_capital

# Performance metrics
strategy_final = data["Total"].iloc[-1]
strategy_return = strategy_final / initial_capital - 1
strategy_volatility = data["Strategy Return"].std() * np.sqrt(252)

col1, col2, col3 = st.columns(3)
col1.metric("📈 Strategy Return", f"{strategy_return:.2%}")
col2.metric("📊 Volatility", f"{strategy_volatility:.2%}")
col3.metric("💰 Final Portfolio Value", f"${strategy_final:,.2f}")

# Plot strategy vs benchmark
st.subheader("📉 Strategy vs. S&P 500")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data.index, data["Total"], label="EMA Strategy")
ax.plot(data.index, benchmark_cum, label="S&P 500")
ax.set_ylabel("Portfolio Value ($)")
ax.set_title(f"{ticker} EMA 8/21 Strategy vs. S&P 500")
ax.legend()
st.pyplot(fig)

# Optional data table
if st.checkbox("Show Raw Data"):
    st.dataframe(data.tail(50))
