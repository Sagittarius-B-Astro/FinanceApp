import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title="Value at Risk", page_icon="📈", layout="wide", initial_sidebar_state="expanded")


@st.cache_data(ttl=3600, show_spinner="Downloading prices...")
def download_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    """Download adjusted closes. Cached for an hour so widget clicks don't re-hit Yahoo,
    and retried with backoff because Yahoo intermittently rate-limits shared server IPs.
    Exceptions are not cached by Streamlit, so a failed attempt is retried on the next run."""
    last_error = None
    for attempt in range(3):
        try:
            df = yf.download(list(tickers), start=start, end=end, auto_adjust=True,
                             progress=False, threads=False)
            if not df.empty and "Close" in df:
                close = df["Close"]
                if isinstance(close, pd.Series):          # single ticker
                    close = close.to_frame(tickers[0])
                close = close.dropna(axis=1, how="all")    # drop tickers Yahoo returned nothing for
                if not close.empty:
                    return close
        except Exception as e:                             # e.g. YFRateLimitError
            last_error = e
        time.sleep(2 * (attempt + 1))
    raise ValueError(f"Yahoo Finance returned no data after 3 attempts"
                     f"{f' (last error: {last_error})' if last_error else ''}. "
                     "This is usually a temporary rate limit; try again in a few minutes.")


class VAR:
    def __init__(self, prices, confidence, rolling, value):
        self.prices = prices
        self.confidence = confidence
        self.roll = rolling
        self.value = value
        self.compute()

    def compute(self):
        self.log_returns_df = np.log(self.prices / self.prices.shift(1)).dropna()
        if self.log_returns_df.empty:
            raise ValueError("Not enough overlapping price history for these tickers and dates.")
        n = self.prices.shape[1]
        self.equal_weights = np.full(n, 1 / n)
        daily = self.log_returns_df @ self.equal_weights
        self.rolling_returns = daily.rolling(window=self.roll).sum().dropna()
        if self.rolling_returns.empty:
            raise ValueError("Rolling window is longer than the available data. "
                             "Shorten the window or widen the date range.")
        # Historical: empirical loss quantile of the rolling-window returns
        self.historical_var = -np.percentile(self.rolling_returns, (1 - self.confidence) * 100) * self.value
        # Parametric: normal approximation, annualised covariance scaled to the window
        cov = self.log_returns_df.cov() * 252
        self.portfolio_std = float(np.sqrt(self.equal_weights @ cov.values @ self.equal_weights))
        self.parametric_var = (self.portfolio_std * norm.ppf(self.confidence)
                               * np.sqrt(self.roll / 252) * self.value)

    def plot_var_results(self, title, var_value, conf_level):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(self.rolling_returns * self.value, bins=50, density=True)
        ax.set_xlabel(f"\n{title} VaR = ${var_value:,.2f}")
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of Portfolio's {self.roll}-Day Returns ({title} VaR)")
        ax.axvline(-var_value, color="r", linestyle="dashed", linewidth=2,
                   label=f"VaR at {conf_level:.0%} confidence level")
        ax.legend()
        fig.tight_layout()
        return fig


def calculate_and_display_var(tickers, start_date, end_date, rolling_window, confidence_level, value):
    # yfinance's `end` is exclusive, so add a day to include the chosen end date.
    end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    try:
        prices = download_prices(tuple(tickers), str(pd.Timestamp(start_date).date()),
                                 str(end_exclusive.date()))
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            st.warning(f"No data found for: {', '.join(missing)}. Calculated with the remaining tickers.")
        var = VAR(prices, confidence_level, rolling_window, value)
    except ValueError as e:
        st.error(str(e))
        return

    col_a, col_b = st.columns(2)
    for col, label, val in ((col_a, "Historical", var.historical_var),
                            (col_b, "Parametric", var.parametric_var)):
        with col:
            st.info(f"{label} VaR Chart")
            fig = var.plot_var_results(label, val, confidence_level)
            st.pyplot(fig)
            plt.close(fig)

    left, right = st.columns(2)
    with left:
        st.info("Input Summary")
        st.write(f"Tickers: {list(prices.columns)}")
        st.write(f"Start Date: {pd.Timestamp(start_date).date()}")
        st.write(f"End Date: {pd.Timestamp(end_date).date()}")
        st.write(f"Rolling Window: {rolling_window} days")
        st.write(f"Confidence Level: {confidence_level:.2%}")
        st.write(f"Portfolio Value: ${value:,.2f}")
    with right:
        st.info("VaR Calculation Output")
        st.table(pd.DataFrame({"Method": ["Historical", "Parametric"],
                               "VaR Value": [f"${var.historical_var:,.2f}", f"${var.parametric_var:,.2f}"]}))
        st.session_state["recent_outputs"].append(
            {"Historical": f"${var.historical_var:,.2f}", "Parametric": f"${var.parametric_var:,.2f}"})
        st.info("Previous VaR Calculation Outputs")
        st.table(pd.DataFrame(st.session_state["recent_outputs"]))


if "recent_outputs" not in st.session_state:
    st.session_state["recent_outputs"] = []

with st.sidebar:
    st.title("Value at Risk Calculator")
    with st.form("var_inputs"):
        raw = st.text_input("Enter Ticker Symbols Separated by Commas:", value="SPY,AAPL,QQQ")
        start_date = st.date_input("Enter Starting Date for Portfolio:", value=pd.Timestamp("2021-01-01"))
        end_date = st.date_input("Enter Ending Date for Portfolio:", value=pd.Timestamp.today())
        confidence = st.slider("Confidence level:", min_value=0.90, max_value=0.99, value=0.95, step=0.01)
        rolling = st.number_input("Rolling period length (days)", min_value=1, max_value=252, value=20)
        val_of_pf = st.number_input("Value of the Portfolio:", value=1000, min_value=1)
        submitted = st.form_submit_button("Calculate VaR")

tickers = list(dict.fromkeys(t.strip().upper() for t in raw.split(",") if t.strip()))  # strip, dedupe

# Run once on first load, then only when the button is pressed (not on every widget change).
if submitted or not st.session_state.get("auto_ran"):
    st.session_state["auto_ran"] = True
    if not tickers:
        st.error("Enter at least one ticker symbol.")
    elif pd.Timestamp(start_date) >= pd.Timestamp(end_date):
        st.error("Start date must be before the end date.")
    else:
        calculate_and_display_var(tickers, start_date, end_date, rolling, confidence, val_of_pf)
