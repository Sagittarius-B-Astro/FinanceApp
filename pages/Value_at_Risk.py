import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

class VAR():
    def __init__(self, tickers, start_date, end_date, confidence, rolling, value):
        self.tickers = tickers
        self.start = start_date
        self.end = end_date
        self.confidence = confidence
        self.roll = rolling
        self.value = value
        self.get_data()

    def get_data(self):
        df = yf.download(self.tickers, str(self.start), str(self.end))

        if df.empty or "Close" not in df:
            st.write(self.start, self.end)
            raise ValueError("No data retrieved. Check ticker symbols and date range.")
        
        self.adj_close_df = df["Close"]
        self.log_returns_df = np.log(self.adj_close_df / self.adj_close_df.shift(1)).dropna()

        if self.log_returns_df.empty:
            raise ValueError("Log returns DataFrame is empty. Ensure sufficient data is available.")

        self.equal_weights = np.array([1 / len(self.tickers)] * len(self.tickers))
        historical_returns = (self.log_returns_df * self.equal_weights).sum(axis=1)
        self.rolling_returns = historical_returns.rolling(window=self.roll).sum().dropna()

        if self.rolling_returns.empty:
            raise ValueError("Rolling returns DataFrame is empty. Increase rolling window or check data availability.")

        self.historical_method()
        self.parametric_method()

    def historical_method(self):
        historical_VaR = -np.percentile(self.rolling_returns, 100 - (self.confidence * 100)) * self.value
        self.historical_var = historical_VaR

    def parametric_method(self):
        self.cov_matrix = self.log_returns_df.cov() * 252
        self.portfolio_std = np.sqrt(np.dot(self.equal_weights.T, np.dot(self.cov_matrix, self.equal_weights)))
        parametric_VaR = self.portfolio_std * norm.ppf(self.confidence) * np.sqrt(self.roll / 252) * self.value
        self.parametric_var = parametric_VaR

    def plot_var_results(self, title, var_value, returns_dollar, conf_level):
        plt.figure(figsize=(12, 6))
        plt.hist(returns_dollar, bins=50, density=True)
        plt.xlabel(f'\n {title} VaR = ${var_value:.2f}')
        plt.ylabel('Frequency')
        plt.title(f"Distribution of Portfolio's {self.roll}-Day Returns ({title} VaR)")
        plt.axvline(-var_value, color='r', linestyle='dashed', linewidth=2, label=f'VaR at {conf_level:.0%} confidence level')
        plt.legend()
        plt.tight_layout()
        return plt

def calculate_and_display_var(tickers, start_date, end_date, rolling_window, confidence_level, value):
    var = VAR(tickers, start_date, end_date, confidence_level, rolling_window, value)
    
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.info("Historical VaR Chart")
        historical_chart = var.plot_var_results("Historical", var.historical_var, var.rolling_returns * var.value, confidence_level)
        st.pyplot(historical_chart)

    with chart_col2:
        st.info("Parametric VaR Chart")
        parametric_chart = var.plot_var_results("Parametric", var.parametric_var, var.rolling_returns * var.value, confidence_level)
        st.pyplot(parametric_chart)

    col1, col3 = st.columns([1, 1])
    
    with col1:
        st.info("Input Summary")
        st.write(f"Tickers: {tickers}")
        st.write(f"Start Date: {start_date}")
        st.write(f"End Date: {end_date}")
        st.write(f"Rolling Window: {rolling_window} days")
        st.write(f"Confidence Level: {confidence_level:.2%}")
        st.write(f"Portfolio Value: ${value:,.2f}")

    with col3:
        st.info("VaR Calculation Output")
        data = {
            "Method": ["Historical", "Parametric"],
            "VaR Value": [f"${var.historical_var:,.2f}", f"${var.parametric_var:,.2f}"]
        }
        df = pd.DataFrame(data)
        st.table(df)

    st.session_state['recent_outputs'].append({
        "Historical": f"${var.historical_var:,.2f}",
        "Parametric": f"${var.parametric_var:,.2f}"
    })

    with col3:
        st.info("Previous VaR Calculation Outputs")
        recent_df = pd.DataFrame(st.session_state['recent_outputs'])
        st.table(recent_df)

st.set_page_config(page_title="Value at Risk", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

if 'recent_outputs' not in st.session_state:
    st.session_state['recent_outputs'] = []

with st.sidebar:
    st.title("Greeks and Implied Volatility")

    tickers = st.text_input("Enter Ticker Symbols Separated by Commas:", value="SPY,AAPL,QQQ").split(",")
    start_date = st.date_input("Enter Starting Date for Portfolio:", value=pd.to_datetime('2021-01-01'))
    end_date = st.date_input("Enter Ending Date for Portfolio:", value=pd.to_datetime("today")) - pd.Timedelta(days=1)
    confidence = st.slider("Enter the confidence in the portfolio:", min_value=0.90, max_value=1.00)
    rolling = st.number_input("Enter rolling period length", min_value=1, max_value=252, value=50)
    val_of_pf = st.number_input("Value of the Portfolio:", value=1000)

if 'first_run' not in st.session_state or st.session_state['first_run']:
    st.session_state['first_run'] = False
    default_tickers = 'SPY,AAPL,QQQ'.split(",")
    default_start_date = pd.to_datetime('2021-01-01')
    default_end_date = pd.to_datetime('today') - pd.Timedelta(days=1)
    default_rolling_window = 20
    default_confidence_level = 0.95
    default_portfolio_val = 1000

    calculate_and_display_var(default_tickers, default_start_date, default_end_date, default_rolling_window, default_confidence_level, default_portfolio_val)

calculate_and_display_var(tickers, start_date, end_date, rolling, confidence, val_of_pf)