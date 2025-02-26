import streamlit as st
import time 
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import timedelta
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import plotly.graph_objects as go
from numpy import exp, sqrt, log
from Finance_Pricing_App import BlackScholesOption

today = pd.Timestamp('today').normalize()

class Greeks(BlackScholesOption):
    def __init__(self, time_to_mature, volatility, interest, cur_price, str_price, dividend_yield):
        super().__init__(time_to_mature, volatility, interest, cur_price, str_price, dividend_yield)
        self.C, self.P = self.price()

    def Delta(self): 
        d1 = self.d1()
        T = self.time_to_mature
        q = self.div_yield
        call_Delta = exp(-q * T) * norm.cdf(d1)
        put_Delta = -exp(-q * T) * norm.cdf(-d1)
        return call_Delta, put_Delta
    
    def Vega(self):
        d1 = self.d1()
        T = self.time_to_mature
        S = self.cur_price
        q = self.div_yield
        Vega = S * exp(-q * T) * sqrt(T) * norm.pdf(d1) / 100
        return Vega, Vega

    def Theta(self):
        d1 = self.d1()
        d2 = self.d2()
        T = self.time_to_mature
        sigma = self.volatility
        r = self.interest
        S = self.cur_price
        K = self.str_price
        q = self.div_yield
        call_Theta = (-(S * sigma * exp(-q * T) * norm.pdf(d1) / (2 * sqrt(T))) - r * K * exp(-r * T) * norm.cdf(d2) + q * S * exp(-q * T) * norm.cdf(d1)) / 365
        put_Theta = (-(S * sigma * exp(-q * T) * norm.pdf(d1) / (2 * sqrt(T))) + r * K * exp(-r * T) * norm.cdf(-d2) - q * S * exp(-q * T) * norm.cdf(-d1)) / 365
        return call_Theta, put_Theta
    
    def Rho(self):
        d2 = self.d2()
        T = self.time_to_mature
        r = self.interest
        K = self.str_price
        call_Rho = K * T * exp(-r * T) * norm.cdf(d2) / 100
        put_Rho = -K * T * exp(-r * T) * norm.cdf(-d2) / 100
        return call_Rho, put_Rho
    
    def Epsilon(self):
        d1 = self.d1()
        S = self.cur_price
        T = self.time_to_mature
        q = self.div_yield
        call_Epsilon, put_Epsilon = -S * T * exp(-q * T) * norm.cdf(d1), S * T * exp(-q * T) * norm.cdf(-d1)
        return call_Epsilon, put_Epsilon

    def Lambda(self):
        S = self.cur_price
        C, P = self.C, self.P
        call_Delta, put_Delta = self.Delta()
        call_Lambda, put_Lambda = call_Delta * S / C, put_Delta * S / P
        return call_Lambda, put_Lambda

    def Gamma(self):
        d1 = self.d1()
        T = self.time_to_mature
        sigma = self.volatility
        S = self.cur_price
        q = self.div_yield
        Gamma = exp(-q * T) * norm.pdf(d1) / (S * sigma * sqrt(T))
        return Gamma, Gamma

    def Vanna(self):
        d1 = self.d1()
        d2 = self.d2()
        T = self.time_to_mature
        sigma = self.volatility
        q = self.div_yield
        Vanna = -exp(-q * T) * norm.pdf(d1) * d2 / sigma
        return Vanna, Vanna
    
    def Charm(self):
        d1 = self.d1()
        d2 = self.d2()
        T = self.time_to_mature
        r = self.interest
        sigma = self.volatility
        q = self.div_yield
        call_Charm = q * exp(-q * T) * norm.cdf(d1) - exp(-q * T) * norm.pdf(d1) * (2 * T * (r - q) - d2 * sigma * sqrt(T)) / (2 * T * sigma * sqrt(T))
        put_Charm = -q * exp(-q * T) * norm.cdf(-d1) - exp(-q * T) * norm.pdf(d1) * (2 * T * (r - q) - d2 * sigma * sqrt(T)) / (2 * T * sigma * sqrt(T))
        return call_Charm, put_Charm
    
    def Vomma(self):
        d1 = self.d1()
        d2 = self.d2()
        sigma = self.volatility
        call_Vega, put_Vega = self.Vega()
        call_Vomma, put_Vomma = call_Vega * d1 * d2 / sigma, put_Vega * d1 * d2 / sigma
        return call_Vomma, put_Vomma

    def Vera(self):
        d1 = self.d1()
        d2 = self.d2()
        sigma = self.volatility
        T = self.time_to_mature
        r = self.interest
        K = self.str_price
        Vera = -K * T * exp(-r * T) * norm.pdf(d2) * d1 / sigma
        return Vera, Vera
    
    def Veta(self):
        d1 = self.d1()
        d2 = self.d2()
        S = self.cur_price
        sigma = self.volatility
        q = self.div_yield
        T = self.time_to_mature
        r = self.interest
        Veta = -S * exp(-q * T) * norm.pdf(d1) * sqrt(T) * (q + (r - q) * d1 / (sigma * sqrt(T)) - (1 + d1 * d2)/(2 * T))
        return Veta, Veta
    
    def Varpi(self):
        S = self.cur_price
        K = self.str_price
        Gamma = self.Gamma()[0]
        Varpi = (S / K)**2 * Gamma
        return Varpi, Varpi
    
    def Speed(self):
        S = self.cur_price
        sigma = self.volatility
        d1 = self.d1()
        T = self.time_to_mature
        Gamma = self.Gamma()[0]
        Speed = -Gamma / S * (d1 / (sigma * sqrt(T)) + 1)
        return Speed, Speed
    
    def Zomma(self):
        sigma = self.volatility
        d1 = self.d1()
        d2 = self.d2()
        Gamma = self.Gamma()[0]
        Zomma = Gamma * (d1 * d2 - 1) / sigma
        return Zomma, Zomma

    def Color(self):
        d1 = self.d1()
        d2 = self.d2()
        T = self.time_to_mature
        sigma = self.volatility
        r = self.interest
        S = self.cur_price
        K = self.str_price
        q = self.div_yield
        Color = -exp(-q * T) * norm.pdf(d1) / (2 * S * T * sigma * sqrt(T)) * (2 * q * T + 1 + (2 * (r - q) * T - d2 * sigma * sqrt(T)) * d1 / (sigma * sqrt(T)))
        return Color, Color

    def Ultima(self):
        d1 = self.d1()
        d2 = self.d2()
        sigma = self.volatility
        call_Vega, put_Vega = self.Vega()
        call_Ultima = -call_Vega / sigma**2 * (d1 * d2 * (1 - d1 * d2) + d1**2 + d2**2)
        put_Ultima = -put_Vega / sigma**2 * (d1 * d2 * (1 - d1 * d2) + d1**2 + d2**2)
        return call_Ultima, put_Ultima
    
    def ParmiCharma(self):
        d1 = self.d1()
        d2 = self.d2()
        T = self.time_to_mature
        sigma = self.volatility
        r = self.interest
        S = self.cur_price
        K = self.str_price
        q = self.div_yield
        call_Charm, put_Charm = self.Charm()
        call_ParmiCharma = (q - (2 * (r - q) * T - d2 * sigma * sqrt(T)) / (2 * T * sigma * sqrt(T))) * call_Charm - exp(-q * T) * norm.pdf(d1) * (2 * d2 * sigma**2 * T - (r \
            - q) * sigma * T * sqrt(T) - sigma**2 * T**2 * (-log(S / K) / (2 * sigma * sqrt(T**3)) + (r - q - sigma**2 / 2) / (2 * sigma * sqrt(T))) / (2 * T**3 * sigma**2))
        put_ParmiCharma = (q - (2 * (r - q) * T - d2 * sigma * sqrt(T)) / (2 * T * sigma * sqrt(T))) * put_Charm - exp(-q * T) * norm.pdf(d1) * (2 * d2 * sigma**2 * T - (r \
            - q) * sigma * T * sqrt(T) - sigma**2 * T**2 * (-log(S / K) / (2 * sigma * sqrt(T**3)) + (r - q - sigma**2 / 2) / (2 * sigma * sqrt(T))) / (2 * T**3 * sigma**2))
        return call_ParmiCharma, put_ParmiCharma
    
    def DualDelta(self):
        d2 = self.d2()
        T = self.time_to_mature
        r = self.interest
        K = self.str_price
        sigma = self.volatility
        call_DualDelta, put_DualDelta = -exp(-r * T) * norm.cdf(d2), exp(-r * T) * norm.cdf(-d2)
        return call_DualDelta, put_DualDelta
    
    def DualGamma(self):
        d2 = self.d2()
        T = self.time_to_mature
        r = self.interest
        sigma = self.volatility
        K = self.str_price
        DualDelta = exp(-r * T) * norm.pdf(d2) / (K * sigma * sqrt(T))
        return DualDelta, DualDelta

    def moneyness(self):
        S = self.cur_price
        K = self.str_price
        r = self.interest
        sigma = self.volatility
        T = time_to_mature       
        moneyness = (log(S / K) + r * T) / (sigma * sqrt(T))
        return moneyness, moneyness

def implied_volatility(T, r, S, K, q, price, option_type):
    if T <= 0 or price <= 0:
        return np.nan

    def objective_function(sigma):
        option = BlackScholesOption(T, sigma, r, S, K, q)
        call, put = option.price()
        return call - price if option_type == "call" else put - price

    try:
        implied_vol = brentq(objective_function, 1e-5, 5)
    except (ValueError, RuntimeError):
        implied_vol = np.nan

    return implied_vol

def volatility_surface(option_data, stock, r, q, str_min, str_max, option_type):
    def find_spot(stock):
        try:
            spot_history = stock.history(period='5d')
            if spot_history.empty:
                st.error(f'Failed to retrieve spot price data for {ticker}.')
                st.stop()
            else:
                spot_price = spot_history['Close'].iloc[-1]
        except Exception as e:
            st.error(f'An error occurred while fetching spot price data: {e}')
            st.stop()
        return spot_price

    def plot(str_or_mon, options_df):
        Y = options_df[str_or_mon].values
        y_label = 'Strike Price ($)' if str_or_mon == 'strike' else 'Moneyness (Strike / Spot)'

        X = options_df['timeToExpiration'].values
        Z = options_df['impliedVolatility'].values

        ti = np.linspace(X.min(), X.max(), 50)
        ki = np.linspace(Y.min(), Y.max(), 50)
        T, K = np.meshgrid(ti, ki)

        Zi = griddata((X, Y), Z, (T, K), method='linear')

        Zi = np.ma.array(Zi, mask=np.isnan(Zi))

        stock_fig = go.Figure(data=[go.Surface(
            x=T, y=K, z=Zi,
            colorscale='Viridis',
            colorbar_title='Implied Volatility (%)'
        )])

        stock_fig.update_layout(
            title=f'Implied Volatility Surface for {ticker} ' + option_type.capitalize() + ' Options with ' + str_or_mon.capitalize(),
            scene=dict(
                xaxis_title='Time to Expiration (years)',
                yaxis_title=y_label,
                zaxis_title='Implied Volatility (%)'
            ),
            autosize=False,
            width=900,
            height=800,
            margin=dict(l=65, r=50, b=65, t=90)
        )

        st.plotly_chart(stock_fig)

    if not option_data:
        st.error('No option data available after filtering.')
    else:
        options_df = pd.DataFrame(option_data)

        spot_price = find_spot(stock)

        options_df['daysToExpiration'] = (options_df['expirationDate'] - today).dt.days
        options_df['timeToExpiration'] = options_df['daysToExpiration'] / 365

        options_df = options_df[(options_df['strike'] >= spot_price * (str_min / 100)) & (options_df['strike'] <= spot_price * (str_max / 100))]

        options_df.reset_index(drop=True, inplace=True)

        bar_text = 'Calculating implied volatility...'
        plot_bar = st.progress(0, bar_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            plot_bar.progress(percent_complete + 1, text=bar_text)
        time.sleep(1)
        plot_bar.empty()

        options_df['impliedVolatility'] = options_df.apply(lambda row: implied_volatility(T=row['timeToExpiration'], r=r, S=spot_price, K=\
            row['strike'], q=q, price=row['mid'], option_type=option_type), axis=1)

        options_df.dropna(subset=['impliedVolatility'], inplace=True)

        options_df['impliedVolatility'] *= 100

        options_df.sort_values('strike', inplace=True)

        options_df['moneyness'] = options_df['strike'] / spot_price

        plot('strike', options_df)
        plot('moneyness', options_df)
    
st.set_page_config(page_title="Greeks and Volatility", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

with st.sidebar:
    st.title("Greeks and Implied Volatility")

    cur_price = st.number_input("Current Asset Price:", value=100.0)
    str_price = st.number_input("Strike Price:", value=100.0)
    time_to_mature = st.number_input("Time to Maturity (Years):", value=1.0)
    interest = st.number_input("Risk-Free Interest Rate:", value=0.05)
    volatility = st.number_input("Volatility (σ):", value=0.2)
    dividend_yield = st.number_input("Dividend Yield:", value=0.0)

    st.header("Implied Volatility Plot Parameters:")

    ticker = st.text_input("Enter Ticker Symbol:", value="SPY")
    str_min = st.number_input("Minimum Price as Percentage of Spot:", value=75.0, min_value=25.0, max_value=199.0)
    str_max = st.number_input("Maximum Price as Percentage of Spot:", value=125.0, min_value=str_min, max_value=200.0)

st.title("Greeks Calculator and Implied Volatility Surface Visualizer")
st.info("The following table lists the first-order (Delta, Theta, Vega, Rho, Epsilon, Lambda), second-order (Gamma, Vanna, Charm, \
        Vomma, Vera, Veta, Varpi), third-order (Speed, Zomma, Color, Ultima, ParmiCharma) Greeks in the Black-Scholes option \
        pricing model. Calculations for the Dual-Delta and Dual-Gamma are given as well for the local volatility.")

all_greeks = Greeks(time_to_mature, volatility, interest, cur_price, str_price, dividend_yield)

greek_methods = {
    "Delta": all_greeks.Delta, "Theta": all_greeks.Theta, "Vega": all_greeks.Vega, "Rho": all_greeks.Rho, "Epsilon": all_greeks.Epsilon, 
    "Lambda": all_greeks.Lambda, "Gamma": all_greeks.Gamma, "Vanna": all_greeks.Vanna, "Charm": all_greeks.Charm, "Vomma": all_greeks.Vomma, 
    "Vera": all_greeks.Vera, "Veta" : all_greeks.Veta, "Varpi": all_greeks.Varpi, "Speed": all_greeks.Speed, "Zomma": all_greeks.Zomma, 
    "Color": all_greeks.Color, "Ultima": all_greeks.Ultima, "ParmiCharma": all_greeks.ParmiCharma, "DualDelta": all_greeks.DualDelta, 
    "DualGamma": all_greeks.DualGamma, "Moneyness": all_greeks.moneyness
}

input_data = {}
for greek_name, method in greek_methods.items():
    call_value, put_value = method()
    input_data[greek_name] = [call_value, put_value]

option_type = ["Call Options", "Put Options"]
input_df = pd.DataFrame(input_data, index=option_type).T

html_table = (
    input_df.style
    .set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]},  # Center align headers
        {'selector': 'td', 'props': [('text-align', 'center')]},  # Center align cells
        {'selector': 'table', 'props': [('width', '100%')]},  # Set table width to 100%
    ])
    .set_caption("Greek Values for Options")
    .to_html()
)

custom_css = """
<style>
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th, td {
        text-align: center;
        padding: 8px;
    }
    caption {
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 10px;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)
st.markdown(html_table, unsafe_allow_html=True)

stock = yf.Ticker(ticker)

try:
    expir = stock.options
except Exception as e:
    st.error(f'Error fetching options for {ticker}: {e}')
    st.stop()

exp_dates = [pd.Timestamp(exp) for exp in expir if pd.Timestamp(exp) - timedelta(days=7) > today] # Filter by T > 7 days

if not exp_dates:
    st.error(f'No available option expiration dates for {ticker}.')
else:
    call_option_data = []
    put_option_data = []

    for exp_date in exp_dates:
        try:
            opt_chain = stock.option_chain(exp_date.strftime('%Y-%m-%d'))
            calls = opt_chain.calls
            puts = opt_chain.puts
        except Exception as e:
            st.warning(f'Failed to fetch option chain for {exp_date.date()}: {e}')
            continue

        calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)]
        puts = puts[(puts['bid'] > 0) & (puts['ask'] > 0)]

        for index, row in calls.iterrows():
            strike = row['strike']
            bid = row['bid']
            ask = row['ask']
            mid_price = (bid + ask) / 2

            call_option_data.append({'expirationDate': exp_date, 'strike': strike, 'bid': bid, 'ask': ask, 'mid': mid_price})

        for index, row in puts.iterrows():
            strike = row['strike']
            bid = row['bid']
            ask = row['ask']
            mid_price = (bid + ask) / 2

            put_option_data.append({'expirationDate': exp_date, 'strike': strike, 'bid': bid, 'ask': ask, 'mid': mid_price})

    volatility_surface(call_option_data, stock, interest, dividend_yield, str_min, str_max, "call")
    volatility_surface(put_option_data, stock, interest, dividend_yield, str_min, str_max, "put")