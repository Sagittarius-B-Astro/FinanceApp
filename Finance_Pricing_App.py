import streamlit as st
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from numpy import log, sqrt, exp, real, inf, pi
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns

class BlackScholesOption:
    def __init__(self, time_to_mature, volatility, interest, cur_price, str_price, dividend_yield):
        self.time_to_mature = time_to_mature
        self.volatility = volatility
        self.interest = interest
        self.cur_price = cur_price
        self.str_price = str_price
        self.div_yield = dividend_yield
    
    def d1(self):
        T = self.time_to_mature
        sigma = self.volatility
        r = self.interest
        S = self.cur_price
        K = self.str_price
        div_yield = self.div_yield
        return (log(S / K) + (r - div_yield + (sigma**2) / 2) * T) / (sigma * sqrt(T))

    def d2(self):
        return self.d1() - self.volatility * sqrt(self.time_to_mature)

    def price(self):
        T = self.time_to_mature
        r = self.interest
        S = self.cur_price
        K = self.str_price

        call_price = norm.cdf(self.d1()) * S - (norm.cdf(self.d2()) * K * exp(-(r * T)))
        put_price = (norm.cdf(-self.d2()) * K * exp(-(r * T))) - norm.cdf(-self.d1()) * S

        return call_price, put_price

class BinaryOption:
    def __init__(self, time_to_mature, volatility, interest, cur_price, str_price, n_steps, dividend_yield):
        self.time_to_mature = time_to_mature
        self.volatility = volatility
        self.interest = interest
        self.cur_price = cur_price
        self.str_price = str_price
        self.n_steps = n_steps
        self.div_yield = dividend_yield
    
    def American_price(self):
        time_to_mature = self.time_to_mature
        volatility = self.volatility
        interest = self.interest
        cur_price = self.cur_price
        str_price = self.str_price
        n_steps = self.n_steps
        div_yield = self.div_yield

        delta_T = time_to_mature/n_steps
        up = exp(volatility*sqrt(delta_T))
        discount_up = (up * exp(-div_yield * delta_T) - exp(-interest * delta_T)) / (up ** 2 - 1)
        discount_dn = exp(-interest * delta_T) - discount_up

        call, put = [0] * (n_steps + 1), [0] * (n_steps + 1)
        for i in range(n_steps+1): # Payoff 
            call[i] = max(cur_price * up ** (2 * i - n_steps) - str_price, 0)
            put[i] = max(str_price - cur_price * up ** (2 * i - n_steps), 0)
        for j in range(n_steps-1, -1, -1):
            for i in range(j + 1):
                call[i] = discount_up * call[i+1] + discount_dn * call[i]
                put[i] = discount_up * put[i+1] + discount_dn * put[i]
                call_exercise = cur_price * up ** (2 * i - j) - str_price
                put_exercise = str_price - cur_price * up ** (2 * i - j)
                call[i] = max(call[i], call_exercise)
                put[i] = max(put[i], put_exercise)
 
        return call[0], put[0]
    
    def European_price(self):
        time_to_mature = self.time_to_mature
        volatility = self.volatility
        interest = self.interest
        cur_price = self.cur_price
        str_price = self.str_price
        n_steps = self.n_steps
        div_yield = self.div_yield

        delta_T = time_to_mature/n_steps
        up = exp(volatility*sqrt(delta_T))
        discount_up = (up * exp(-div_yield * delta_T) - exp(-interest * delta_T)) / (up ** 2 - 1)
        discount_dn = exp(-interest * delta_T) - discount_up

        call, put = [0] * (n_steps + 1), [0] * (n_steps + 1)
        for i in range(n_steps+1): # Payoff 
            call[i] = max(cur_price * up ** (2 * i - n_steps) - str_price, 0)
            put[i] = max(str_price - cur_price * up ** (2 * i - n_steps), 0)
        for j in range(n_steps-1, -1, -1):
            for i in range(j + 1):
                call[i] = discount_up * call[i+1] + discount_dn * call[i]
                put[i] = discount_up * put[i+1] + discount_dn * put[i]
 
        return call[0], put[0]

class HestonOption:
    def __init__(self, time_to_mature, interest, cur_price, str_price, kappa, theta, sigma, rho, v0):
        self.time_to_mature = time_to_mature
        self.interest = interest
        self.cur_price = cur_price
        self.str_price = str_price
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0

    def price(self):
        T = self.time_to_mature
        r = self.interest
        S0 = self.cur_price
        K = self.str_price
        kappa = self.kappa
        theta = self.theta
        sigma = self.sigma
        rho = self.rho
        v0 = self.v0

        def heston_char_func(x, S0, K, r, T, kappa, theta, sigma, rho, v0):
            k = kappa - rho * sigma * 1j * x
            M = sqrt((rho * sigma * 1j * x - k)**2 + sigma**2 * (x * 1j + x**2))
            N = (rho * sigma * 1j * x - k + M) / (rho * sigma * 1j * x - k - M)
            A = r * 1j * x * T + (kappa * theta) / sigma**2 * ((k - rho * sigma * 1j * x - M) * T - 2 * log((1 - N * exp(-M * T)) / (1 - N)))
            C = (rho * sigma * 1j * x - k + M) * (exp(-M * T) - 1) / (sigma**2 * (1 - N * exp(-M * T)))
            return exp(A + C * v0 + 1j * x * log(S0))

        integrand = lambda x: real(exp(-1j * x * log(K)) / (1j * x) * heston_char_func(x - 1j, S0, K, r, T, kappa, theta, sigma, rho, v0))
        integral, _ = quad(integrand, 0, inf)
        call = exp(-r * T) * 0.5 * S0 - exp(-r * T) / pi * integral
        put = exp(-r * T) / pi * integral - S0 + K * exp(-r *T)
        return call, put

def plot_heatmap(spot_range, vol_range, strike, option_price, model, model_type):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            if model_type == "Binary American":
                bin_temp = BinaryOption(time_to_mature=model.time_to_mature, volatility=vol, interest=model.interest, 
                cur_price=spot, str_price=strike, n_steps=model.n_steps, dividend_yield=model.div_yield)
                call, put = bin_temp.American_price()
                call_prices[i, j] = call - option_price
                put_prices[i, j] = put - option_price
            elif model_type == "Binary European":
                bin_temp = BinaryOption(time_to_mature=model.time_to_mature, volatility=vol, interest=model.interest, 
                cur_price=spot, str_price=strike, n_steps=model.n_steps, dividend_yield=model.div_yield)
                call, put = bin_temp.European_price()
                call_prices[i, j] = call - option_price
                put_prices[i, j] = put - option_price
            elif model_type == "Black-Scholes":
                bs_temp = BlackScholesOption(time_to_mature=model.time_to_mature, volatility=vol, 
                interest=model.interest, cur_price=spot, str_price=strike, dividend_yield=model.div_yield)
                call, put = bs_temp.price()
                call_prices[i, j] = call - option_price
                put_prices[i, j] = put - option_price
            else:
                hes_temp = HestonOption(time_to_mature=model.time_to_mature, interest=model.interest, cur_price=spot, 
                str_price=strike, kappa=model.kappa, theta=model.theta, sigma=vol, rho=model.rho, v0=model.v0)
                call, put = hes_temp.price()
                call_prices[i, j] = call - option_price
                put_prices[i, j] = put - option_price 
    
    colors = ['red', 'white', 'green']
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap=custom_cmap, center=0, ax=ax_call)
    ax_call.set_title('CALL')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility of Volatility' if model_type == "Heston" else 'Volatility')
    
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap=custom_cmap, center=0, ax=ax_put)
    ax_put.set_title('PUT')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility of Volatility' if model_type == "Heston" else 'Volatility')
    
    return fig_call, fig_put

st.set_page_config(page_title="Finance Pricing App", page_icon=":money_with_wings", layout="wide", initial_sidebar_state="expanded")

with st.sidebar: #sidebar appearance
    st.title("Finance Pricing App")
    st.markdown(f'<a href="{"https://www.linkedin.com/in/yejun-kim-212779142/"}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Yejun Moon Kim`</a>', unsafe_allow_html=True)

st.title("Finance Pricing App")
st.info("This app was created to put into practice my knowledge of finance, and to make more calculated decisions in my \
        personal portfolio. It contains pages for option pricing, greeks, implied volatility, and Value-at-Risk.")
st.header("Option Visualizer and Pricer")
st.write("In this page, three pricing models (Binomial, Black-Scholes, Heston) are used to calculate the price of an option. \
         The Binomial Model assumes that the stock price varies at each time step by an up or down factor, such that up = 1/down.\
         This model can be used to calculate the price of American and European options, which differ in that American options allow \
         early execution of the option, whereas the European option must be executed at expiry. The Black-Scholes option takes in \
         the same inputs and represents a continuous-time approach to option pricing. This model excels when the underlying asset \
         follows a log-normal distribution and has a constant volatility. The Heston model accounts for this shortcoming by using a \
         set of parameters that represents the random motion of volatility. However, the formula used for this model is more complex \
         and frankly, I am not sure whether my calculations are correct. I am considering using a Monte-Carlo approach instead.")
st.header("Greeks and Implied Volatility")
st.write("The Greeks are often used as financial indicators to assess risk of an asset. There are five primary greeks, which are \
         Delta, Theta, Gamma, Vega, Rho. Other Greeks that are increasingly being used are Lambda, Epsilon, Vomma, Vera, Zomma, \
         and Ultima. Delta represents the rate of change between option price and underlying asset price. Theta is the rate of change \
         option price and time. Gamma is the rate of change between Delta and underlying asset price. Vega represents the rate of \
         change between the option price and implied volatility of the asset. Rho represents the rate of change between the option \
         price and interest rate. Lambda represents rate of change between Delta relative to implied volatility. Epsilon is the rate \
         of change of the option's price relative to dividend rate. Vomma is the second derivative of option price to implied \
         volatility. Vera is Rho's sensitivity to volatility. Zomma is Gamma's sensitivity to volatility. Ultima is Vomma's rate of \
         change relative to changes in volatility. Implied Volatility is a measure of how much the market expects an asset to fluctuate \
         , and can be found by backing out the asset's price in the market.")
st.header("Value at Risk Calculator")
st.write("In finance, Value at Risk (VaR) is a metric for the greatest amount of losses over a given time frame. Used by financial firms \
         banks in investment analysis, it allows people to see the amount of risk they are exposed to through their portfolio. There are \
         various methods for calculation of VaR, such as the historical, parametric, and Monte Carlo simulation methods. Due to its \
         versatility and simple application, it can be used on stocks, bonds, and currencies, and is used widely in portfolio management. \
         However, each method has its disadvantages, with the historical method often giving a pessimistic outlook, while the Monte Carlo \
         method is optimistic. It also does not account for correlations between assets, which impact the calculated risk. This tool \
         allows the user to find the VaR for a given portfolio, with a time range selected by the user.")