import streamlit as st

st.set_page_config(page_title="Finance Pricing App", page_icon="💸", layout="wide", initial_sidebar_state="expanded")

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
         and its prices were checked against a Monte-Carlo simulation and the Black-Scholes limit (as vol-of-vol approaches zero).")
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