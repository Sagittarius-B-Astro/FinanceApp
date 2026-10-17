import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from models import (BlackScholesOption, BinomialOption, HestonOption,
                    compute_pnl_grids, plot_heatmaps)

st.set_page_config(page_title="Option Visualizer and Pricer", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.metric-container { display: flex; flex-direction: column; justify-content: center; align-items: center;
    text-align: center; padding: 8px; width: auto; margin: 0 auto; }
.metric-call { background-color: #03ac13; color: black; margin-right: 15px; border-radius: 20px; }
.metric-put { background-color: #d30000; color: black; border-radius: 20px; }
.metric-value { font-size: 1.5rem; font-weight: bold; margin: 0; }
.metric-label { font-size: 1rem; margin-bottom: 4px; }
</style>""", unsafe_allow_html=True)


def clamp_range(center, lo_mult=0.5, hi_mult=1.5, lo=0.01, hi=1.0):
    """Default slider range that always stays inside the slider's min/max."""
    return (min(max(center * lo_mult, lo), hi), min(max(center * hi_mult, lo), hi))


@st.cache_data(show_spinner="Computing heatmap...")
def cached_grids(model_type, p, spot_range, y_range, call_cost, put_cost, heston_axis):
    return compute_pnl_grids(model_type, dict(p), spot_range, y_range, call_cost, put_cost, heston_axis)


with st.sidebar:
    st.title("Option Visualizer and Pricer")

    call_cost = st.number_input("Price Paid for Call:", value=8.0, min_value=0.0)
    put_cost = st.number_input("Price Paid for Put:", value=6.0, min_value=0.0)
    cur_price = st.number_input("Current Asset Price:", value=100.0, min_value=0.01)
    str_price = st.number_input("Strike Price:", value=100.0, min_value=0.01)
    time_to_mature = st.number_input("Time to Maturity (Years):", value=1.0, min_value=0.01)
    interest = st.number_input("Risk-Free Interest Rate:", value=0.05, format="%.4f")
    volatility = st.number_input("Volatility (σ):", value=0.2, min_value=0.01, max_value=1.0)
    dividend_yield = st.number_input("Dividend Yield:", value=0.0, min_value=0.0, format="%.4f")
    n_steps = st.number_input("Number of Steps (for Binomial Model):", value=100, min_value=1, step=1)

    st.markdown("---")
    st.subheader("Heston Parameters")
    sigma = st.number_input("Volatility of Volatility:", value=0.25, min_value=0.01, max_value=1.0)
    kappa = st.number_input("Mean Reversion Rate:", value=2.0, min_value=0.01)
    theta_vol = st.number_input("Long-Term Average Volatility:", value=0.2, min_value=0.01, max_value=1.0)
    v0_vol = st.number_input("Initial Volatility:", value=0.2, min_value=0.01, max_value=1.0)
    rho = st.number_input("Correlation (Asset Price vs Volatility):", value=-0.5, min_value=-0.99,
                          max_value=0.99)
    # The Heston formulas work in VARIANCE, so convert the volatilities entered above.
    theta, v0 = theta_vol**2, v0_vol**2

    st.markdown("---")
    st.header("Heatmap Values")
    spot_min = st.number_input("Minimum Spot Price", value=75.0, min_value=0.01)
    spot_max = st.number_input("Maximum Spot Price", value=125.0, min_value=0.01)
    vol_min, vol_max = st.slider("Volatility Range for Heatmaps", min_value=0.01, max_value=1.0,
                                 value=clamp_range(volatility))
    sigma_min, sigma_max = st.slider("Volatility of Volatility Range for Heatmaps", min_value=0.01,
                                     max_value=1.0, value=clamp_range(sigma))
    heston_axis_label = st.radio("Heston heatmap y-axis", ["Volatility of Volatility", "Initial Volatility"])
    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)
    sigma_range = np.linspace(sigma_min, sigma_max, 10)

if spot_min >= spot_max:
    st.error("Minimum spot price must be below the maximum.")
    st.stop()

# Feller condition: 2*kappa*theta >= sigma^2 keeps the variance process strictly positive.
if 2 * kappa * theta < sigma**2:
    st.warning(f"Feller condition violated for the base inputs (2κθ = {2*kappa*theta:.3f} < σ² = {sigma**2:.3f}). "
               "Heston variance can hit zero; prices are still computed but treat them with caution.")
elif 2 * kappa * theta < sigma_max**2:
    st.warning("The Heston heatmap's upper volatility-of-volatility rows violate the Feller condition "
               f"(2κθ = {2*kappa*theta:.3f} < σ² = {sigma_max**2:.3f}).")

# ---- Main output ----
st.title("Option Pricing Models")
st.info("The call and put option prices are calculated using the Binomial, Black-Scholes, and Heston models given user input.")

bin_model = BinomialOption(time_to_mature, volatility, interest, cur_price, str_price, n_steps, dividend_yield)
bin_am_call, bin_am_put = bin_model.american_price()
bin_eu_call, bin_eu_put = bin_model.european_price()
bs_call, bs_put = BlackScholesOption(time_to_mature, volatility, interest, cur_price, str_price,
                                     dividend_yield).price()
hes_call, hes_put = HestonOption(time_to_mature, interest, cur_price, str_price, kappa, theta, sigma,
                                 rho, v0, dividend_yield).price()


def metric(kind, label, value):
    return f"""<div class="metric-container metric-{kind}"><div>
        <div class="metric-label">{label}</div><div class="metric-value">${value:.2f}</div>
        </div></div><br>"""


call_col, put_col = st.columns(2, gap="small")
with call_col:
    st.markdown(metric("call", "Binomial American CALL Value", bin_am_call)
                + metric("call", "Binomial European CALL Value", bin_eu_call)
                + metric("call", "Black-Scholes CALL Value", bs_call)
                + metric("call", "Heston CALL Value", hes_call), unsafe_allow_html=True)
with put_col:
    st.markdown(metric("put", "Binomial American PUT Value", bin_am_put)
                + metric("put", "Binomial European PUT Value", bin_eu_put)
                + metric("put", "Black-Scholes PUT Value", bs_put)
                + metric("put", "Heston PUT Value", hes_put), unsafe_allow_html=True)

st.markdown("")
st.title("Options PNL Heatmaps")
st.info("The heatmaps show profit/loss (model value minus the price you paid) across spot price and volatility. "
        "The strike price is held constant.")

base = dict(T=time_to_mature, r=interest, K=str_price, q=dividend_yield, n_steps=int(n_steps),
            kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0)
base_items = tuple(sorted(base.items()))  # hashable for st.cache_data

use_sigma_axis = heston_axis_label == "Volatility of Volatility"
heston_y = sigma_range if use_sigma_axis else vol_range
heston_ylabel = "Volatility of Volatility" if use_sigma_axis else "Initial Volatility"

heatmaps = [
    ("Binomial American", vol_range, "Volatility", "sigma"),
    ("Binomial European", vol_range, "Volatility", "sigma"),
    ("Black-Scholes", vol_range, "Volatility", "sigma"),
    ("Heston", heston_y, heston_ylabel, "sigma" if use_sigma_axis else "v0"),
]
figs = {}
for name, y_range, y_label, axis in heatmaps:
    call_grid, put_grid = cached_grids(name, base_items, spot_range, y_range, call_cost, put_cost, axis)
    figs[name] = plot_heatmaps(call_grid, put_grid, spot_range, y_range, y_label)

c_col, p_col = st.columns(2, gap="small")
for name, _, _, _ in heatmaps:
    fig_call, fig_put = figs[name]
    with c_col:
        st.subheader(f"{name} Call P&L Heatmap")
        st.pyplot(fig_call)
        plt.close(fig_call)
    with p_col:
        st.subheader(f"{name} Put P&L Heatmap")
        st.pyplot(fig_put)
        plt.close(fig_put)
