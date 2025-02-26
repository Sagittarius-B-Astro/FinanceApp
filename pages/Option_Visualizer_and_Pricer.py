import streamlit as st
import numpy as np
from Finance_Pricing_App import BlackScholesOption, BinaryOption, HestonOption, plot_heatmap

st.set_page_config(page_title="Option Visualizer and Pricer", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.metric-container {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 8px; /* Adjust the padding to control height */
    width: auto; /* Auto width for responsiveness, or set a fixed width if necessary */
    margin: 0 auto; /* Center the container */
}

.metric-call {
    background-color: #03ac13; /* Light green background */
    color: black; /* Black font color */
    margin-right: 15px; /* Spacing between CALL and PUT */
    border-radius: 20px; /* Rounded corners */
}

.metric-put {
    background-color: #d30000; /* Light red background */
    color: black; /* Black font color */
    border-radius: 20px; /* Rounded corners */
}

/* Style for the value text */
.metric-value {
    font-size: 1.5rem; /* Adjust font size */
    font-weight: bold;
    margin: 0; /* Remove default margins */
}

/* Style for the label text */
.metric-label {
    font-size: 1rem; /* Adjust font size */
    margin-bottom: 4px; /* Spacing between label and value */
}

</style>""", unsafe_allow_html=True)

with st.sidebar: #sidebar appearance
    st.title("Option Visualizer and Pricer")

    option_price = st.number_input("Price of Option:", value=8.0)
    cur_price = st.number_input("Current Asset Price:", value=100.0)
    str_price = st.number_input("Strike Price:", value=100.0)
    time_to_mature = st.number_input("Time to Maturity (Years):", value=1.0)
    interest = st.number_input("Risk-Free Interest Rate:", value=0.05)
    volatility = st.number_input("Volatility (σ):", value=0.2)
    dividend_yield = st.number_input("Dividend Yield:", value=0.0)
    n_steps = st.number_input("Number of Steps (for Binomial Model):", value=1)
    sigma = st.number_input("Volatility of Volatility (for Heston Model):", value=0.25)
    kappa = st.number_input("Mean Reversion Rate (for Heston Model):", value=2.0)
    theta = st.number_input("Long-Term Average Volatility (for Heston Model):", value=0.03)
    v0 = st.number_input("Initial Volatility (for Heston Model):", value=0.05)
    rho = st.number_input("Correlation Coefficient between Asset Price and Volatility (for Heston Model):", value=-0.5)

    st.markdown("---")

    st.header("Heatmap Values")
    spot_min = st.number_input("Minimum Spot Price", value=75.0)
    spot_max = st.number_input("Maximum Spot Price", value=125.0)
    vol_min, vol_max = st.slider("Volatility Range for Heatmaps", min_value=0.01, max_value=1.0, value=(volatility*0.5, volatility*1.5))
    sigma_min, sigma_max = st.slider("Volatility of Volatility Range for Heatmaps", min_value=0.01, max_value=1.0, value=(sigma*0.5, sigma*1.5))
    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)
    sigma_range = np.linspace(sigma_min, sigma_max, 10)

#main output
st.title("Option Pricing Models")
st.info("The call and put option prices are calculated using the Binomial, Black-Scholes, Heston models given user input.")

bin_model = BinaryOption(time_to_mature, volatility, interest, cur_price, str_price, n_steps, dividend_yield)
bin_am_call_price, bin_am_put_price = bin_model.American_price()
bin_eu_call_price, bin_eu_put_price = bin_model.European_price()

bs_model = BlackScholesOption(time_to_mature, volatility, interest, cur_price, str_price, dividend_yield)
bs_call_price, bs_put_price = bs_model.price()

hes_model = HestonOption(time_to_mature, interest, cur_price, str_price, kappa, theta, sigma, rho, v0)
hes_call_price, hes_put_price = hes_model.price()

# Display Call and Put Values in colored tables
call, put = st.columns([1,1], gap="small")

with call:
    # Using the custom class for CALL value
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">Binary American CALL Value</div>
                <div class="metric-value">${bin_am_call_price:.2f}</div>
            </div>    
        </div>
        <br>
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">Binary European CALL Value</div>
                <div class="metric-value">${bin_eu_call_price:.2f}</div>
            </div>
        </div>
        <br>
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">Black-Scholes CALL Value</div>
                <div class="metric-value">${bs_call_price:.2f}</div>
            </div>
        </div>
        <br>
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">Heston CALL Value</div>
                <div class="metric-value">${hes_call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with put:
    # Using the custom class for PUT value
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">Binary American PUT Value</div>
                <div class="metric-value">${bin_am_put_price:.2f}</div>
            </div>
        </div>
        <br>
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">Binary European PUT Value</div>
                <div class="metric-value">${bin_eu_put_price:.2f}</div>
            </div>
        </div>
        <br>
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">Black-Scholes PUT Value</div>
                <div class="metric-value">${bs_put_price:.2f}</div>
            </div>
        </div>
        <br>
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">Heston PUT Value</div>
                <div class="metric-value">${hes_put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.title("Options PNL Heatmaps")
st.info("The following heatmaps demonstrate how option prices change in response to different spot price and volatility ranges. The strike price remains constant.")

call, put = st.columns([1,1], gap="small")

bin_am_heatmap_fig_call, bin_am_heatmap_fig_put = plot_heatmap(spot_range, vol_range, str_price, option_price, bin_model, "Binary American")
bin_eu_heatmap_fig_call, bin_eu_heatmap_fig_put = plot_heatmap(spot_range, vol_range, str_price, option_price, bin_model, "Binary European")
bs_heatmap_fig_call, bs_heatmap_fig_put = plot_heatmap(spot_range, vol_range, str_price, option_price, bs_model, "Black-Scholes")
hes_heatmap_fig_call, hes_heatmap_fig_put = plot_heatmap(spot_range, sigma_range, str_price, option_price, hes_model, "Heston")

with call:
    st.subheader("Binary American Call P&L Heatmap")
    st.pyplot(bin_am_heatmap_fig_call)

    st.subheader("Binary European Call P&L Heatmap")
    st.pyplot(bin_eu_heatmap_fig_call)

    st.subheader("Black-Scholes Call P&L Heatmap")
    st.pyplot(bs_heatmap_fig_call)

    st.subheader("Heston Call P&L Heatmap")
    st.pyplot(hes_heatmap_fig_call)

with put:
    st.subheader("Binary American Put P&L Heatmap")
    st.pyplot(bin_am_heatmap_fig_put)

    st.subheader("Binary European Put P&L Heatmap")
    st.pyplot(bin_eu_heatmap_fig_put)

    st.subheader("Black-Scholes Put P&L Heatmap")
    st.pyplot(bs_heatmap_fig_put)

    st.subheader("Heston Put P&L Heatmap")
    st.pyplot(hes_heatmap_fig_put)