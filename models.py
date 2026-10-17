"""Pricing models and heatmap helpers (no Streamlit code here, so it is safe to import from any page)."""
import numpy as np
from numpy import log, sqrt, exp
from scipy.stats import norm
from scipy.integrate import quad
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns


class BlackScholesOption:
    def __init__(self, time_to_mature, volatility, interest, cur_price, str_price, dividend_yield=0.0):
        self.time_to_mature = time_to_mature
        self.volatility = volatility
        self.interest = interest
        self.cur_price = cur_price
        self.str_price = str_price
        self.div_yield = dividend_yield

    def d1(self):
        T, sigma, r, S, K, q = (self.time_to_mature, self.volatility, self.interest,
                                self.cur_price, self.str_price, self.div_yield)
        return (log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * sqrt(T))

    def d2(self):
        return self.d1() - self.volatility * sqrt(self.time_to_mature)

    def price(self):
        T, r, S, K, q = (self.time_to_mature, self.interest, self.cur_price,
                         self.str_price, self.div_yield)
        d1, d2 = self.d1(), self.d2()
        # FIX: spot must be discounted by exp(-qT); the old version ignored dividends in the price.
        call = S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        put = K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)
        return call, put


class BinomialOption:
    """CRR binomial tree (renamed from BinaryOption: 'binary' means digital option)."""

    def __init__(self, time_to_mature, volatility, interest, cur_price, str_price, n_steps, dividend_yield=0.0):
        self.time_to_mature = time_to_mature
        self.volatility = volatility
        self.interest = interest
        self.cur_price = cur_price
        self.str_price = str_price
        self.n_steps = int(n_steps)
        self.div_yield = dividend_yield

    def _price(self, american):
        T, sigma, r, S, K, n, q = (self.time_to_mature, self.volatility, self.interest,
                                   self.cur_price, self.str_price, self.n_steps, self.div_yield)
        dt = T / n
        up = exp(sigma * sqrt(dt))
        disc_up = (up * exp(-q * dt) - exp(-r * dt)) / (up**2 - 1)
        disc_dn = exp(-r * dt) - disc_up

        i = np.arange(n + 1)
        spot = S * up ** (2 * i - n)
        call, put = np.maximum(spot - K, 0.0), np.maximum(K - spot, 0.0)
        for j in range(n - 1, -1, -1):          # vectorised over nodes
            call = disc_up * call[1:] + disc_dn * call[:-1]
            put = disc_up * put[1:] + disc_dn * put[:-1]
            if american:
                spot = S * up ** (2 * np.arange(j + 1) - j)
                call = np.maximum(call, spot - K)
                put = np.maximum(put, K - spot)
        return call[0], put[0]

    def american_price(self):
        return self._price(True)

    def european_price(self):
        return self._price(False)


class HestonOption:
    """Heston model priced with the P1/P2 Fourier formulas and the 'little trap' characteristic function.

    theta and v0 are VARIANCES (e.g. 0.04 = 20% vol). sigma is the vol-of-vol.
    """

    def __init__(self, time_to_mature, interest, cur_price, str_price, kappa, theta, sigma, rho, v0,
                 dividend_yield=0.0):
        self.time_to_mature = time_to_mature
        self.interest = interest
        self.cur_price = cur_price
        self.str_price = str_price
        self.kappa, self.theta, self.sigma = kappa, theta, sigma
        self.rho, self.v0 = rho, v0
        self.div_yield = dividend_yield

    def _char_func(self, u):
        T, r, q = self.time_to_mature, self.interest, self.div_yield
        S0, kappa, theta, sigma, rho, v0 = (self.cur_price, self.kappa, self.theta,
                                            self.sigma, self.rho, self.v0)
        b = kappa - rho * sigma * 1j * u
        d = np.sqrt(b**2 + sigma**2 * (1j * u + u**2))
        g = (b - d) / (b + d)
        e = np.exp(-d * T)
        C = (r - q) * 1j * u * T + kappa * theta / sigma**2 * (
            (b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
        D = (b - d) / sigma**2 * (1 - e) / (1 - g * e)
        return np.exp(C + D * v0 + 1j * u * np.log(S0))

    def price(self, u_max=200):
        T, r, q = self.time_to_mature, self.interest, self.div_yield
        S0, K = self.cur_price, self.str_price
        lnK = np.log(K)
        phi_mi = self._char_func(-1j)  # E[S_T]

        f1 = lambda u: np.real(np.exp(-1j * u * lnK) * self._char_func(u - 1j) / (1j * u * phi_mi))
        f2 = lambda u: np.real(np.exp(-1j * u * lnK) * self._char_func(u) / (1j * u))
        P1 = 0.5 + quad(f1, 1e-8, u_max, limit=500)[0] / np.pi
        P2 = 0.5 + quad(f2, 1e-8, u_max, limit=500)[0] / np.pi

        call = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
        put = call - S0 * np.exp(-q * T) + K * np.exp(-r * T)  # put-call parity
        return call, put


def compute_pnl_grids(model_type, p, spot_range, y_range, call_cost, put_cost, heston_axis="sigma"):
    """Return (call_pnl, put_pnl), each shaped (len(y_range), len(spot_range)).

    p: dict of base parameters. For Heston, y_range is vol-of-vol (heston_axis="sigma")
    or initial volatility (heston_axis="v0", given as a volatility, squared internally).
    """
    call_grid = np.zeros((len(y_range), len(spot_range)))
    put_grid = np.zeros_like(call_grid)
    for i, y in enumerate(y_range):
        for j, spot in enumerate(spot_range):
            if model_type in ("Binomial American", "Binomial European"):
                m = BinomialOption(p["T"], y, p["r"], spot, p["K"], p["n_steps"], p["q"])
                c, pt = m.american_price() if model_type == "Binomial American" else m.european_price()
            elif model_type == "Black-Scholes":
                c, pt = BlackScholesOption(p["T"], y, p["r"], spot, p["K"], p["q"]).price()
            else:  # Heston
                sigma = y if heston_axis == "sigma" else p["sigma"]
                v0 = p["v0"] if heston_axis == "sigma" else y**2
                c, pt = HestonOption(p["T"], p["r"], spot, p["K"], p["kappa"], p["theta"],
                                     sigma, p["rho"], v0, p["q"]).price()
            call_grid[i, j] = c - call_cost
            put_grid[i, j] = pt - put_cost
    return call_grid, put_grid


def plot_heatmaps(call_grid, put_grid, spot_range, y_range, y_label):
    cmap = LinearSegmentedColormap.from_list("pnl", ["red", "white", "green"])
    figs = []
    for grid, title in ((call_grid, "CALL"), (put_grid, "PUT")):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(grid, xticklabels=np.round(spot_range, 2), yticklabels=np.round(y_range, 3),
                    annot=True, fmt=".2f", cmap=cmap, center=0, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Spot Price")
        ax.set_ylabel(y_label)
        ax.invert_yaxis()  # low values at the bottom, like a normal axis
        figs.append(fig)
    return figs
