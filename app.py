"""
Motor Insurance Portfolio Analysis - Interactive Streamlit App
Based on the actuarial analysis by Mazy Djezzar, Prescilya Fabi, Samantha López
HEC - University of Lausanne, Master's in Actuarial Science
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Motor Insurance Portfolio Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a cleaner look
st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1300px; }
    h1, h2, h3 { color: #1f3a5f; }
    .metric-card {
        background-color: #f0f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4682B4;
    }
    div[data-testid="stMetricValue"] { color: #1f3a5f; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOADING
# ============================================================
GSHEET_ID = "1FmcbyY42xezqifz1xMic33zDGUZKcR1r6koeDlWSD1I"
GSHEET_CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{GSHEET_ID}/gviz/tq?tqx=out:csv&sheet=data"
)


@st.cache_data(show_spinner="Loading data...")
def load_data_from_url(url: str) -> pd.DataFrame:
    """Load data from a Google Sheets CSV export URL."""
    df = pd.read_csv(url, na_values=[" -   ", "-", "NA", "", " "])
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data(show_spinner="Loading data...")
def load_data_from_file(uploaded_file) -> pd.DataFrame:
    """Load data from an uploaded CSV file (semicolon-separated)."""
    content = uploaded_file.read()
    # Try semicolon first (original format), then comma
    for sep in [";", ","]:
        try:
            df = pd.read_csv(
                io.BytesIO(content), sep=sep, encoding="utf-8-sig",
                na_values=[" -   ", "-", "NA", "", " "],
            )
            if df.shape[1] > 5:  # Heuristic: real data has many columns
                df.columns = [c.strip() for c in df.columns]
                return df
        except Exception:
            continue
    raise ValueError("Could not parse CSV file.")


@st.cache_data(show_spinner="Loading data...")
def load_default_data() -> pd.DataFrame:
    """Load the bundled CSV file."""
    df = pd.read_csv(
        "data/DATA_SET_2.csv",
        sep=";", encoding="utf-8-sig",
        na_values=[" -   ", "-", "NA", "", " "],
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_data(df: pd.DataFrame):
    """Extract claim_count and claim_severity from the raw dataframe."""
    claim_count = df["CLM_FREQ"].astype(int).values
    amt_cols = [c for c in df.columns if c.startswith("CLM_AMT")]
    for c in amt_cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce")
    sev_raw = df[amt_cols].values.flatten()
    sev_raw = sev_raw[~np.isnan(sev_raw)]
    claim_severity = sev_raw[sev_raw > 0]  # remove negatives
    n_removed = len(sev_raw) - len(claim_severity)
    return claim_count, claim_severity, n_removed, amt_cols


# ============================================================
# MODEL FITTING
# ============================================================
@st.cache_data
def fit_poisson(claim_count):
    lam = float(np.mean(claim_count))
    ll = float(stats.poisson.logpmf(claim_count, lam).sum())
    aic = 2 * 1 - 2 * ll
    bic = np.log(len(claim_count)) * 1 - 2 * ll
    return {"lambda": lam, "loglik": ll, "aic": aic, "bic": bic}


@st.cache_data
def fit_negbinom(claim_count):
    """Fit NB by MLE. Returns R-style parameters: size, mu, plus scipy n,p."""
    def neg_ll(params, x):
        n, p = params
        if n <= 0 or p <= 0 or p >= 1:
            return 1e10
        return -stats.nbinom.logpmf(x, n, p).sum()

    m = float(np.mean(claim_count))
    v = float(np.var(claim_count, ddof=1))
    p0 = m / v if v > m else 0.5
    n0 = m * p0 / (1 - p0) if p0 < 1 else 1.0
    res = minimize(neg_ll, [n0, p0], args=(claim_count,), method="Nelder-Mead")
    n_nb, p_nb = res.x
    ll = float(-res.fun)
    aic = 2 * 2 - 2 * ll
    bic = np.log(len(claim_count)) * 2 - 2 * ll
    mu = n_nb * (1 - p_nb) / p_nb
    return {"size": n_nb, "p": p_nb, "mu": mu, "loglik": ll, "aic": aic, "bic": bic}


@st.cache_data
def fit_severity_models(claim_severity):
    """Fit Gamma, Lognormal, Weibull. Returns dict of params + AIC/BIC."""
    n = len(claim_severity)
    out = {}

    # Gamma (loc=0)
    a, _, scale_g = stats.gamma.fit(claim_severity, floc=0)
    ll_g = stats.gamma.logpdf(claim_severity, a, loc=0, scale=scale_g).sum()
    out["Gamma"] = {
        "shape": a, "rate": 1 / scale_g, "scale": scale_g,
        "loglik": ll_g, "aic": 2 * 2 - 2 * ll_g, "bic": np.log(n) * 2 - 2 * ll_g,
    }

    # Lognormal (loc=0)
    sigma, _, scale_ln = stats.lognorm.fit(claim_severity, floc=0)
    ll_ln = stats.lognorm.logpdf(claim_severity, sigma, loc=0, scale=scale_ln).sum()
    out["Lognormal"] = {
        "meanlog": np.log(scale_ln), "sdlog": sigma, "scipy_s": sigma, "scipy_scale": scale_ln,
        "loglik": ll_ln, "aic": 2 * 2 - 2 * ll_ln, "bic": np.log(n) * 2 - 2 * ll_ln,
    }

    # Weibull (loc=0)
    c, _, scale_w = stats.weibull_min.fit(claim_severity, floc=0)
    ll_w = stats.weibull_min.logpdf(claim_severity, c, loc=0, scale=scale_w).sum()
    out["Weibull"] = {
        "shape": c, "scale": scale_w,
        "loglik": ll_w, "aic": 2 * 2 - 2 * ll_w, "bic": np.log(n) * 2 - 2 * ll_w,
    }

    return out


# ============================================================
# GOODNESS-OF-FIT
# ============================================================
def chisq_gof_freq(claim_count, fit, distr):
    """Chi-squared GOF for discrete frequency models, with bin pooling."""
    obs_table = pd.Series(claim_count).value_counts().sort_index()
    k_vals = obs_table.index.values
    obs_vec = obs_table.values.astype(float)
    n = len(claim_count)

    if distr == "pois":
        probs = stats.poisson.pmf(k_vals, fit["lambda"])
        n_params = 1
    else:
        probs = stats.nbinom.pmf(k_vals, fit["size"], fit["p"])
        n_params = 2

    expected = n * probs
    obs_vec = obs_vec.copy()

    # Pool bins with expected < 5
    while np.any(expected < 5) and len(expected) > 2:
        i = int(np.where(expected < 5)[0][0])
        if i == len(expected) - 1:
            i -= 1
        obs_vec[i + 1] += obs_vec[i]
        expected[i + 1] += expected[i]
        obs_vec = np.delete(obs_vec, i)
        expected = np.delete(expected, i)

    chi2 = float(np.sum((obs_vec - expected) ** 2 / expected))
    df = max(len(obs_vec) - 1 - n_params, 1)
    p_value = float(1 - stats.chi2.cdf(chi2, df))
    return {"chi2": chi2, "df": df, "p_value": p_value}


def ks_test_severity(claim_severity, sev_fits):
    out = {}
    out["Gamma"] = stats.kstest(
        claim_severity, "gamma",
        args=(sev_fits["Gamma"]["shape"], 0, sev_fits["Gamma"]["scale"]),
    )
    out["Lognormal"] = stats.kstest(
        claim_severity, "lognorm",
        args=(sev_fits["Lognormal"]["scipy_s"], 0, sev_fits["Lognormal"]["scipy_scale"]),
    )
    out["Weibull"] = stats.kstest(
        claim_severity, "weibull_min",
        args=(sev_fits["Weibull"]["shape"], 0, sev_fits["Weibull"]["scale"]),
    )
    return out


# ============================================================
# MONTE CARLO
# ============================================================
@st.cache_data
def simulate_aggregate_loss(n_sim, freq_model, freq_params, sev_model, sev_params, seed=42):
    """Simulate aggregate losses for n_sim policies."""
    rng = np.random.default_rng(seed)

    # Frequency
    if freq_model == "Poisson":
        sim_counts = rng.poisson(freq_params["lambda"], size=n_sim)
    else:  # NegBinomial
        # scipy uses (n, p); we have size, p
        sim_counts = stats.nbinom.rvs(
            freq_params["size"], freq_params["p"], size=n_sim,
            random_state=rng,
        )

    # Severity sums per policy
    agg_loss = np.zeros(n_sim)
    total_claims = int(sim_counts.sum())
    if total_claims == 0:
        return agg_loss, sim_counts

    if sev_model == "Gamma":
        all_sev = rng.gamma(sev_params["shape"], scale=sev_params["scale"], size=total_claims)
    elif sev_model == "Lognormal":
        all_sev = rng.lognormal(sev_params["meanlog"], sev_params["sdlog"], size=total_claims)
    else:  # Weibull
        all_sev = sev_params["scale"] * rng.weibull(sev_params["shape"], size=total_claims)

    # Distribute claims to policies
    idx = 0
    for i, nc in enumerate(sim_counts):
        if nc > 0:
            agg_loss[i] = all_sev[idx:idx + nc].sum()
            idx += nc

    return agg_loss, sim_counts


# ============================================================
# SIDEBAR — DATA SOURCE & NAVIGATION
# ============================================================
st.sidebar.title("🚗 Motor Insurance")
st.sidebar.caption("Portfolio Analysis Dashboard")

st.sidebar.divider()
st.sidebar.subheader("📂 Data source")
data_source = st.sidebar.radio(
    "Choose where to load data from:",
    ["Bundled CSV (default)", "Google Sheets URL", "Upload CSV"],
    index=0,
    label_visibility="collapsed",
)

df = None
load_error = None

try:
    if data_source == "Bundled CSV (default)":
        df = load_default_data()
    elif data_source == "Google Sheets URL":
        url = st.sidebar.text_input("Google Sheets CSV URL", value=GSHEET_CSV_URL)
        st.sidebar.caption("⚠️ The sheet must be shared as 'Anyone with the link'.")
        if url:
            df = load_data_from_url(url)
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df = load_data_from_file(uploaded)
except Exception as e:
    load_error = str(e)

if df is None:
    if load_error:
        st.error(f"❌ Failed to load data: {load_error}")
    else:
        st.info("⬅️ Please select or upload a data source from the sidebar.")
    st.stop()

# Clean data
claim_count, claim_severity, n_removed, amt_cols = clean_data(df.copy())
n_policies = len(df)
premium = df["PREMIUM"].values
fees = df["FEES"].values

# Fit models (cached)
fit_pois = fit_poisson(claim_count)
fit_nb = fit_negbinom(claim_count)
sev_fits = fit_severity_models(claim_severity)

# Determine best models
best_freq_name = "Negative Binomial" if fit_nb["aic"] < fit_pois["aic"] else "Poisson"
best_freq_params = fit_nb if best_freq_name == "Negative Binomial" else fit_pois

best_sev_name = min(sev_fits, key=lambda k: sev_fits[k]["aic"])
best_sev_params = sev_fits[best_sev_name]

st.sidebar.divider()
st.sidebar.subheader("🧭 Navigation")
page = st.sidebar.radio(
    "Section",
    [
        "🏠 Executive Summary",
        "📊 Data Analysis",
        "🎲 Frequency Model",
        "💰 Severity Model",
        "🔄 Monte Carlo & Variance Reduction",
        "📈 Risk Premium & VaR",
        "🛡️ Reinsurance",
        "🔬 Sensitivity Analysis",
        "📝 Conclusion & Recommendations",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption(
    f"**Best frequency model:** {best_freq_name}\n\n"
    f"**Best severity model:** {best_sev_name}"
)
st.sidebar.caption(
    "Mazy Djezzar · Prescilya Fabi · Samantha López\n\n"
    "HEC - University of Lausanne"
)


# ============================================================
# COMMON SIMULATION (used across pages)
# ============================================================
N_SIM_DEFAULT = 50000


@st.cache_data
def get_main_simulation(n_sim, best_freq_name, best_freq_params, best_sev_name, best_sev_params):
    return simulate_aggregate_loss(
        n_sim, best_freq_name, best_freq_params, best_sev_name, best_sev_params, seed=42
    )


# ============================================================
# PAGE: EXECUTIVE SUMMARY
# ============================================================
if page == "🏠 Executive Summary":
    st.title("🚗 Motor Insurance Portfolio Analysis")
    st.markdown(
        "**HEC – University of Lausanne** · Simulation Methods in Finance and Insurance"
    )
    st.caption("Mazy Djezzar · Prescilya Fabi · Samantha López")
    st.divider()

    # Headline metrics
    avg_premium = float(np.mean(premium))
    avg_fees = float(np.mean(fees))
    total_claims_data = float(np.sum(claim_severity))
    combined_ratio = (total_claims_data + np.sum(fees)) / np.sum(premium)
    risk_premium_emp = total_claims_data / n_policies

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Policyholders", f"{n_policies:,}")
    c2.metric("Total Claims", f"{len(claim_severity):,}")
    c3.metric("Combined Ratio", f"{combined_ratio:.3f}",
              delta="Profitable" if combined_ratio < 1 else "Unprofitable")
    c4.metric("Profit Margin", f"{(1 - combined_ratio) * 100:.2f}%")

    st.divider()
    st.subheader("Executive Summary")
    st.markdown(
        f"""
The analysis focuses on a motor insurance portfolio of **{n_policies:,} policyholders**
to assess whether the existing pricing approach remains viable and whether
lowering premiums could be justified.

The portfolio is currently **profitable**: a combined ratio of **{combined_ratio:.2f}**
indicates that premium income comfortably covers both claims and operating costs,
leaving a margin of **{(1 - combined_ratio) * 100:.2f}%**.

However, this margin leaves little flexibility. **Any premium reduction exceeding
{(1 - combined_ratio) * 100:.2f}%** would put the company in an unprofitable position.

Our risk assessment indicates that, under severe scenarios, aggregate losses
could reach nearly **four times** the average expected loss. Although the company
remains profitable under normal conditions, such events represent a significant
financial threat that cannot be ignored.

**Recommendations:**
1. **Maintain the current tariff** to preserve profitability.
2. **Implement an Excess of Loss reinsurance contract** to protect against
   exceptionally large losses, ensuring long-term financial stability.
        """
    )

    st.divider()
    st.subheader("How this dashboard is organised")
    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            """
- **📊 Data Analysis** — explore the raw portfolio data
- **🎲 Frequency Model** — Poisson vs Negative Binomial
- **💰 Severity Model** — Gamma vs Lognormal vs Weibull
- **🔄 Monte Carlo & Variance Reduction** — convergence and antithetic / control variates
            """
        )
    with cols[1]:
        st.markdown(
            """
- **📈 Risk Premium & VaR** — aggregate loss distribution, SCR
- **🛡️ Reinsurance** — XoL retention strategy
- **🔬 Sensitivity Analysis** — robustness across model combinations
- **📝 Conclusion** — final recommendations
            """
        )


# ============================================================
# PAGE: DATA ANALYSIS
# ============================================================
elif page == "📊 Data Analysis":
    st.title("📊 Data Analysis")
    st.markdown(
        "Exploration of the raw motor insurance portfolio. The dataset has been "
        "cleaned of negative claim amounts and missing values."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Policies", f"{n_policies:,}")
    c2.metric("Observed claims", f"{len(claim_severity):,}")
    c3.metric("Negatives removed", n_removed)
    c4.metric("Avg premium", f"{np.mean(premium):.2f}")

    with st.expander("👀 Preview raw data"):
        st.dataframe(df.head(20), use_container_width=True)

    st.divider()

    # Frequency distribution
    st.subheader("1. Frequency of Claims")
    cA, cB = st.columns([2, 1])
    with cA:
        freq_table = pd.Series(claim_count).value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=freq_table.index, y=freq_table.values,
            marker=dict(color="#4682B4"), name="Policies",
        ))
        fig.update_layout(
            title="Distribution of Claim Frequency",
            xaxis_title="Number of Claims", yaxis_title="Number of Policyholders",
            template="simple_white", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with cB:
        st.markdown("**Summary statistics**")
        m, v = float(np.mean(claim_count)), float(np.var(claim_count, ddof=1))
        st.metric("Mean", f"{m:.4f}")
        st.metric("Variance", f"{v:.4f}")
        st.metric("Var/Mean ratio", f"{v / m:.4f}",
                  delta="Overdispersion" if v / m > 1 else None)
        st.caption(
            "A variance-to-mean ratio above 1 indicates overdispersion, "
            "which motivates the use of a Negative Binomial model rather "
            "than a Poisson."
        )

    st.divider()

    # Severity distribution
    st.subheader("2. Severity of Claims")
    cA, cB = st.columns([2, 1])
    with cA:
        bins = st.slider("Histogram bins", 20, 80, 40, key="sev_bins")
        fig = go.Figure(go.Histogram(
            x=claim_severity, nbinsx=bins,
            marker=dict(color="#4682B4", line=dict(color="white", width=1)),
            name="Claims",
        ))
        fig.update_layout(
            title="Distribution of Claim Severity",
            xaxis_title="Claim Amount", yaxis_title="Count",
            template="simple_white", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with cB:
        st.markdown("**Summary statistics**")
        st.metric("Mean", f"{np.mean(claim_severity):.2f}")
        st.metric("Median", f"{np.median(claim_severity):.2f}")
        st.metric("Std deviation", f"{np.std(claim_severity, ddof=1):.2f}")
        st.metric("Max", f"{np.max(claim_severity):.2f}")
        st.caption(
            "The distribution is clearly right-skewed: most claims fall between "
            "50 and 250, with a long tail extending beyond 500."
        )

    st.divider()

    # Premium and Fees
    st.subheader("3. Premium and Fees")
    c1, c2, c3 = st.columns(3)
    c1.metric("Average premium", f"{np.mean(premium):.2f}")
    c2.metric("Average fees", f"{np.mean(fees):.4f}")
    c3.metric("Total premium", f"{np.sum(premium):,.0f}")

    # Demographic breakdown
    st.divider()
    st.subheader("4. Portfolio breakdown")
    if all(c in df.columns for c in ["GENDER", "AREA", "CAR_TYPE", "CAR_USE"]):
        c1, c2 = st.columns(2)
        with c1:
            for col in ["GENDER", "AREA"]:
                vc = df[col].value_counts()
                fig = px.pie(values=vc.values, names=vc.index, title=f"By {col}",
                             color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            for col in ["CAR_TYPE", "CAR_USE"]:
                vc = df[col].value_counts()
                fig = px.pie(values=vc.values, names=vc.index, title=f"By {col}",
                             color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: FREQUENCY MODEL
# ============================================================
elif page == "🎲 Frequency Model":
    st.title("🎲 Frequency Model Selection")
    st.markdown(
        "Two candidate distributions are tested for the number of claims per policy: "
        "**Poisson** (one parameter, mean = variance) and **Negative Binomial** "
        "(two parameters, allows for overdispersion)."
    )

    st.divider()
    st.subheader("Maximum-Likelihood Estimates")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Poisson**")
        st.markdown(f"- λ = `{fit_pois['lambda']:.4f}`")
        st.markdown(f"- Log-likelihood = `{fit_pois['loglik']:.2f}`")
        st.markdown(f"- AIC = `{fit_pois['aic']:.2f}`")
        st.markdown(f"- BIC = `{fit_pois['bic']:.2f}`")
    with c2:
        st.markdown("**Negative Binomial**")
        st.markdown(f"- size = `{fit_nb['size']:.4f}`")
        st.markdown(f"- μ = `{fit_nb['mu']:.4f}`")
        st.markdown(f"- Log-likelihood = `{fit_nb['loglik']:.2f}`")
        st.markdown(f"- AIC = `{fit_nb['aic']:.2f}`")
        st.markdown(f"- BIC = `{fit_nb['bic']:.2f}`")

    st.divider()
    st.subheader("Goodness-of-Fit (Chi-squared)")

    gof_p = chisq_gof_freq(claim_count, fit_pois, "pois")
    gof_n = chisq_gof_freq(claim_count, fit_nb, "nbinom")

    gof_df = pd.DataFrame({
        "Distribution": ["Poisson", "Negative Binomial"],
        "χ²": [gof_p["chi2"], gof_n["chi2"]],
        "df": [gof_p["df"], gof_n["df"]],
        "p-value": [gof_p["p_value"], gof_n["p_value"]],
        "AIC": [fit_pois["aic"], fit_nb["aic"]],
    })
    st.dataframe(
        gof_df.style.format({"χ²": "{:.4f}", "p-value": "{:.6f}", "AIC": "{:.2f}"})
        .highlight_min(axis=0, subset=["AIC"], color="#d4edda"),
        use_container_width=True,
    )

    if gof_n["p_value"] > 0.05 and gof_p["p_value"] < 0.05:
        st.success(
            f"✅ The Poisson model is **rejected** (p = {gof_p['p_value']:.6f}), "
            f"while the Negative Binomial **cannot be rejected** "
            f"(p = {gof_n['p_value']:.4f}). The Negative Binomial is selected."
        )

    st.divider()
    st.subheader("Observed vs Fitted PMF")

    freq_table = pd.Series(claim_count).value_counts().sort_index()
    k_seq = freq_table.index.values
    obs_prop = freq_table.values / freq_table.sum()
    p_pois = stats.poisson.pmf(k_seq, fit_pois["lambda"])
    p_nb = stats.nbinom.pmf(k_seq, fit_nb["size"], fit_nb["p"])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=k_seq, y=obs_prop, name="Observed",
                         marker=dict(color="#4682B4", opacity=0.6)))
    fig.add_trace(go.Scatter(x=k_seq, y=p_pois, name="Poisson",
                             mode="lines+markers",
                             line=dict(color="red", width=2),
                             marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=k_seq, y=p_nb, name="Negative Binomial",
                             mode="lines+markers",
                             line=dict(color="darkgreen", width=2, dash="dash"),
                             marker=dict(size=8)))
    fig.update_layout(
        title="Claim Frequency: Observed vs Fitted PMF",
        xaxis_title="Number of Claims", yaxis_title="Proportion",
        template="simple_white", height=450,
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: SEVERITY MODEL
# ============================================================
elif page == "💰 Severity Model":
    st.title("💰 Severity Model Selection")
    st.markdown(
        "Three candidate distributions are evaluated for individual claim amounts: "
        "**Gamma**, **Lognormal**, and **Weibull**. All are defined on the positive "
        "real line and can accommodate right-skewed data."
    )

    st.divider()
    st.subheader("AIC / BIC Comparison")

    aic_df = pd.DataFrame({
        "Distribution": list(sev_fits.keys()),
        "AIC": [sev_fits[k]["aic"] for k in sev_fits],
        "BIC": [sev_fits[k]["bic"] for k in sev_fits],
    })
    st.dataframe(
        aic_df.style.format({"AIC": "{:.2f}", "BIC": "{:.2f}"})
        .highlight_min(axis=0, subset=["AIC", "BIC"], color="#d4edda"),
        use_container_width=True,
    )

    ks = ks_test_severity(claim_severity, sev_fits)
    st.subheader("Kolmogorov–Smirnov Test")
    ks_df = pd.DataFrame({
        "Distribution": list(ks.keys()),
        "D": [ks[k].statistic for k in ks],
        "p-value": [ks[k].pvalue for k in ks],
    })
    st.dataframe(
        ks_df.style.format({"D": "{:.5f}", "p-value": "{:.6f}"})
        .highlight_max(axis=0, subset=["p-value"], color="#d4edda"),
        use_container_width=True,
    )

    st.success(
        f"✅ The **{best_sev_name}** distribution provides the best fit "
        f"(lowest AIC/BIC, highest KS p-value)."
    )

    st.divider()
    st.subheader("Observed vs Fitted Densities")

    x_seq = np.linspace(claim_severity.min(), claim_severity.max(), 500)
    d_g = stats.gamma.pdf(x_seq, sev_fits["Gamma"]["shape"], scale=sev_fits["Gamma"]["scale"])
    d_ln = stats.lognorm.pdf(x_seq, sev_fits["Lognormal"]["scipy_s"],
                             scale=sev_fits["Lognormal"]["scipy_scale"])
    d_w = stats.weibull_min.pdf(x_seq, sev_fits["Weibull"]["shape"],
                                scale=sev_fits["Weibull"]["scale"])

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=claim_severity, histnorm="probability density",
        nbinsx=40, marker=dict(color="#4682B4", opacity=0.5),
        name="Observed",
    ))
    fig.add_trace(go.Scatter(x=x_seq, y=d_g, name="Gamma", line=dict(color="red", width=2)))
    fig.add_trace(go.Scatter(x=x_seq, y=d_ln, name="Lognormal",
                             line=dict(color="darkgreen", width=2)))
    fig.add_trace(go.Scatter(x=x_seq, y=d_w, name="Weibull",
                             line=dict(color="orange", width=2)))
    fig.update_layout(
        title="Claim Severity: Observed vs Fitted Densities",
        xaxis_title="Claim Amount", yaxis_title="Density",
        template="simple_white", height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("QQ Plots")

    n_obs = len(claim_severity)
    probs = (np.arange(1, n_obs + 1) - 0.5) / n_obs
    obs_q = np.quantile(claim_severity, probs)

    qq_data = {
        "Gamma": stats.gamma.ppf(probs, sev_fits["Gamma"]["shape"],
                                 scale=sev_fits["Gamma"]["scale"]),
        "Lognormal": stats.lognorm.ppf(probs, sev_fits["Lognormal"]["scipy_s"],
                                       scale=sev_fits["Lognormal"]["scipy_scale"]),
        "Weibull": stats.weibull_min.ppf(probs, sev_fits["Weibull"]["shape"],
                                         scale=sev_fits["Weibull"]["scale"]),
    }

    fig = make_subplots(rows=1, cols=3, subplot_titles=list(qq_data.keys()))
    for i, (name, theo) in enumerate(qq_data.items(), 1):
        fig.add_trace(
            go.Scatter(x=theo, y=obs_q, mode="markers", showlegend=False,
                       marker=dict(color="#4682B4", size=4, opacity=0.5)),
            row=1, col=i,
        )
        ref = np.linspace(min(theo.min(), obs_q.min()), max(theo.max(), obs_q.max()), 50)
        fig.add_trace(
            go.Scatter(x=ref, y=ref, mode="lines", showlegend=False,
                       line=dict(color="red", width=1.5)),
            row=1, col=i,
        )
    fig.update_xaxes(title_text="Theoretical")
    fig.update_yaxes(title_text="Observed")
    fig.update_layout(height=400, template="simple_white",
                      title="QQ Plots – Claim Severity")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: MONTE CARLO & VARIANCE REDUCTION
# ============================================================
elif page == "🔄 Monte Carlo & Variance Reduction":
    st.title("🔄 Monte Carlo Estimation & Variance Reduction")

    st.markdown(
        "Monte Carlo simulation is used to estimate the expected claim frequency, "
        "the expected claim severity, and ultimately the aggregate loss distribution. "
        "Variance reduction techniques (Antithetic Variates and Control Variate) are "
        "tested to improve estimator efficiency."
    )

    n_mc = st.slider("Number of Monte Carlo simulations (N)",
                     1000, 200000, 100000, step=1000)

    rng = np.random.default_rng(42)

    st.divider()
    st.subheader("Convergence Plots")

    # Frequency
    if best_freq_name == "Poisson":
        sim_freq = rng.poisson(fit_pois["lambda"], size=n_mc)
        true_mean_freq = fit_pois["lambda"]
    else:
        sim_freq = stats.nbinom.rvs(fit_nb["size"], fit_nb["p"], size=n_mc, random_state=rng)
        true_mean_freq = fit_nb["mu"]
    running_freq = np.cumsum(sim_freq) / np.arange(1, n_mc + 1)

    # Severity
    sim_sev = rng.lognormal(sev_fits["Lognormal"]["meanlog"],
                            sev_fits["Lognormal"]["sdlog"], size=n_mc)
    true_mean_sev = float(np.exp(sev_fits["Lognormal"]["meanlog"]
                                 + sev_fits["Lognormal"]["sdlog"] ** 2 / 2))
    running_sev = np.cumsum(sim_sev) / np.arange(1, n_mc + 1)

    c1, c2 = st.columns(2)
    with c1:
        # Plot at downsampled rate to keep things fast
        step = max(1, n_mc // 2000)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(1, n_mc + 1)[::step], y=running_freq[::step],
            mode="lines", line=dict(color="#4682B4"), name="Running mean",
        ))
        fig.add_hline(y=float(np.mean(claim_count)), line=dict(color="red", dash="dash"),
                      annotation_text="Empirical mean", annotation_position="top right")
        fig.add_hline(y=true_mean_freq, line=dict(color="darkgreen", dash="dot"),
                      annotation_text="Theoretical mean", annotation_position="bottom right")
        fig.update_layout(title="MC Convergence – Claim Frequency",
                          xaxis_title="Number of Simulations",
                          yaxis_title="Running Mean",
                          template="simple_white", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        step = max(1, n_mc // 2000)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(1, n_mc + 1)[::step], y=running_sev[::step],
            mode="lines", line=dict(color="#4682B4"), name="Running mean",
        ))
        fig.add_hline(y=float(np.mean(claim_severity)), line=dict(color="red", dash="dash"),
                      annotation_text="Empirical mean", annotation_position="top right")
        fig.add_hline(y=true_mean_sev, line=dict(color="darkgreen", dash="dot"),
                      annotation_text="Theoretical mean", annotation_position="bottom right")
        fig.update_layout(title="MC Convergence – Claim Severity",
                          xaxis_title="Number of Simulations",
                          yaxis_title="Running Mean (Claim Amount)",
                          template="simple_white", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Statistical Properties of Monte Carlo Estimators")

    # Frequency variance
    theo_var_freq = (true_mean_freq + true_mean_freq ** 2 / fit_nb["size"]) / n_mc \
        if best_freq_name == "Negative Binomial" else true_mean_freq / n_mc
    emp_var_freq = float(np.var(sim_freq, ddof=1) / n_mc)
    mse_freq = float(np.mean((sim_freq - true_mean_freq) ** 2) / n_mc \
                     + (np.mean(sim_freq) - np.mean(claim_count)) ** 2)

    # Severity variance (Lognormal)
    ml = sev_fits["Lognormal"]["meanlog"]
    sl = sev_fits["Lognormal"]["sdlog"]
    theo_var_sev = (np.exp(sl ** 2) - 1) * np.exp(2 * ml + sl ** 2) / n_mc
    emp_var_sev = float(np.var(sim_sev, ddof=1) / n_mc)
    mse_sev = float(np.mean((sim_sev - true_mean_sev) ** 2) / n_mc \
                    + (np.mean(sim_sev) - np.mean(claim_severity)) ** 2)

    props_df = pd.DataFrame({
        "Frequency": [theo_var_freq, emp_var_freq, mse_freq],
        "Severity": [theo_var_sev, emp_var_sev, mse_sev],
    }, index=["Theoretical variance", "Empirical variance", "MSE"])
    st.dataframe(props_df.style.format("{:.8f}"), use_container_width=True)

    st.divider()
    st.subheader("Variance Reduction: Antithetic Variates")
    st.markdown(
        "For each uniform draw $U$, a paired draw $(U, 1 - U)$ is generated. "
        "Their average exploits negative correlation to reduce variance."
    )

    n_av = n_mc // 2
    U = rng.uniform(size=n_av)

    # Frequency
    if best_freq_name == "Poisson":
        x1_f = stats.poisson.ppf(U, fit_pois["lambda"])
        x2_f = stats.poisson.ppf(1 - U, fit_pois["lambda"])
    else:
        x1_f = stats.nbinom.ppf(U, fit_nb["size"], fit_nb["p"])
        x2_f = stats.nbinom.ppf(1 - U, fit_nb["size"], fit_nb["p"])
    av_freq = (x1_f + x2_f) / 2
    var_av_freq = float(np.var(av_freq, ddof=1) / n_av)
    var_std_freq = float(np.var(sim_freq, ddof=1) / n_mc)

    # Severity
    U2 = rng.uniform(size=n_av)
    x1_s = stats.lognorm.ppf(U2, sev_fits["Lognormal"]["scipy_s"],
                             scale=sev_fits["Lognormal"]["scipy_scale"])
    x2_s = stats.lognorm.ppf(1 - U2, sev_fits["Lognormal"]["scipy_s"],
                             scale=sev_fits["Lognormal"]["scipy_scale"])
    av_sev = (x1_s + x2_s) / 2
    var_av_sev = float(np.var(av_sev, ddof=1) / n_av)
    var_std_sev = float(np.var(sim_sev, ddof=1) / n_mc)

    red_freq_av = 100 * (1 - var_av_freq / var_std_freq)
    red_sev_av = 100 * (1 - var_av_sev / var_std_sev)

    c1, c2 = st.columns(2)
    c1.metric("Variance reduction – Frequency", f"{red_freq_av:.2f}%")
    c2.metric("Variance reduction – Severity", f"{red_sev_av:.2f}%")

    st.subheader("Variance Reduction: Control Variate (Exponential)")
    st.markdown(
        "Uses an Exponential variable with known mean as a control. "
        "Effectiveness depends on correlation with the target."
    )

    rng2 = np.random.default_rng(123)
    sim_freq_cv = (rng2.poisson(fit_pois["lambda"], size=n_mc)
                   if best_freq_name == "Poisson"
                   else stats.nbinom.rvs(fit_nb["size"], fit_nb["p"], size=n_mc, random_state=rng2))
    C_freq = rng2.exponential(scale=true_mean_freq, size=n_mc)
    cov_xc_f = np.cov(sim_freq_cv, C_freq, ddof=1)[0, 1]
    var_c_f = float(np.var(C_freq, ddof=1))
    c_star_f = -cov_xc_f / var_c_f
    X_cv_f = sim_freq_cv + c_star_f * (C_freq - true_mean_freq)
    var_cv_freq = float(np.var(X_cv_f, ddof=1) / n_mc)
    corr_freq = float(np.corrcoef(sim_freq_cv, C_freq)[0, 1])

    rng3 = np.random.default_rng(456)
    sim_sev_cv = rng3.lognormal(ml, sl, size=n_mc)
    C_sev = rng3.exponential(scale=true_mean_sev, size=n_mc)
    cov_xc_s = np.cov(sim_sev_cv, C_sev, ddof=1)[0, 1]
    var_c_s = float(np.var(C_sev, ddof=1))
    c_star_s = -cov_xc_s / var_c_s
    X_cv_s = sim_sev_cv + c_star_s * (C_sev - true_mean_sev)
    var_cv_sev = float(np.var(X_cv_s, ddof=1) / n_mc)
    corr_sev = float(np.corrcoef(sim_sev_cv, C_sev)[0, 1])

    red_freq_cv = 100 * (1 - var_cv_freq / var_std_freq)
    red_sev_cv = 100 * (1 - var_cv_sev / var_std_sev)

    c1, c2 = st.columns(2)
    c1.metric("Variance reduction – Frequency", f"{red_freq_cv:.2f}%",
              delta=f"corr = {corr_freq:.4f}")
    c2.metric("Variance reduction – Severity", f"{red_sev_cv:.2f}%",
              delta=f"corr = {corr_sev:.4f}")

    st.divider()
    st.subheader("Comparison")

    plot_df = pd.DataFrame({
        "Method": ["Standard MC", "Antithetic", "Control Variate"] * 2,
        "Variable": ["Frequency"] * 3 + ["Severity"] * 3,
        "Reduction (%)": [0, red_freq_av, red_freq_cv, 0, red_sev_av, red_sev_cv],
    })
    fig = px.bar(plot_df, x="Method", y="Reduction (%)", color="Method",
                 facet_col="Variable", text="Reduction (%)",
                 color_discrete_map={"Standard MC": "#888", "Antithetic": "#4682B4",
                                     "Control Variate": "#e67e22"})
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(template="simple_white", height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "💡 **Conclusion:** Antithetic variates achieve a strong variance reduction "
        "(>75%) thanks to the negative correlation induced by construction. "
        "The control variate method, however, performs poorly because the "
        "correlation between the target and the exponential control is near zero. "
        "**Antithetic variates are the preferred technique for this portfolio.**"
    )


# ============================================================
# PAGE: RISK PREMIUM & VAR
# ============================================================
elif page == "📈 Risk Premium & VaR":
    st.title("📈 Risk Premium, VaR and Solvency Capital")

    st.markdown(
        "The aggregate claim amount $S$ for a single policy is the sum of $N$ individual "
        "claim amounts, where $N$ follows the fitted Negative Binomial distribution and "
        "each claim follows the fitted Lognormal. Since $S$ has no closed form, Monte Carlo "
        "simulation is used."
    )

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        n_sim = st.slider("Number of MC simulations",
                          5000, 200000, N_SIM_DEFAULT, step=5000)
    with c2:
        alpha = st.slider("VaR confidence level", 0.90, 0.999, 0.995, step=0.005)
    with c3:
        bins_loss = st.slider("Histogram bins", 30, 100, 60)

    agg_loss, sim_counts = simulate_aggregate_loss(
        n_sim, best_freq_name, best_freq_params, best_sev_name, best_sev_params, seed=42
    )

    risk_premium_mc = float(agg_loss.mean())
    avg_premium = float(np.mean(premium))
    avg_fees = float(np.mean(fees))
    risk_premium_emp = float(claim_severity.sum() / n_policies)
    true_premium = risk_premium_mc + avg_fees
    margin = avg_premium - risk_premium_mc - avg_fees
    margin_pct = 100 * margin / avg_premium

    VaR = float(np.quantile(agg_loss, alpha))
    TVaR = float(agg_loss[agg_loss >= VaR].mean())
    SCR = VaR - risk_premium_mc

    st.divider()
    st.subheader("Risk Premium")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Empirical RP (data)", f"{risk_premium_emp:.2f}")
    c2.metric("MC simulated RP", f"{risk_premium_mc:.2f}",
              delta=f"{risk_premium_mc - risk_premium_emp:+.2f}")
    c3.metric("Average premium", f"{avg_premium:.2f}")
    c4.metric("Profit margin", f"{margin_pct:.2f}%",
              delta="Adequate" if margin > 0 else "Insufficient")

    st.divider()
    st.subheader("Aggregate Loss Distribution")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=agg_loss, nbinsx=bins_loss,
        marker=dict(color="#4682B4", opacity=0.75, line=dict(color="white", width=1)),
        name="Simulated S",
    ))
    fig.add_vline(x=risk_premium_mc, line=dict(color="darkgreen", width=2, dash="dash"),
                  annotation_text=f"Mean (RP) = {risk_premium_mc:.0f}",
                  annotation_position="top")
    fig.add_vline(x=VaR, line=dict(color="red", width=2, dash="dash"),
                  annotation_text=f"VaR {alpha * 100:.1f}% = {VaR:.0f}",
                  annotation_position="top")
    fig.add_vline(x=TVaR, line=dict(color="orange", width=2, dash="dashdot"),
                  annotation_text=f"TVaR {alpha * 100:.1f}% = {TVaR:.0f}",
                  annotation_position="top")
    fig.update_layout(
        title="Simulated Aggregate Loss Distribution",
        xaxis_title="Aggregate Loss S", yaxis_title="Count",
        template="simple_white", height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader(f"Solvency Capital Requirement (Solvency II, α = {alpha:.3f})")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"VaR {alpha * 100:.1f}%", f"{VaR:.2f}")
    c2.metric(f"TVaR {alpha * 100:.1f}%", f"{TVaR:.2f}")
    c3.metric("SCR (VaR − RP)", f"{SCR:.2f}")

    ratio = TVaR / risk_premium_mc
    st.info(
        f"📊 In the worst {(1 - alpha) * 100:.1f}% of scenarios, the average aggregate "
        f"loss reaches **{TVaR:.0f}**, approximately **{ratio:.1f}× the expected loss**. "
        "The TVaR being substantially higher than the VaR confirms a heavy right tail."
    )


# ============================================================
# PAGE: REINSURANCE
# ============================================================
elif page == "🛡️ Reinsurance":
    st.title("🛡️ Reinsurance Strategy")
    st.markdown(
        "Although the portfolio is profitable, an Excess of Loss (XoL) reinsurance "
        "treaty can protect against extreme losses. Use the slider to choose a "
        "retention percentile and observe the impact on ceded loss and SCR."
    )

    n_sim = 50000
    agg_loss, _ = simulate_aggregate_loss(
        n_sim, best_freq_name, best_freq_params, best_sev_name, best_sev_params, seed=42
    )

    risk_premium_mc = float(agg_loss.mean())

    c1, c2 = st.columns([1, 1])
    with c1:
        retention_pct = st.slider("Retention percentile",
                                  0.80, 0.999, 0.95, step=0.005)
    with c2:
        st.write("")
        st.write("")

    retention_limit = float(np.quantile(agg_loss, retention_pct))
    ceded = np.maximum(agg_loss - retention_limit, 0)
    expected_ceded = float(ceded.mean())
    pct_ceded = 100 * expected_ceded / risk_premium_mc
    pct_scenarios = 100 * float(np.mean(agg_loss > retention_limit))

    # Net losses after reinsurance
    net_loss = np.minimum(agg_loss, retention_limit)
    new_VaR995 = float(np.quantile(net_loss, 0.995))
    old_VaR995 = float(np.quantile(agg_loss, 0.995))
    new_SCR = new_VaR995 - risk_premium_mc
    old_SCR = old_VaR995 - risk_premium_mc

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retention limit", f"{retention_limit:.2f}",
              delta=f"P{retention_pct * 100:.1f}")
    c2.metric("Expected loss ceded", f"{expected_ceded:.2f}",
              delta=f"{pct_ceded:.2f}% of total risk")
    c3.metric("Scenarios triggered", f"{pct_scenarios:.2f}%")
    c4.metric("SCR reduction", f"{old_SCR - new_SCR:.0f}",
              delta=f"−{(old_SCR - new_SCR) / old_SCR * 100:.1f}%")

    st.divider()

    # Two charts: original vs net
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Before Reinsurance", "After Reinsurance (Net)"))
    fig.add_trace(go.Histogram(
        x=agg_loss, nbinsx=60, marker=dict(color="#4682B4", opacity=0.75),
        showlegend=False,
    ), row=1, col=1)
    fig.add_vline(x=retention_limit, line=dict(color="red", dash="dash"),
                  annotation_text=f"Retention = {retention_limit:.0f}",
                  row=1, col=1)
    fig.add_vline(x=old_VaR995, line=dict(color="orange", dash="dot"),
                  annotation_text=f"VaR 99.5% = {old_VaR995:.0f}",
                  row=1, col=1)

    fig.add_trace(go.Histogram(
        x=net_loss, nbinsx=60, marker=dict(color="#27ae60", opacity=0.75),
        showlegend=False,
    ), row=1, col=2)
    fig.add_vline(x=new_VaR995, line=dict(color="orange", dash="dot"),
                  annotation_text=f"VaR 99.5% = {new_VaR995:.0f}",
                  row=1, col=2)

    fig.update_xaxes(title_text="Aggregate Loss S")
    fig.update_yaxes(title_text="Count")
    fig.update_layout(template="simple_white", height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.success(
        f"✅ With a retention at the {retention_pct * 100:.1f}th percentile "
        f"({retention_limit:.0f}), only **{pct_scenarios:.2f}%** of scenarios "
        f"trigger reinsurance. The expected loss ceded is **{expected_ceded:.2f}** "
        f"per policy ({pct_ceded:.2f}% of total expected risk). "
        f"The SCR drops from **{old_SCR:.0f}** to **{new_SCR:.0f}**."
    )


# ============================================================
# PAGE: SENSITIVITY ANALYSIS
# ============================================================
elif page == "🔬 Sensitivity Analysis":
    st.title("🔬 Sensitivity Analysis")
    st.markdown(
        "We test the robustness of our results by simulating the aggregate loss "
        "under four alternative model combinations."
    )

    n_sens = st.slider("Simulations per scenario", 5000, 100000, 30000, step=5000)

    @st.cache_data
    def run_sens(n_sens, fit_pois, fit_nb, sev_fits):
        scenarios = [
            ("NB", "Lognormal"),
            ("NB", "Gamma"),
            ("Poisson", "Lognormal"),
            ("Poisson", "Gamma"),
        ]
        rows = []
        for i, (fname, sname) in enumerate(scenarios):
            freq_model = "Negative Binomial" if fname == "NB" else "Poisson"
            freq_params = fit_nb if fname == "NB" else fit_pois
            sev_params = sev_fits[sname]
            agg, _ = simulate_aggregate_loss(
                n_sens, freq_model, freq_params, sname, sev_params, seed=42 + i
            )
            rp = float(agg.mean())
            var = float(np.quantile(agg, 0.995))
            scr = var - rp
            rows.append({
                "Frequency": fname, "Severity": sname,
                "Risk Premium": rp, "VaR 99.5%": var, "SCR": scr,
            })
        return pd.DataFrame(rows)

    sens_df = run_sens(n_sens, fit_pois, fit_nb, sev_fits)

    st.dataframe(
        sens_df.style.format({
            "Risk Premium": "{:.2f}", "VaR 99.5%": "{:.2f}", "SCR": "{:.2f}"
        }), use_container_width=True,
    )

    st.divider()

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Risk Premium", "VaR 99.5%", "SCR"))
    labels = sens_df["Frequency"] + " + " + sens_df["Severity"]
    fig.add_trace(go.Bar(x=labels, y=sens_df["Risk Premium"], showlegend=False,
                         marker_color="#4682B4"), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=sens_df["VaR 99.5%"], showlegend=False,
                         marker_color="#e67e22"), row=1, col=2)
    fig.add_trace(go.Bar(x=labels, y=sens_df["SCR"], showlegend=False,
                         marker_color="#c0392b"), row=1, col=3)
    fig.update_layout(template="simple_white", height=400)
    fig.update_xaxes(tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "💡 The choice of **frequency model** has a much larger impact on tail-risk "
        "measures than the choice of severity model. The Poisson model underestimates "
        "overdispersion and thus produces a lower (more optimistic) SCR than the "
        "Negative Binomial. The risk premium itself is robust across all combinations."
    )


# ============================================================
# PAGE: CONCLUSION
# ============================================================
elif page == "📝 Conclusion & Recommendations":
    st.title("📝 Conclusion & Recommendations")

    avg_premium = float(np.mean(premium))
    avg_fees = float(np.mean(fees))
    total_claims_data = float(claim_severity.sum())
    combined_ratio = (total_claims_data + np.sum(fees)) / np.sum(premium)
    margin_pct = (1 - combined_ratio) * 100

    # Tariff what-if
    st.subheader("💼 Tariff what-if simulator")
    reduction_pct = st.slider("Premium reduction (%)", 0.0, 20.0, 0.0, step=0.5)

    new_premium = avg_premium * (1 - reduction_pct / 100)
    new_total_premium = np.sum(premium) * (1 - reduction_pct / 100)
    new_combined_ratio = (total_claims_data + np.sum(fees)) / new_total_premium

    c1, c2, c3 = st.columns(3)
    c1.metric("New average premium", f"{new_premium:.2f}",
              delta=f"−{reduction_pct:.1f}%")
    c2.metric("New combined ratio", f"{new_combined_ratio:.3f}",
              delta="Profitable" if new_combined_ratio < 1 else "Loss-making",
              delta_color="normal" if new_combined_ratio < 1 else "inverse")
    c3.metric("New profit margin",
              f"{(1 - new_combined_ratio) * 100:.2f}%")

    if new_combined_ratio >= 1:
        st.error(
            f"🚨 **Warning:** A {reduction_pct:.1f}% reduction pushes the combined "
            f"ratio above 1.0. The portfolio becomes unprofitable."
        )
    elif reduction_pct > margin_pct - 2:
        st.warning(
            f"⚠️ A {reduction_pct:.1f}% reduction leaves only "
            f"{(1 - new_combined_ratio) * 100:.2f}% of margin — risky given tail exposure."
        )
    else:
        st.success(
            f"✅ A {reduction_pct:.1f}% reduction keeps the portfolio profitable "
            f"(margin = {(1 - new_combined_ratio) * 100:.2f}%)."
        )

    st.divider()

    st.subheader("🎯 Final Recommendations")
    st.markdown(
        f"""
1. **Tariff strategy: maintain the current premium.**
   The portfolio is currently profitable with a combined ratio of
   **{combined_ratio:.3f}** and a margin of **{margin_pct:.2f}%**. Any reduction
   above ~{margin_pct:.0f}% would push the company into a loss-making position.
   Given the heavy-tailed loss distribution (TVaR 99.5% ≈ 4× the expected loss),
   we advise against reducing the tariff.

2. **Reinsurance: implement an Excess-of-Loss (XoL) treaty.**
   Recommended attachment point: ~1,073 (95th percentile of the simulated
   aggregate loss). This transfers the most severe tail events to the reinsurer
   while only ceding ~2.83% of total expected risk. The Solvency Capital
   Requirement is meaningfully reduced and financial stability is enhanced.

3. **Model risk awareness.**
   - The Negative Binomial + Lognormal combination is the most conservative for
     SCR estimation and is the prudent choice.
   - A strong positive correlation (Spearman ρ ≈ 0.88) was observed between
     frequency and severity, violating the independence assumption — future
     work should consider a copula-based approach.
   - The analysis is based on a single year of data; inflation, seasonality,
     and macroeconomic effects are not yet modelled.
        """
    )

    st.divider()
    st.caption(
        "Mazy Djezzar · Prescilya Fabi · Samantha López — "
        "HEC University of Lausanne, Master's in Actuarial Science"
    )
