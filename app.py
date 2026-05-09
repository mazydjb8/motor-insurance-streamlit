"""
Motor Insurance Portfolio Analysis — Interactive Dashboard

Companion application to the actuarial study of a motor insurance portfolio
of 1,002 policyholders. Replicates the analysis carried out in R using Python,
and adds interactive controls for Monte Carlo simulation, solvency capital
estimation, and reinsurance design.

HEC University of Lausanne — Master's in Actuarial Science
Course: Simulation Methods in Finance and Insurance
Authors: Mazy Djezzar, Prescilya Fabi, Samantha López
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
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Motor Insurance Portfolio Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }
    h1, h2, h3, h4 { color: #1f3a5f; font-family: Georgia, serif; }
    h1 { border-bottom: 2px solid #1f3a5f; padding-bottom: 0.4rem; }
    .stMarkdown p { line-height: 1.6; }
    div[data-testid="stMetricValue"] { color: #1f3a5f; }
    section[data-testid="stSidebar"] h1 { font-size: 1.2rem; }
    section[data-testid="stSidebar"] h2 { font-size: 1.0rem; margin-top: 1rem; }
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
    df = pd.read_csv(url, na_values=[" -   ", "-", "NA", "", " "])
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data(show_spinner="Loading data...")
def load_data_from_file(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.read()
    for sep in [";", ","]:
        try:
            df = pd.read_csv(
                io.BytesIO(content), sep=sep, encoding="utf-8-sig",
                na_values=[" -   ", "-", "NA", "", " "],
            )
            if df.shape[1] > 5:
                df.columns = [c.strip() for c in df.columns]
                return df
        except Exception:
            continue
    raise ValueError("Could not parse the CSV file.")


def clean_data(df: pd.DataFrame):
    claim_count = df["CLM_FREQ"].astype(int).values
    amt_cols = [c for c in df.columns if c.startswith("CLM_AMT")]
    for c in amt_cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce")
    sev_raw = df[amt_cols].values.flatten()
    sev_raw = sev_raw[~np.isnan(sev_raw)]
    claim_severity = sev_raw[sev_raw > 0]
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
    n = len(claim_severity)
    out = {}

    a, _, scale_g = stats.gamma.fit(claim_severity, floc=0)
    ll_g = stats.gamma.logpdf(claim_severity, a, loc=0, scale=scale_g).sum()
    out["Gamma"] = {
        "shape": a, "rate": 1 / scale_g, "scale": scale_g,
        "loglik": ll_g, "aic": 2 * 2 - 2 * ll_g, "bic": np.log(n) * 2 - 2 * ll_g,
    }

    sigma, _, scale_ln = stats.lognorm.fit(claim_severity, floc=0)
    ll_ln = stats.lognorm.logpdf(claim_severity, sigma, loc=0, scale=scale_ln).sum()
    out["Lognormal"] = {
        "meanlog": np.log(scale_ln), "sdlog": sigma,
        "scipy_s": sigma, "scipy_scale": scale_ln,
        "loglik": ll_ln, "aic": 2 * 2 - 2 * ll_ln, "bic": np.log(n) * 2 - 2 * ll_ln,
    }

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
    rng = np.random.default_rng(seed)

    if freq_model == "Poisson":
        sim_counts = rng.poisson(freq_params["lambda"], size=n_sim)
    else:
        sim_counts = stats.nbinom.rvs(
            freq_params["size"], freq_params["p"], size=n_sim, random_state=rng,
        )

    agg_loss = np.zeros(n_sim)
    total_claims = int(sim_counts.sum())
    if total_claims == 0:
        return agg_loss, sim_counts

    if sev_model == "Gamma":
        all_sev = rng.gamma(sev_params["shape"], scale=sev_params["scale"], size=total_claims)
    elif sev_model == "Lognormal":
        all_sev = rng.lognormal(sev_params["meanlog"], sev_params["sdlog"], size=total_claims)
    else:
        all_sev = sev_params["scale"] * rng.weibull(sev_params["shape"], size=total_claims)

    idx = 0
    for i, nc in enumerate(sim_counts):
        if nc > 0:
            agg_loss[i] = all_sev[idx:idx + nc].sum()
            idx += nc

    return agg_loss, sim_counts


# ============================================================
# SIDEBAR — DATA SOURCE & NAVIGATION
# ============================================================
st.sidebar.title("Motor Insurance Portfolio")
st.sidebar.caption("Actuarial Analysis Dashboard")

st.sidebar.markdown("---")
st.sidebar.subheader("Data source")
st.sidebar.caption(
    "Data is loaded from the project's Google Sheets by default. "
    "An alternative CSV file can be uploaded below."
)

uploaded = st.sidebar.file_uploader(
    "Upload an alternative CSV (optional)",
    type=["csv"],
    help="If no file is uploaded, the project's Google Sheets is used.",
)

df = None
load_error = None
data_source_label = None

try:
    if uploaded is not None:
        df = load_data_from_file(uploaded)
        data_source_label = f"Uploaded file: {uploaded.name}"
    else:
        df = load_data_from_url(GSHEET_CSV_URL)
        data_source_label = "Project Google Sheets"
except Exception as e:
    load_error = str(e)

if df is None:
    if load_error:
        st.error(
            f"Failed to load data from Google Sheets: {load_error}\n\n"
            "If the issue persists, the spreadsheet may not be publicly shared. "
            "You can upload a CSV file from the sidebar as an alternative."
        )
    else:
        st.info("Loading data...")
    st.stop()

st.sidebar.success(f"Loaded from: {data_source_label}")

claim_count, claim_severity, n_removed, amt_cols = clean_data(df.copy())
n_policies = len(df)
premium = df["PREMIUM"].values
fees = df["FEES"].values

fit_pois = fit_poisson(claim_count)
fit_nb = fit_negbinom(claim_count)
sev_fits = fit_severity_models(claim_severity)

best_freq_name = "Negative Binomial" if fit_nb["aic"] < fit_pois["aic"] else "Poisson"
best_freq_params = fit_nb if best_freq_name == "Negative Binomial" else fit_pois

best_sev_name = min(sev_fits, key=lambda k: sev_fits[k]["aic"])
best_sev_params = sev_fits[best_sev_name]

st.sidebar.markdown("---")
st.sidebar.subheader("Sections")
page = st.sidebar.radio(
    "Section selection",
    [
        "Executive summary",
        "Data analysis",
        "Frequency model",
        "Severity model",
        "Monte Carlo and variance reduction",
        "Risk premium and Value at Risk",
        "Reinsurance",
        "Sensitivity analysis",
        "Conclusion and recommendations",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Selected frequency model:** {best_freq_name}  \n"
    f"**Selected severity model:** {best_sev_name}"
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Mazy Djezzar — Prescilya Fabi — Samantha López  \n"
    "HEC University of Lausanne  \n"
    "Master's in Actuarial Science"
)


# ============================================================
# COMMON HELPERS
# ============================================================
N_SIM_DEFAULT = 50000

PLOTLY_TEMPLATE = "simple_white"
COLOR_PRIMARY = "#1f3a5f"
COLOR_SECONDARY = "#4682B4"
COLOR_ACCENT = "#8B0000"
COLOR_NEUTRAL = "#666666"


# ============================================================
# PAGE: EXECUTIVE SUMMARY
# ============================================================
if page == "Executive summary":
    st.title("Motor Insurance Portfolio Analysis")
    st.markdown(
        "**HEC University of Lausanne** — Simulation Methods in Finance and Insurance  \n"
        "Master's in Actuarial Science  \n"
        "Authors: Mazy Djezzar, Prescilya Fabi, Samantha López"
    )
    st.markdown("---")

    avg_premium = float(np.mean(premium))
    avg_fees = float(np.mean(fees))
    total_claims_data = float(np.sum(claim_severity))
    combined_ratio = (total_claims_data + np.sum(fees)) / np.sum(premium)
    risk_premium_emp = total_claims_data / n_policies

    # Monte Carlo risk premium (used to compute the MC-based profit margin,
    # consistent with the methodology of the written report)
    agg_loss_es, _ = simulate_aggregate_loss(
        N_SIM_DEFAULT, best_freq_name, best_freq_params,
        best_sev_name, best_sev_params, seed=42,
    )
    risk_premium_mc = float(agg_loss_es.mean())
    margin_mc = avg_premium - risk_premium_mc - avg_fees
    margin_mc_pct = 100 * margin_mc / avg_premium

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Policyholders", f"{n_policies:,}")
    c2.metric("Observed claims", f"{len(claim_severity):,}")
    c3.metric("Combined ratio", f"{combined_ratio:.3f}")
    c4.metric("Profit margin", f"{margin_mc_pct:.2f}%")

    st.markdown("---")
    st.header("Executive summary")
    st.markdown(
        f"""
The analysis focuses on a motor insurance portfolio of **{n_policies:,} policyholders**
to assess whether the existing pricing approach remains viable and whether
lowering premiums could be justified.

The portfolio is currently profitable: a combined ratio of **{combined_ratio:.2f}**
indicates that premium income comfortably covers both claims and operating costs,
leaving a margin of **{margin_mc_pct:.2f}%**. However, this margin
leaves limited flexibility. Any premium reduction exceeding
**{margin_mc_pct:.2f}%** would put the company in an unprofitable
position.

The risk assessment indicates that, under severe scenarios, aggregate losses
could reach nearly four times the average expected loss. Although the company
remains profitable under normal conditions, such events represent a significant
financial threat that cannot be ignored.

Two recommendations follow from these findings. First, the current tariff should
be maintained to preserve profitability. Second, a reinsurance arrangement
should be put in place to protect the company against exceptionally large
losses, ensuring its long-term financial stability.
        """
    )

    st.markdown("---")
    st.header("Structure of the dashboard")
    st.markdown(
        """
The dashboard mirrors the structure of the written report:

- **Data analysis** — exploration of the raw portfolio data
- **Frequency model** — selection between Poisson and Negative Binomial
- **Severity model** — selection between Gamma, Lognormal, and Weibull
- **Monte Carlo and variance reduction** — convergence assessment and
  variance reduction techniques (antithetic and control variates)
- **Risk premium and Value at Risk** — aggregate loss distribution and
  Solvency Capital Requirement under Solvency II
- **Reinsurance** — Excess-of-Loss treaty design with retention slider
- **Sensitivity analysis** — robustness across model combinations
- **Conclusion and recommendations** — tariff what-if simulator
        """
    )


# ============================================================
# PAGE: DATA ANALYSIS
# ============================================================
elif page == "Data analysis":
    st.title("Data analysis")
    st.markdown(
        "The dataset has been reviewed and cleaned: negative claim amounts and "
        "missing values have been removed before analysis."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Policies", f"{n_policies:,}")
    c2.metric("Observed claims", f"{len(claim_severity):,}")
    c3.metric("Negative entries removed", n_removed)
    c4.metric("Average premium", f"{np.mean(premium):.2f}")

    with st.expander("Preview of the raw data"):
        st.dataframe(df.head(20), use_container_width=True)

    st.markdown("---")

    # Frequency
    st.header("1. Frequency of claims")
    cA, cB = st.columns([2, 1])
    with cA:
        freq_table = pd.Series(claim_count).value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=freq_table.index, y=freq_table.values,
            marker=dict(color=COLOR_SECONDARY),
        ))
        fig.update_layout(
            title="Distribution of claim frequency",
            xaxis_title="Number of claims",
            yaxis_title="Number of policyholders",
            template=PLOTLY_TEMPLATE, height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with cB:
        st.subheader("Summary statistics")
        m, v = float(np.mean(claim_count)), float(np.var(claim_count, ddof=1))
        st.metric("Mean", f"{m:.4f}")
        st.metric("Variance", f"{v:.4f}")
        st.metric("Variance / mean ratio", f"{v / m:.4f}")
        st.markdown(
            "A variance-to-mean ratio above one indicates overdispersion, "
            "which motivates the use of a Negative Binomial model rather "
            "than a Poisson."
        )

    st.markdown("---")

    # Severity
    st.header("2. Severity of claims")
    cA, cB = st.columns([2, 1])
    with cA:
        bins = st.slider("Histogram bins", 20, 80, 40, key="sev_bins")
        fig = go.Figure(go.Histogram(
            x=claim_severity, nbinsx=bins,
            marker=dict(color=COLOR_SECONDARY, line=dict(color="white", width=1)),
        ))
        fig.update_layout(
            title="Distribution of claim severity",
            xaxis_title="Claim amount", yaxis_title="Count",
            template=PLOTLY_TEMPLATE, height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with cB:
        st.subheader("Summary statistics")
        st.metric("Mean", f"{np.mean(claim_severity):.2f}")
        st.metric("Median", f"{np.median(claim_severity):.2f}")
        st.metric("Standard deviation", f"{np.std(claim_severity, ddof=1):.2f}")
        st.metric("Maximum", f"{np.max(claim_severity):.2f}")
        st.markdown(
            "The distribution is clearly right-skewed: most claims fall between "
            "50 and 250, with a long tail extending beyond 500."
        )

    st.markdown("---")

    # Premium and Fees
    st.header("3. Premium and fees")
    c1, c2, c3 = st.columns(3)
    c1.metric("Average premium", f"{np.mean(premium):.2f}")
    c2.metric("Average fees", f"{np.mean(fees):.4f}")
    c3.metric("Total premium collected", f"{np.sum(premium):,.0f}")

    st.markdown("---")
    st.header("4. Portfolio composition")
    if all(c in df.columns for c in ["GENDER", "AREA", "CAR_TYPE", "CAR_USE"]):
        c1, c2 = st.columns(2)
        with c1:
            for col in ["GENDER", "AREA"]:
                vc = df[col].value_counts()
                fig = px.pie(
                    values=vc.values, names=vc.index,
                    title=f"By {col.lower()}",
                    color_discrete_sequence=["#1f3a5f", "#4682B4", "#8B0000",
                                             "#666666", "#a0522d"],
                )
                fig.update_layout(height=300, template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            for col in ["CAR_TYPE", "CAR_USE"]:
                vc = df[col].value_counts()
                fig = px.pie(
                    values=vc.values, names=vc.index,
                    title=f"By {col.replace('_', ' ').lower()}",
                    color_discrete_sequence=["#1f3a5f", "#4682B4", "#8B0000",
                                             "#666666", "#a0522d"],
                )
                fig.update_layout(height=300, template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: FREQUENCY MODEL
# ============================================================
elif page == "Frequency model":
    st.title("Frequency model selection")
    st.markdown(
        "Two candidate distributions are tested for the number of claims per policy. "
        "The Poisson distribution is included as a benchmark; the Negative Binomial "
        "is preferred when the data exhibit overdispersion."
    )

    st.markdown("---")
    st.header("Maximum-likelihood estimates")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Poisson")
        st.markdown(
            f"- Lambda: `{fit_pois['lambda']:.4f}`\n"
            f"- Log-likelihood: `{fit_pois['loglik']:.2f}`\n"
            f"- AIC: `{fit_pois['aic']:.2f}`\n"
            f"- BIC: `{fit_pois['bic']:.2f}`"
        )
    with c2:
        st.subheader("Negative Binomial")
        st.markdown(
            f"- Size: `{fit_nb['size']:.4f}`\n"
            f"- Mu: `{fit_nb['mu']:.4f}`\n"
            f"- Log-likelihood: `{fit_nb['loglik']:.2f}`\n"
            f"- AIC: `{fit_nb['aic']:.2f}`\n"
            f"- BIC: `{fit_nb['bic']:.2f}`"
        )

    st.markdown("---")
    st.header("Chi-squared goodness-of-fit test")

    gof_p = chisq_gof_freq(claim_count, fit_pois, "pois")
    gof_n = chisq_gof_freq(claim_count, fit_nb, "nbinom")

    gof_df = pd.DataFrame({
        "Distribution": ["Poisson", "Negative Binomial"],
        "Chi-squared": [gof_p["chi2"], gof_n["chi2"]],
        "Degrees of freedom": [gof_p["df"], gof_n["df"]],
        "p-value": [gof_p["p_value"], gof_n["p_value"]],
        "AIC": [fit_pois["aic"], fit_nb["aic"]],
    })
    st.dataframe(
        gof_df.style.format({
            "Chi-squared": "{:.4f}", "p-value": "{:.6f}", "AIC": "{:.2f}",
        }).highlight_min(axis=0, subset=["AIC"], color="#e8f0fe"),
        use_container_width=True,
    )

    if gof_n["p_value"] > 0.05 and gof_p["p_value"] < 0.05:
        st.markdown(
            f"> **Conclusion.** The Poisson model is rejected "
            f"(p = {gof_p['p_value']:.6f}), while the Negative Binomial "
            f"cannot be rejected (p = {gof_n['p_value']:.4f}). "
            f"The Negative Binomial is selected as the frequency model."
        )

    st.markdown("---")
    st.header("Observed versus fitted probability mass functions")

    freq_table = pd.Series(claim_count).value_counts().sort_index()
    k_seq = freq_table.index.values
    obs_prop = freq_table.values / freq_table.sum()
    p_pois = stats.poisson.pmf(k_seq, fit_pois["lambda"])
    p_nb = stats.nbinom.pmf(k_seq, fit_nb["size"], fit_nb["p"])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=k_seq, y=obs_prop, name="Observed",
        marker=dict(color=COLOR_SECONDARY, opacity=0.6),
    ))
    fig.add_trace(go.Scatter(
        x=k_seq, y=p_pois, name="Poisson",
        mode="lines+markers",
        line=dict(color=COLOR_ACCENT, width=2),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=k_seq, y=p_nb, name="Negative Binomial",
        mode="lines+markers",
        line=dict(color=COLOR_PRIMARY, width=2, dash="dash"),
        marker=dict(size=8),
    ))
    fig.update_layout(
        title="Claim frequency: observed versus fitted PMF",
        xaxis_title="Number of claims", yaxis_title="Proportion",
        template=PLOTLY_TEMPLATE, height=450,
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: SEVERITY MODEL
# ============================================================
elif page == "Severity model":
    st.title("Severity model selection")
    st.markdown(
        "Three candidate distributions are evaluated for individual claim amounts: "
        "Gamma, Lognormal, and Weibull. All three are defined on the positive real "
        "line and can accommodate right-skewed data."
    )

    st.markdown("---")
    st.header("Information criteria")

    aic_df = pd.DataFrame({
        "Distribution": list(sev_fits.keys()),
        "AIC": [sev_fits[k]["aic"] for k in sev_fits],
        "BIC": [sev_fits[k]["bic"] for k in sev_fits],
    })
    st.dataframe(
        aic_df.style.format({"AIC": "{:.2f}", "BIC": "{:.2f}"})
        .highlight_min(axis=0, subset=["AIC", "BIC"], color="#e8f0fe"),
        use_container_width=True,
    )

    ks = ks_test_severity(claim_severity, sev_fits)
    st.header("Kolmogorov-Smirnov test")
    ks_df = pd.DataFrame({
        "Distribution": list(ks.keys()),
        "D statistic": [ks[k].statistic for k in ks],
        "p-value": [ks[k].pvalue for k in ks],
    })
    st.dataframe(
        ks_df.style.format({"D statistic": "{:.5f}", "p-value": "{:.6f}"})
        .highlight_max(axis=0, subset=["p-value"], color="#e8f0fe"),
        use_container_width=True,
    )

    st.markdown(
        f"> **Conclusion.** The {best_sev_name} distribution provides the best fit "
        f"according to all three criteria (lowest AIC and BIC, highest KS p-value)."
    )

    st.markdown("---")
    st.header("Observed versus fitted densities")

    x_seq = np.linspace(claim_severity.min(), claim_severity.max(), 500)
    d_g = stats.gamma.pdf(x_seq, sev_fits["Gamma"]["shape"], scale=sev_fits["Gamma"]["scale"])
    d_ln = stats.lognorm.pdf(
        x_seq, sev_fits["Lognormal"]["scipy_s"], scale=sev_fits["Lognormal"]["scipy_scale"]
    )
    d_w = stats.weibull_min.pdf(
        x_seq, sev_fits["Weibull"]["shape"], scale=sev_fits["Weibull"]["scale"]
    )

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=claim_severity, histnorm="probability density",
        nbinsx=40, marker=dict(color=COLOR_SECONDARY, opacity=0.5),
        name="Observed",
    ))
    fig.add_trace(go.Scatter(x=x_seq, y=d_g, name="Gamma",
                             line=dict(color=COLOR_ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=x_seq, y=d_ln, name="Lognormal",
                             line=dict(color=COLOR_PRIMARY, width=2)))
    fig.add_trace(go.Scatter(x=x_seq, y=d_w, name="Weibull",
                             line=dict(color="#a0522d", width=2)))
    fig.update_layout(
        title="Claim severity: observed versus fitted densities",
        xaxis_title="Claim amount", yaxis_title="Density",
        template=PLOTLY_TEMPLATE, height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.header("Quantile-quantile plots")

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
                       marker=dict(color=COLOR_SECONDARY, size=4, opacity=0.5)),
            row=1, col=i,
        )
        ref = np.linspace(min(theo.min(), obs_q.min()),
                          max(theo.max(), obs_q.max()), 50)
        fig.add_trace(
            go.Scatter(x=ref, y=ref, mode="lines", showlegend=False,
                       line=dict(color=COLOR_ACCENT, width=1.5)),
            row=1, col=i,
        )
    fig.update_xaxes(title_text="Theoretical quantiles")
    fig.update_yaxes(title_text="Observed quantiles")
    fig.update_layout(height=400, template=PLOTLY_TEMPLATE,
                      title="Quantile-quantile plots — claim severity")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: MONTE CARLO AND VARIANCE REDUCTION
# ============================================================
elif page == "Monte Carlo and variance reduction":
    st.title("Monte Carlo estimation and variance reduction")
    st.markdown(
        "Monte Carlo simulation is used to estimate the expected claim frequency, "
        "the expected claim severity, and ultimately the aggregate loss "
        "distribution. Variance reduction techniques (antithetic variates and "
        "control variates) are tested to improve estimator efficiency."
    )

    n_mc = st.slider(
        "Number of Monte Carlo simulations",
        1000, 200000, 100000, step=1000,
    )

    rng = np.random.default_rng(42)

    st.markdown("---")
    st.header("Convergence of the running mean")

    if best_freq_name == "Poisson":
        sim_freq = rng.poisson(fit_pois["lambda"], size=n_mc)
        true_mean_freq = fit_pois["lambda"]
    else:
        sim_freq = stats.nbinom.rvs(fit_nb["size"], fit_nb["p"], size=n_mc, random_state=rng)
        true_mean_freq = fit_nb["mu"]
    running_freq = np.cumsum(sim_freq) / np.arange(1, n_mc + 1)

    sim_sev = rng.lognormal(
        sev_fits["Lognormal"]["meanlog"], sev_fits["Lognormal"]["sdlog"], size=n_mc,
    )
    true_mean_sev = float(np.exp(
        sev_fits["Lognormal"]["meanlog"] + sev_fits["Lognormal"]["sdlog"] ** 2 / 2
    ))
    running_sev = np.cumsum(sim_sev) / np.arange(1, n_mc + 1)

    c1, c2 = st.columns(2)
    with c1:
        step = max(1, n_mc // 2000)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(1, n_mc + 1)[::step], y=running_freq[::step],
            mode="lines", line=dict(color=COLOR_SECONDARY),
            name="Running mean",
        ))
        fig.add_hline(y=float(np.mean(claim_count)),
                      line=dict(color=COLOR_ACCENT, dash="dash"),
                      annotation_text="Empirical mean",
                      annotation_position="top right")
        fig.add_hline(y=true_mean_freq,
                      line=dict(color=COLOR_PRIMARY, dash="dot"),
                      annotation_text="Theoretical mean",
                      annotation_position="bottom right")
        fig.update_layout(
            title="Convergence — claim frequency",
            xaxis_title="Number of simulations", yaxis_title="Running mean",
            template=PLOTLY_TEMPLATE, height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        step = max(1, n_mc // 2000)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(1, n_mc + 1)[::step], y=running_sev[::step],
            mode="lines", line=dict(color=COLOR_SECONDARY),
            name="Running mean",
        ))
        fig.add_hline(y=float(np.mean(claim_severity)),
                      line=dict(color=COLOR_ACCENT, dash="dash"),
                      annotation_text="Empirical mean",
                      annotation_position="top right")
        fig.add_hline(y=true_mean_sev,
                      line=dict(color=COLOR_PRIMARY, dash="dot"),
                      annotation_text="Theoretical mean",
                      annotation_position="bottom right")
        fig.update_layout(
            title="Convergence — claim severity",
            xaxis_title="Number of simulations",
            yaxis_title="Running mean (claim amount)",
            template=PLOTLY_TEMPLATE, height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.header("Statistical properties of the Monte Carlo estimators")

    theo_var_freq = (
        (true_mean_freq + true_mean_freq ** 2 / fit_nb["size"]) / n_mc
        if best_freq_name == "Negative Binomial" else true_mean_freq / n_mc
    )
    emp_var_freq = float(np.var(sim_freq, ddof=1) / n_mc)
    mse_freq = float(
        np.mean((sim_freq - true_mean_freq) ** 2) / n_mc
        + (np.mean(sim_freq) - np.mean(claim_count)) ** 2
    )

    ml = sev_fits["Lognormal"]["meanlog"]
    sl = sev_fits["Lognormal"]["sdlog"]
    theo_var_sev = (np.exp(sl ** 2) - 1) * np.exp(2 * ml + sl ** 2) / n_mc
    emp_var_sev = float(np.var(sim_sev, ddof=1) / n_mc)
    mse_sev = float(
        np.mean((sim_sev - true_mean_sev) ** 2) / n_mc
        + (np.mean(sim_sev) - np.mean(claim_severity)) ** 2
    )

    props_df = pd.DataFrame({
        "Frequency": [theo_var_freq, emp_var_freq, mse_freq],
        "Severity": [theo_var_sev, emp_var_sev, mse_sev],
    }, index=["Theoretical variance", "Empirical variance", "Mean squared error"])
    st.dataframe(props_df.style.format("{:.8f}"), use_container_width=True)

    st.markdown("---")
    st.header("Variance reduction: antithetic variates")
    st.markdown(
        "For each uniform draw $U$, a paired draw $(U, 1 - U)$ is generated. "
        "Their average exploits the negative correlation induced by the "
        "transformation to reduce variance."
    )

    n_av = n_mc // 2
    U = rng.uniform(size=n_av)

    if best_freq_name == "Poisson":
        x1_f = stats.poisson.ppf(U, fit_pois["lambda"])
        x2_f = stats.poisson.ppf(1 - U, fit_pois["lambda"])
    else:
        x1_f = stats.nbinom.ppf(U, fit_nb["size"], fit_nb["p"])
        x2_f = stats.nbinom.ppf(1 - U, fit_nb["size"], fit_nb["p"])
    av_freq = (x1_f + x2_f) / 2
    var_av_freq = float(np.var(av_freq, ddof=1) / n_av)
    var_std_freq = float(np.var(sim_freq, ddof=1) / n_mc)

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
    c1.metric("Variance reduction — frequency", f"{red_freq_av:.2f}%")
    c2.metric("Variance reduction — severity", f"{red_sev_av:.2f}%")

    st.header("Variance reduction: control variate (Exponential)")
    st.markdown(
        "An Exponential variable with known mean is used as a control. "
        "The effectiveness of the technique depends on the correlation "
        "between the control and the target."
    )

    rng2 = np.random.default_rng(123)
    sim_freq_cv = (
        rng2.poisson(fit_pois["lambda"], size=n_mc)
        if best_freq_name == "Poisson"
        else stats.nbinom.rvs(fit_nb["size"], fit_nb["p"], size=n_mc, random_state=rng2)
    )
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
    c1.metric("Variance reduction — frequency", f"{red_freq_cv:.2f}%",
              delta=f"correlation = {corr_freq:.4f}", delta_color="off")
    c2.metric("Variance reduction — severity", f"{red_sev_cv:.2f}%",
              delta=f"correlation = {corr_sev:.4f}", delta_color="off")

    st.markdown("---")
    st.header("Comparison of methods")

    plot_df = pd.DataFrame({
        "Method": ["Standard Monte Carlo", "Antithetic variates", "Control variate"] * 2,
        "Variable": ["Frequency"] * 3 + ["Severity"] * 3,
        "Reduction (%)": [0, red_freq_av, red_freq_cv, 0, red_sev_av, red_sev_cv],
    })
    fig = px.bar(
        plot_df, x="Method", y="Reduction (%)", color="Method",
        facet_col="Variable", text="Reduction (%)",
        color_discrete_map={
            "Standard Monte Carlo": COLOR_NEUTRAL,
            "Antithetic variates": COLOR_SECONDARY,
            "Control variate": "#a0522d",
        },
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "> **Conclusion.** Antithetic variates achieve a substantial variance "
        "reduction owing to the negative correlation induced by construction. "
        "The control variate method, by contrast, performs poorly because the "
        "correlation between the target variable and the exponential control "
        "is close to zero. Antithetic variates are therefore the preferred "
        "technique for this portfolio."
    )


# ============================================================
# PAGE: RISK PREMIUM AND VAR
# ============================================================
elif page == "Risk premium and Value at Risk":
    st.title("Risk premium, Value at Risk, and Solvency Capital")
    st.markdown(
        "The aggregate claim amount $S$ for a single policy is the sum of $N$ "
        "individual claim amounts, where $N$ follows the fitted Negative Binomial "
        "distribution and each claim follows the fitted Lognormal. Since $S$ has "
        "no closed-form distribution, Monte Carlo simulation is used."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        n_sim = st.slider("Number of Monte Carlo simulations",
                          5000, 200000, N_SIM_DEFAULT, step=5000)
    with c2:
        alpha = st.slider("VaR confidence level", 0.90, 0.999, 0.995, step=0.005)
    with c3:
        bins_loss = st.slider("Histogram bins", 30, 100, 60)

    agg_loss, sim_counts = simulate_aggregate_loss(
        n_sim, best_freq_name, best_freq_params,
        best_sev_name, best_sev_params, seed=42,
    )

    risk_premium_mc = float(agg_loss.mean())
    avg_premium = float(np.mean(premium))
    avg_fees = float(np.mean(fees))
    risk_premium_emp = float(claim_severity.sum() / n_policies)
    margin = avg_premium - risk_premium_mc - avg_fees
    margin_pct = 100 * margin / avg_premium

    VaR = float(np.quantile(agg_loss, alpha))
    TVaR = float(agg_loss[agg_loss >= VaR].mean())
    SCR = VaR - risk_premium_mc

    st.markdown("---")
    st.header("Risk premium")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Empirical risk premium", f"{risk_premium_emp:.2f}")
    c2.metric("Monte Carlo risk premium", f"{risk_premium_mc:.2f}")
    c3.metric("Average premium collected", f"{avg_premium:.2f}")
    c4.metric("Profit margin", f"{margin_pct:.2f}%")

    st.markdown("---")
    st.header("Aggregate loss distribution")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=agg_loss, nbinsx=bins_loss,
        marker=dict(color=COLOR_SECONDARY, opacity=0.75,
                    line=dict(color="white", width=1)),
        showlegend=False,
    ))
    fig.add_vline(
        x=risk_premium_mc, line=dict(color=COLOR_PRIMARY, width=2, dash="dash"),
        annotation_text=f"Mean (RP) = {risk_premium_mc:.0f}",
        annotation_position="top",
    )
    fig.add_vline(
        x=VaR, line=dict(color=COLOR_ACCENT, width=2, dash="dash"),
        annotation_text=f"VaR {alpha * 100:.1f}% = {VaR:.0f}",
        annotation_position="top",
    )
    fig.add_vline(
        x=TVaR, line=dict(color="#a0522d", width=2, dash="dashdot"),
        annotation_text=f"TVaR {alpha * 100:.1f}% = {TVaR:.0f}",
        annotation_position="top",
    )
    fig.update_layout(
        title="Simulated aggregate loss distribution",
        xaxis_title="Aggregate loss S", yaxis_title="Count",
        template=PLOTLY_TEMPLATE, height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.header(f"Solvency Capital Requirement (Solvency II, alpha = {alpha:.3f})")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"VaR {alpha * 100:.1f}%", f"{VaR:.2f}")
    c2.metric(f"TVaR {alpha * 100:.1f}%", f"{TVaR:.2f}")
    c3.metric("SCR (VaR minus RP)", f"{SCR:.2f}")

    ratio = TVaR / risk_premium_mc
    st.markdown(
        f"> In the worst {(1 - alpha) * 100:.1f}% of scenarios, the average "
        f"aggregate loss reaches **{TVaR:.0f}**, approximately "
        f"**{ratio:.1f} times the expected loss**. The fact that the TVaR is "
        f"substantially higher than the VaR confirms a heavy right tail."
    )


# ============================================================
# PAGE: REINSURANCE
# ============================================================
elif page == "Reinsurance":
    st.title("Reinsurance strategy")
    st.markdown(
        "Although the portfolio is profitable, an Excess-of-Loss (XoL) "
        "reinsurance treaty can be used to protect against extreme losses. "
        "Use the slider below to choose a retention percentile and observe "
        "the impact on ceded loss and Solvency Capital Requirement."
    )

    n_sim = 50000
    agg_loss, _ = simulate_aggregate_loss(
        n_sim, best_freq_name, best_freq_params,
        best_sev_name, best_sev_params, seed=42,
    )

    risk_premium_mc = float(agg_loss.mean())

    retention_pct = st.slider(
        "Retention percentile", 0.80, 0.999, 0.95, step=0.005,
    )

    retention_limit = float(np.quantile(agg_loss, retention_pct))
    ceded = np.maximum(agg_loss - retention_limit, 0)
    expected_ceded = float(ceded.mean())
    pct_ceded = 100 * expected_ceded / risk_premium_mc
    pct_scenarios = 100 * float(np.mean(agg_loss > retention_limit))

    net_loss = np.minimum(agg_loss, retention_limit)
    new_VaR995 = float(np.quantile(net_loss, 0.995))
    old_VaR995 = float(np.quantile(agg_loss, 0.995))
    new_SCR = new_VaR995 - risk_premium_mc
    old_SCR = old_VaR995 - risk_premium_mc

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retention limit", f"{retention_limit:.2f}",
              delta=f"P{retention_pct * 100:.1f}", delta_color="off")
    c2.metric("Expected loss ceded", f"{expected_ceded:.2f}",
              delta=f"{pct_ceded:.2f}% of total risk", delta_color="off")
    c3.metric("Scenarios triggered", f"{pct_scenarios:.2f}%")
    c4.metric("SCR reduction", f"{old_SCR - new_SCR:.0f}",
              delta=f"-{(old_SCR - new_SCR) / old_SCR * 100:.1f}%")

    st.markdown("---")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Before reinsurance", "After reinsurance (net)"),
    )
    fig.add_trace(go.Histogram(
        x=agg_loss, nbinsx=60,
        marker=dict(color=COLOR_SECONDARY, opacity=0.75),
        showlegend=False,
    ), row=1, col=1)
    fig.add_vline(x=retention_limit, line=dict(color=COLOR_ACCENT, dash="dash"),
                  annotation_text=f"Retention = {retention_limit:.0f}",
                  row=1, col=1)
    fig.add_vline(x=old_VaR995, line=dict(color=COLOR_PRIMARY, dash="dot"),
                  annotation_text=f"VaR 99.5% = {old_VaR995:.0f}",
                  row=1, col=1)

    fig.add_trace(go.Histogram(
        x=net_loss, nbinsx=60,
        marker=dict(color=COLOR_PRIMARY, opacity=0.75),
        showlegend=False,
    ), row=1, col=2)
    fig.add_vline(x=new_VaR995, line=dict(color=COLOR_ACCENT, dash="dot"),
                  annotation_text=f"VaR 99.5% = {new_VaR995:.0f}",
                  row=1, col=2)

    fig.update_xaxes(title_text="Aggregate loss S")
    fig.update_yaxes(title_text="Count")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"> With a retention at the {retention_pct * 100:.1f}th percentile "
        f"({retention_limit:.0f}), only {pct_scenarios:.2f}% of scenarios "
        f"trigger reinsurance. The expected loss ceded is {expected_ceded:.2f} "
        f"per policy ({pct_ceded:.2f}% of total expected risk). "
        f"The Solvency Capital Requirement falls from {old_SCR:.0f} to {new_SCR:.0f}."
    )


# ============================================================
# PAGE: SENSITIVITY ANALYSIS
# ============================================================
elif page == "Sensitivity analysis":
    st.title("Sensitivity analysis")
    st.markdown(
        "The robustness of the results is assessed by simulating the aggregate "
        "loss under four alternative model combinations: Negative Binomial "
        "versus Poisson for frequency, and Lognormal versus Gamma for severity."
    )

    n_sens = st.slider(
        "Simulations per scenario", 5000, 100000, 30000, step=5000,
    )

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
                n_sens, freq_model, freq_params, sname, sev_params, seed=42 + i,
            )
            rp = float(agg.mean())
            var = float(np.quantile(agg, 0.995))
            scr = var - rp
            rows.append({
                "Frequency": fname, "Severity": sname,
                "Risk premium": rp, "VaR 99.5%": var, "SCR": scr,
            })
        return pd.DataFrame(rows)

    sens_df = run_sens(n_sens, fit_pois, fit_nb, sev_fits)

    st.dataframe(
        sens_df.style.format({
            "Risk premium": "{:.2f}", "VaR 99.5%": "{:.2f}", "SCR": "{:.2f}",
        }),
        use_container_width=True,
    )

    st.markdown("---")

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Risk premium", "VaR 99.5%", "SCR"),
    )
    labels = sens_df["Frequency"] + " + " + sens_df["Severity"]
    fig.add_trace(go.Bar(x=labels, y=sens_df["Risk premium"], showlegend=False,
                         marker_color=COLOR_SECONDARY), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=sens_df["VaR 99.5%"], showlegend=False,
                         marker_color=COLOR_PRIMARY), row=1, col=2)
    fig.add_trace(go.Bar(x=labels, y=sens_df["SCR"], showlegend=False,
                         marker_color=COLOR_ACCENT), row=1, col=3)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=400)
    fig.update_xaxes(tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "> **Interpretation.** The choice of frequency model has a substantially "
        "larger impact on tail-risk measures than the choice of severity model. "
        "The Poisson model underestimates overdispersion and therefore produces "
        "a lower (more optimistic) Solvency Capital Requirement than the "
        "Negative Binomial. The risk premium itself remains stable across all "
        "combinations, indicating robustness in the central tendency."
    )


# ============================================================
# PAGE: CONCLUSION AND RECOMMENDATIONS
# ============================================================
elif page == "Conclusion and recommendations":
    st.title("Conclusion and recommendations")

    avg_premium = float(np.mean(premium))
    avg_fees = float(np.mean(fees))
    total_claims_data = float(claim_severity.sum())
    combined_ratio = (total_claims_data + np.sum(fees)) / np.sum(premium)

    # Monte Carlo risk premium for consistency with the methodology
    # used in the written report
    agg_loss_concl, _ = simulate_aggregate_loss(
        N_SIM_DEFAULT, best_freq_name, best_freq_params,
        best_sev_name, best_sev_params, seed=42,
    )
    risk_premium_mc = float(agg_loss_concl.mean())
    margin_mc = avg_premium - risk_premium_mc - avg_fees
    margin_pct = 100 * margin_mc / avg_premium

    st.header("Tariff what-if simulator")
    st.markdown(
        "The slider below estimates the impact of a hypothetical premium "
        "reduction on the combined ratio and profit margin. The combined "
        "ratio shown is computed using the Monte Carlo risk premium, "
        "consistent with the methodology of the written report."
    )
    reduction_pct = st.slider(
        "Premium reduction (%)", 0.0, 20.0, 0.0, step=0.5,
    )

    new_premium = avg_premium * (1 - reduction_pct / 100)
    new_total_premium = np.sum(premium) * (1 - reduction_pct / 100)
    # Use MC risk premium scaled to the portfolio for consistency
    expected_total_claims = risk_premium_mc * n_policies
    new_combined_ratio = (expected_total_claims + np.sum(fees)) / new_total_premium

    c1, c2, c3 = st.columns(3)
    c1.metric("New average premium", f"{new_premium:.2f}",
              delta=f"-{reduction_pct:.1f}%", delta_color="off")
    c2.metric("New combined ratio (MC)", f"{new_combined_ratio:.3f}")
    c3.metric("New profit margin",
              f"{(1 - new_combined_ratio) * 100:.2f}%")

    if new_combined_ratio >= 1:
        st.markdown(
            f"> **Warning.** A reduction of {reduction_pct:.1f}% pushes the "
            f"combined ratio above 1.0. The portfolio becomes unprofitable."
        )
    elif reduction_pct > margin_pct - 2:
        st.markdown(
            f"> **Caution.** A reduction of {reduction_pct:.1f}% leaves only "
            f"{(1 - new_combined_ratio) * 100:.2f}% of margin, which is risky "
            f"given the heavy-tailed loss distribution."
        )
    else:
        st.markdown(
            f"> A reduction of {reduction_pct:.1f}% keeps the portfolio "
            f"profitable, with a margin of "
            f"{(1 - new_combined_ratio) * 100:.2f}%."
        )

    st.markdown("---")

    st.header("Final recommendations")
    st.markdown(
        f"""
**Tariff strategy: maintain the current premium.**
The portfolio is currently profitable with a combined ratio of
**{combined_ratio:.3f}** and a Monte Carlo profit margin of
**{margin_pct:.2f}%**. Any reduction above approximately
{margin_pct:.0f}% would push the company into a loss-making position.
Given the heavy-tailed loss distribution, where the TVaR at 99.5% is
approximately four times the expected loss, a reduction of the tariff
is not advised.

**Reinsurance: implement an Excess-of-Loss treaty.**
The recommended attachment point is approximately 1,073, corresponding
to the 95th percentile of the simulated aggregate loss distribution.
This structure transfers the most severe tail events to the reinsurer
while only ceding around 2.83% of the total expected risk. The Solvency
Capital Requirement is meaningfully reduced and financial stability is
enhanced.

**Limitations and model risk.**
The Negative Binomial / Lognormal combination is the most conservative
choice for SCR estimation and is therefore prudent. A strong positive
correlation (Spearman's rho approximately 0.88) was observed between
frequency and severity, which violates the independence assumption
underlying the compound model; future work should consider a
copula-based approach. The analysis is based on a single year of data;
inflation, seasonality, and macroeconomic effects are not modelled.
        """
    )

    st.markdown("---")
    st.caption(
        "Mazy Djezzar — Prescilya Fabi — Samantha López. "
        "HEC University of Lausanne, Master's in Actuarial Science."
    )
