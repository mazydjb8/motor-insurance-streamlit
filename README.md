# 🚗 Motor Insurance Portfolio Analysis — Streamlit Dashboard

Interactive dashboard built on top of the actuarial analysis of a motor
insurance portfolio (1,002 policyholders). Replicates the R/Rmd analysis
in Python and adds interactive sliders for Monte Carlo simulations,
VaR / TVaR / SCR computation, and a reinsurance simulator.

**Authors:** Mazy Djezzar · Prescilya Fabi · Samantha López
**Course:** Simulation Methods in Finance and Insurance — HEC University of Lausanne
**Master's in Actuarial Science**

---

## 📑 Sections of the dashboard

1. **🏠 Executive Summary** — headline metrics, profitability, recommendations
2. **📊 Data Analysis** — frequency, severity, premium, demographic breakdown
3. **🎲 Frequency Model** — Poisson vs Negative Binomial (AIC, BIC, χ²)
4. **💰 Severity Model** — Gamma vs Lognormal vs Weibull (AIC, KS, QQ-plots)
5. **🔄 Monte Carlo & Variance Reduction** — convergence, antithetic & control variates
6. **📈 Risk Premium & VaR** — interactive aggregate loss, VaR / TVaR / SCR
7. **🛡️ Reinsurance** — XoL retention slider, ceded loss, SCR reduction
8. **🔬 Sensitivity Analysis** — robustness across the 4 model combinations
9. **📝 Conclusion & Recommendations** — tariff what-if simulator

---

## 🧰 Run locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/motor-insurance-streamlit.git
cd motor-insurance-streamlit

# 2. (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## ☁️ Deploy to Streamlit Community Cloud (free)

1. **Push this folder to a public GitHub repo** (see *Push to GitHub* below).
2. Go to **<https://share.streamlit.io>** and sign in with GitHub.
3. Click **"New app"**, then:
   - **Repository**: `your-username/motor-insurance-streamlit`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy**. After ~2 minutes you'll have a public URL like
   `https://your-app-name.streamlit.app`.

That's it — no server, no Docker, no payment.

---

## 🐙 Push to GitHub (step-by-step)

```bash
# In the streamlit_app/ folder
git init
git add .
git commit -m "Initial commit: motor insurance dashboard"

# Create a new repo on github.com (Public, no README/license — leave empty),
# then copy the URL it gives you and run:
git branch -M main
git remote add origin https://github.com/<your-username>/motor-insurance-streamlit.git
git push -u origin main
```

**Tip:** if `git push` asks for a password, use a [Personal Access Token](https://github.com/settings/tokens)
(Settings → Developer settings → Personal access tokens), not your GitHub password.

---

## 📂 Project structure

```
streamlit_app/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .streamlit/
│   └── config.toml         # Theme (steel blue)
└── data/
    └── DATA_SET_2.csv      # Portfolio data (1,002 policies)
```

---

## 🔌 Data sources

The app supports three data sources (selectable from the sidebar):

1. **Bundled CSV** — uses `data/DATA_SET_2.csv` (default, works out of the box)
2. **Google Sheets URL** — paste a link to a publicly-shared Google Sheet
   (must be shared as *"Anyone with the link can view"*). The default URL points
   to the project sheet, sheet name `data`.
3. **Upload CSV** — upload a custom file (auto-detects `;` or `,` separator)

---

## 📊 Methodology recap

- **Frequency model:** Negative Binomial (chosen by AIC = 3 834.74 vs 3 859.73 for Poisson)
- **Severity model:** Lognormal (chosen by AIC = 28 830.81; KS p = 0.63)
- **Aggregate loss:** Compound NB-Lognormal, simulated via 50 000 Monte Carlo draws
- **Variance reduction:** Antithetic variates (~85% reduction); control variate
  with Exponential proves ineffective due to near-zero correlation
- **Solvency Capital:** SCR = VaR(99.5%) − E[S] under Solvency II
- **Reinsurance:** Excess-of-Loss treaty at 95th percentile retention

---

## 📝 License

Academic project — for educational use.
