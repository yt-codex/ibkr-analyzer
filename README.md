# IBKR Portfolio Analyzer (Streamlit)

A modern dark-mode Streamlit app for analyzing Interactive Brokers (IBKR) Portfolio Analyst CSV reports.

## What this app does

- Upload and parse IBKR Portfolio Analyst CSV reports.
- Show a comprehensive dashboard with:
  - Overview KPIs and NAV trend
  - Benchmark and portfolio performance comparisons
  - Drawdown and monthly return heatmap
  - Holdings breakdowns (sector, currency, top positions)
  - Cashflow, dividends, fees, interest, and projected income
  - Risk and ESG sections
  - Raw table explorer for all parsed sections

## Privacy / Data Handling

- Uploaded files are processed **in memory** for the active Streamlit session.
- The app **does not write uploaded data to disk**, database, or external storage.
- No caching is used for uploaded report contents.

## IBKR export steps (concise)

1. Sign in to **IBKR Client Portal**.
2. Go to **Performance & Reports**.
3. Open **PortfolioAnalyst** and select account/date range.
4. Export/download report as **CSV**.
5. Upload that CSV in this app.

## Local setup

### 1) Create virtual environment (recommended)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Run the app

```powershell
streamlit run app.py
```

The app will open in your browser.

## Deploy online with Streamlit Community Cloud

### Prerequisites

- A GitHub account
- A Streamlit account (uses GitHub sign-in)

### If you don't have a Streamlit account yet

1. Go to `https://share.streamlit.io/`
2. Click **Continue with GitHub** and authorize Streamlit.
3. Your Streamlit account is created during this sign-in flow.

### Deploy steps

1. Push this project to a GitHub repository.
2. Open `https://share.streamlit.io/` and click **New app**.
3. Select your repo/branch.
4. Set **Main file path** to `app.py`.
5. Click **Deploy**.

### Notes

- No secrets are required for this app.
- Upload privacy behavior remains the same on deployment: in-memory session processing, no file persistence by this app.
