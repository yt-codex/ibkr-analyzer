from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from ibkr_analyzer.report_utils import (
    ParsedIBKRReport,
    build_report_summary_html,
    extract_report_period,
    find_profile_info,
    parse_ibkr_report,
    period_years,
)
from ibkr_analyzer.ui.constants import set_active_theme
from ibkr_analyzer.ui import (
    inject_custom_css,
    render_cashflow_income_tab,
    render_concentration_tab,
    render_holdings_tab,
    render_overview_tab,
    render_performance_tab,
    render_raw_tables_tab,
    render_risk_esg_tab,
)


def streamlit_app() -> None:
    st.set_page_config(
        page_title="IBKR Portfolio Analyzer",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.markdown("### Appearance")
        use_editorial_theme = st.toggle(
            "Use editorial theme",
            value=bool(st.session_state.get("use_editorial_theme", False)),
            key="use_editorial_theme",
            help="Switch to a warmer graphite-and-copper presentation with more editorial typography.",
        )
        st.caption("Default: Slate + Mint. Toggle on for Warm Graphite + Copper.")

    theme_name = "editorial" if use_editorial_theme else "slate_mint"
    set_active_theme(theme_name)
    inject_custom_css(theme_name)

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-copy">
                <div class="hero-eyebrow">Interactive Brokers Portfolio Analyst</div>
                <div class="hero-title">IBKR Portfolio Analyzer</div>
                <p class="hero-sub">
                    Upload a Portfolio Analyst CSV and turn it into a boardroom-style view of
                    performance, concentration, cashflows, benchmarks, and portfolio risk.
                </p>
            </div>
            <div class="hero-badges">
                <span class="hero-badge">In-memory session only</span>
                <span class="hero-badge">Benchmark-aware dashboards</span>
                <span class="hero-badge">Raw tables preserved</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Upload Report")
        st.caption("Accepted format: `*.csv` from IBKR Portfolio Analyst.")

        uploaded_file = st.file_uploader(
            "IBKR CSV report",
            type=["csv"],
            help="Your file is processed in memory during this session only.",
        )

        sample_files = sorted(Path("data").glob("*.csv"))
        use_sample = st.checkbox(
            "Use bundled sample report",
            value=False,
            disabled=not bool(sample_files),
            help="Useful for testing the app before uploading your own report.",
        )

        st.markdown(
            """
            <div class="panel">
                <b>Privacy:</b> This app does not write uploaded reports to disk, database, or external storage.
                Parsing and charting are done in-memory only for your active session.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("How to export this report from IBKR"):
            st.markdown(
                """
                1. Sign in to **IBKR Client Portal**.
                2. Go to **Performance & Reports**.
                3. Open **PortfolioAnalyst**.
                4. Click **Report**.
                5. Download the **since inception** report as **CSV**.
                6. Upload the CSV file in this app.
                """
            )

    report_bytes: bytes | None = None
    report_source = ""

    if use_sample and sample_files:
        sample_path = sample_files[0]
        report_bytes = sample_path.read_bytes()
        report_source = f"Sample report: {sample_path.name}"
    elif uploaded_file is not None:
        report_bytes = uploaded_file.getvalue()
        report_source = f"Uploaded report: {uploaded_file.name}"

    if report_bytes is None:
        st.info("Upload your IBKR CSV report (or enable the sample file) to start analysis.")
        return

    try:
        report_digest = hashlib.sha256(report_bytes).hexdigest()
        cached_digest = st.session_state.get("report_cache_digest")
        cached_report = st.session_state.get("report_cache_value")
        if cached_digest == report_digest and isinstance(cached_report, ParsedIBKRReport):
            report = cached_report
        else:
            report = parse_ibkr_report(report_bytes)
            st.session_state["report_cache_digest"] = report_digest
            st.session_state["report_cache_value"] = report
    except Exception as error:  # noqa: BLE001
        st.error(f"Failed to parse report: {error}")
        return

    if not report.tables:
        st.error(
            "No report tables were parsed. Please confirm you uploaded an IBKR Portfolio Analyst CSV."
        )
        return

    profile, key_stats_row = find_profile_info(report)
    account_name = profile.get("Name", "Unknown")
    account_id = profile.get("Account", "")
    base_currency = profile.get("BaseCurrency", "")
    analysis_period = profile.get("AnalysisPeriod", "")
    performance_measure = profile.get("PerformanceMeasure", "")
    period_start, period_end = extract_report_period(report, profile)
    analysis_years = period_years(period_start, period_end)
    period_length_display = f"{analysis_years:.2f} years" if pd.notna(analysis_years) else "-"

    st.markdown(
        build_report_summary_html(
            report_source=report_source,
            account_name=account_name,
            account_id=account_id,
            base_currency=base_currency,
            performance_measure=performance_measure,
            analysis_period=analysis_period,
            period_length_display=period_length_display,
            parsed_sections=len(report.tables),
        ),
        unsafe_allow_html=True,
    )

    (
        overview_tab,
        performance_tab,
        holdings_tab,
        concentration_tab,
        cashflow_tab,
        risk_esg_tab,
        raw_tab,
    ) = st.tabs(
        [
            "Overview",
            "Performance",
            "Holdings",
            "Concentration",
            "Cashflow & Income",
            "Risk & ESG",
            "Raw Tables",
        ]
    )

    with overview_tab:
        render_overview_tab(
            report,
            key_stats_row,
            base_currency,
            analysis_years,
            performance_measure,
        )

    with performance_tab:
        render_performance_tab(report, account_id, performance_measure)

    with holdings_tab:
        render_holdings_tab(report, base_currency)

    with concentration_tab:
        render_concentration_tab(report)

    with cashflow_tab:
        render_cashflow_income_tab(report, base_currency)

    with risk_esg_tab:
        render_risk_esg_tab(report)

    with raw_tab:
        render_raw_tables_tab(report)


if __name__ == "__main__":
    streamlit_app()
