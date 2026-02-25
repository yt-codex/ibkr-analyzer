from __future__ import annotations

import csv
import html
import io
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


@dataclass
class ParsedIBKRReport:
    tables: dict[str, list[pd.DataFrame]]
    metadata: dict[str, list[list[str]]]


NULL_MARKERS = {"", "-", "--", "N/A", "NA", "null", "None"}
PLOTLY_TEMPLATE = "plotly_dark"
CHART_COLORS = ["#28d5b5", "#5ca3ff", "#ff8e53", "#ff5f8f", "#81f495", "#eeb6ff"]


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=DM+Sans:wght@400;500;700&display=swap');

            :root {
                --card-bg: rgba(16, 23, 42, 0.78);
                --card-border: rgba(131, 151, 183, 0.22);
                --text-soft: #9fb1d1;
                --accent: #28d5b5;
            }

            html, body, [class*="css"] {
                font-family: "DM Sans", "Segoe UI", sans-serif;
            }

            .stApp {
                background:
                    radial-gradient(circle at 10% 5%, rgba(40, 213, 181, 0.18), transparent 35%),
                    radial-gradient(circle at 90% 10%, rgba(92, 163, 255, 0.2), transparent 40%),
                    linear-gradient(165deg, #0a0f1e 0%, #0d1328 50%, #101a34 100%);
                color: #e8efff;
            }

            .block-container {
                max-width: 1300px;
                padding-top: 2.2rem;
                padding-bottom: 1.25rem;
            }

            h1, h2, h3 {
                font-family: "Space Grotesk", "Segoe UI", sans-serif;
                letter-spacing: -0.02em;
            }

            .hero-card {
                background: linear-gradient(140deg, rgba(40, 213, 181, 0.15), rgba(92, 163, 255, 0.14));
                border: 1px solid rgba(148, 189, 255, 0.25);
                border-radius: 18px;
                padding: 1.1rem 1.25rem;
                margin-top: 0.35rem;
                margin-bottom: 1rem;
                box-shadow: 0 16px 32px rgba(4, 8, 20, 0.45);
            }

            .hero-title {
                font-size: 1.75rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }

            .hero-sub {
                color: var(--text-soft);
                font-size: 0.95rem;
                margin-bottom: 0;
            }

            .panel {
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                border-radius: 14px;
                padding: 0.8rem 0.95rem;
                margin-bottom: 0.9rem;
                backdrop-filter: blur(8px);
            }

            div[data-testid="stMetric"] {
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                padding: 0.55rem 0.75rem;
                border-radius: 12px;
            }

            div[data-testid="stMetricLabel"] {
                color: #9bb0d6;
            }

            div[data-testid="stMetricValue"] {
                color: #f5fbff;
                font-family: "Space Grotesk", "Segoe UI", sans-serif;
            }

            .method-tip {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                font-family: "Space Grotesk", "Segoe UI", sans-serif;
                font-size: 1rem;
                font-weight: 700;
                margin: 0.2rem 0 0.35rem 0;
                color: #eff5ff;
            }

            .hint-icon {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 1rem;
                height: 1rem;
                border-radius: 999px;
                border: 1px solid rgba(160, 185, 225, 0.5);
                color: #aec5ed;
                font-size: 0.7rem;
                cursor: help;
                line-height: 1rem;
            }

            .hint-value {
                font-family: "DM Sans", "Segoe UI", sans-serif;
                font-size: 0.75rem;
                font-weight: 600;
                color: #a8bce0;
                background: rgba(95, 125, 178, 0.2);
                border: 1px solid rgba(151, 178, 224, 0.28);
                border-radius: 999px;
                padding: 0.1rem 0.44rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def make_unique_headers(headers: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique_headers: list[str] = []
    for index, header in enumerate(headers, start=1):
        cleaned = header.strip() if header else ""
        cleaned = cleaned or f"Column_{index}"
        counts[cleaned] = counts.get(cleaned, 0) + 1
        if counts[cleaned] > 1:
            cleaned = f"{cleaned}_{counts[cleaned]}"
        unique_headers.append(cleaned)
    return unique_headers


def parse_ibkr_report(file_bytes: bytes) -> ParsedIBKRReport:
    decoded = file_bytes.decode("utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(decoded))

    raw_tables: dict[str, list[dict[str, list]]] = defaultdict(list)
    metadata: dict[str, list[list[str]]] = defaultdict(list)
    active_index: dict[str, int] = {}

    for row in reader:
        if len(row) < 2:
            continue

        section = row[0].strip()
        row_type = row[1].strip()
        payload = [cell.strip() for cell in row[2:]]

        if row_type == "Header":
            raw_tables[section].append(
                {
                    "columns": make_unique_headers(payload),
                    "rows": [],
                }
            )
            active_index[section] = len(raw_tables[section]) - 1
            continue

        if row_type == "MetaInfo":
            metadata[section].append(payload)
            continue

        if row_type != "Data":
            continue

        if section not in active_index:
            raw_tables[section].append(
                {
                    "columns": make_unique_headers(
                        [f"Column_{idx}" for idx in range(1, len(payload) + 1)]
                    ),
                    "rows": [],
                }
            )
            active_index[section] = len(raw_tables[section]) - 1

        table = raw_tables[section][active_index[section]]
        columns = table["columns"]
        row_values = payload[:]

        if len(row_values) < len(columns):
            row_values.extend([""] * (len(columns) - len(row_values)))
        elif len(row_values) > len(columns):
            extra_columns_count = len(row_values) - len(columns)
            start = len(columns) + 1
            extra_columns = [
                f"Extra_{idx}" for idx in range(start, start + extra_columns_count)
            ]
            table["columns"].extend(extra_columns)
            for existing_row in table["rows"]:
                existing_row.extend([""] * extra_columns_count)

        table["rows"].append(row_values)

    tables: dict[str, list[pd.DataFrame]] = {}
    for section, section_tables in raw_tables.items():
        parsed_tables: list[pd.DataFrame] = []
        for section_table in section_tables:
            data_frame = pd.DataFrame(
                section_table["rows"], columns=section_table["columns"]
            )
            for column in data_frame.columns:
                data_frame[column] = data_frame[column].map(
                    lambda value: value.strip() if isinstance(value, str) else value
                )
            parsed_tables.append(data_frame)
        tables[section] = parsed_tables

    return ParsedIBKRReport(tables=tables, metadata=dict(metadata))


def get_table(
    report: ParsedIBKRReport,
    section: str,
    index: int = 0,
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    section_tables = report.tables.get(section, [])
    if not section_tables:
        return pd.DataFrame()

    if required_columns:
        required = set(required_columns)
        for table in section_tables:
            if required.issubset(table.columns):
                return table.copy()
        return pd.DataFrame()

    if index < 0 or index >= len(section_tables):
        return pd.DataFrame()
    return section_tables[index].copy()


def parse_number(value: object) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (float, int)):
        return float(value)

    text = str(value).strip()
    if text in NULL_MARKERS:
        return np.nan

    text = text.replace(",", "").replace("%", "")
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1]}"

    try:
        return float(text)
    except ValueError:
        return np.nan


def to_numeric(series: pd.Series) -> pd.Series:
    return series.map(parse_number)


def parse_report_date(value: object) -> pd.Timestamp:
    if value is None:
        return pd.NaT

    text = str(value).strip()
    if not text or text in NULL_MARKERS or text.lower() == "total":
        return pd.NaT

    for date_format in ("%Y%m%d", "%Y%m", "%m/%d/%Y", "%m/%d/%y", "%b %Y", "%B %Y"):
        parsed = pd.to_datetime(text, format=date_format, errors="coerce")
        if pd.notna(parsed):
            return parsed

    quarter_match = re.fullmatch(r"(\d{4})\s*Q([1-4])", text)
    if quarter_match:
        year = int(quarter_match.group(1))
        quarter = int(quarter_match.group(2))
        return pd.Timestamp(year=year, month=((quarter - 1) * 3) + 1, day=1)

    return pd.to_datetime(text, errors="coerce")


def parse_analysis_period_text(text: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    cleaned = re.sub(r"\(.*?\)", "", str(text)).strip()
    cleaned = cleaned.replace(" to ", " - ")
    parts = re.split(r"\s*-\s*", cleaned, maxsplit=1)
    if len(parts) != 2:
        return pd.NaT, pd.NaT

    start = pd.to_datetime(parts[0].strip(), errors="coerce")
    end = pd.to_datetime(parts[1].strip(), errors="coerce")
    return start, end


def period_years(start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    if pd.isna(start_date) or pd.isna(end_date):
        return np.nan
    days = (end_date - start_date).days
    if days <= 0:
        return np.nan
    return days / 365.25


def annualize_return(cumulative_return_pct: float, years: float) -> float:
    if pd.isna(cumulative_return_pct) or pd.isna(years) or years <= 0:
        return np.nan
    growth = 1 + (cumulative_return_pct / 100.0)
    if growth <= 0:
        return np.nan
    return ((growth ** (1 / years)) - 1) * 100


def extract_report_period(
    report: ParsedIBKRReport, profile: dict[str, str]
) -> tuple[pd.Timestamp, pd.Timestamp]:
    key_stats_meta = report.metadata.get("Key Statistics", [])
    for row in key_stats_meta:
        if len(row) >= 2 and "analysis period" in str(row[0]).lower():
            start, end = parse_analysis_period_text(row[1])
            if pd.notna(start) and pd.notna(end):
                return start, end

    profile_period = profile.get("AnalysisPeriod", "")
    if profile_period:
        start, end = parse_analysis_period_text(profile_period)
        if pd.notna(start) and pd.notna(end):
            return start, end

    allocation = get_table(report, "Allocation by Asset Class", required_columns=["Date"])
    if not allocation.empty:
        parsed_dates = allocation["Date"].map(parse_report_date).dropna().sort_values()
        if not parsed_dates.empty:
            return parsed_dates.iloc[0], parsed_dates.iloc[-1]

    return pd.NaT, pd.NaT


def return_method_label_and_tooltip(performance_measure: str) -> tuple[str, str]:
    normalized = str(performance_measure or "").strip().upper()
    if normalized == "TWR":
        return (
            "TWR",
            "IBKR PerformanceMeasure is TWR (Time-Weighted Return). "
            "Cumulative and annualized returns here are time-weighted and chain-linked.",
        )
    if normalized == "MWR":
        return (
            "MWR",
            "IBKR PerformanceMeasure is MWR (Money-Weighted Return). "
            "Cumulative and annualized returns here are cash-flow weighted.",
        )
    if normalized:
        return (
            normalized,
            f"IBKR PerformanceMeasure is reported as {normalized}. "
            "Cumulative and annualized returns follow that method.",
        )
    return (
        "Unknown",
        "Performance method was not found in the report. "
        "Cumulative and annualized returns are derived from the report's return series.",
    )


def format_money(value: float, currency: str = "") -> str:
    if pd.isna(value):
        return "-"
    prefix = f"{currency} " if currency else ""
    return f"{prefix}{value:,.2f}"


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.2f}%"


def sanitize_total_rows(
    data_frame: pd.DataFrame, column_name: str, drop_blank: bool = False
) -> pd.DataFrame:
    filtered = data_frame.copy()
    if column_name not in filtered.columns:
        return filtered

    mask = filtered[column_name].astype(str).str.contains("Total", case=False, na=False)
    filtered = filtered.loc[~mask]
    if drop_blank:
        filtered = filtered.loc[filtered[column_name].astype(str).str.strip() != ""]
    return filtered


def build_benchmark_long(data_frame: pd.DataFrame) -> pd.DataFrame:
    required = {"Date", "BM1", "BM1Return", "BM2", "BM2Return", "BM3", "BM3Return"}
    if data_frame.empty or not required.issubset(set(data_frame.columns)):
        return pd.DataFrame()

    working = data_frame.copy()
    working["DateParsed"] = working["Date"].map(parse_report_date)

    series_pairs: list[tuple[str, str]] = []
    for benchmark_col, return_col in (
        ("BM1", "BM1Return"),
        ("BM2", "BM2Return"),
        ("BM3", "BM3Return"),
    ):
        benchmark_name_series = (
            working[benchmark_col].replace("", np.nan).dropna()
            if benchmark_col in working.columns
            else pd.Series(dtype=object)
        )
        benchmark_name = (
            str(benchmark_name_series.iloc[0])
            if not benchmark_name_series.empty
            else benchmark_col
        )
        if return_col in working.columns:
            series_pairs.append((benchmark_name, return_col))

    known_columns = {
        "Date",
        "DateParsed",
        "BM1",
        "BM1Return",
        "BM2",
        "BM2Return",
        "BM3",
        "BM3Return",
    }
    extra_columns = [column for column in working.columns if column not in known_columns]
    account_return_column = ""
    account_name = ""

    return_like_columns = [
        column for column in extra_columns if column.lower().endswith("return")
    ]
    if return_like_columns:
        account_return_column = return_like_columns[0]
        account_name_candidates = [
            column for column in extra_columns if column != account_return_column
        ]
        if account_name_candidates:
            account_series = (
                working[account_name_candidates[0]].replace("", np.nan).dropna()
            )
            if not account_series.empty:
                account_name = str(account_series.iloc[0])
        if not account_name:
            account_name = account_return_column.replace("Return", "").strip() or "Portfolio"
    elif extra_columns:
        account_return_column = extra_columns[-1]
        account_name = account_return_column.replace("Return", "").strip() or "Portfolio"

    if account_return_column and account_return_column in working.columns:
        series_pairs.append((account_name, account_return_column))

    normalized_frames: list[pd.DataFrame] = []
    for series_name, return_column in series_pairs:
        normalized = pd.DataFrame(
            {
                "Date": working["DateParsed"],
                "Series": series_name,
                "Return": to_numeric(working[return_column]),
            }
        )
        normalized = normalized.dropna(subset=["Date", "Return"])
        normalized_frames.append(normalized)

    if not normalized_frames:
        return pd.DataFrame()
    return pd.concat(normalized_frames, ignore_index=True).sort_values("Date")


def find_profile_info(report: ParsedIBKRReport) -> tuple[dict[str, str], pd.Series]:
    profile: dict[str, str] = {}
    intro_table = get_table(report, "Introduction")
    if not intro_table.empty:
        intro_row = intro_table.iloc[0]
        for column in intro_table.columns:
            profile[column] = str(intro_row[column])

    key_stats_table = get_table(report, "Key Statistics")
    key_stats_row = (
        key_stats_table.iloc[0] if not key_stats_table.empty else pd.Series(dtype=object)
    )
    return profile, key_stats_row


def render_overview_tab(
    report: ParsedIBKRReport,
    key_stats_row: pd.Series,
    base_currency: str,
    analysis_years: float,
    performance_measure: str,
) -> None:
    ending_nav = parse_number(key_stats_row.get("EndingNAV"))
    cumulative_return = parse_number(key_stats_row.get("CumulativeReturn"))
    annualized_return = annualize_return(cumulative_return, analysis_years)
    mtm = parse_number(key_stats_row.get("MTM"))
    deposits = parse_number(key_stats_row.get("Deposits & Withdrawals"))
    dividends = parse_number(key_stats_row.get("Dividends"))
    interest = parse_number(key_stats_row.get("Interest"))
    fees = parse_number(key_stats_row.get("Fees & Commissions"))
    method_label, method_tip = return_method_label_and_tooltip(performance_measure)

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Ending NAV", format_money(ending_nav, base_currency))
    metric_col_2.metric(
        "Cumulative Return",
        format_pct(cumulative_return),
        help=method_tip,
    )
    metric_col_3.metric(
        "Annualized Return",
        format_pct(annualized_return),
        help=method_tip,
    )

    st.caption(f"Return method: {method_label}")

    metric_col_4, metric_col_5, metric_col_6, metric_col_7 = st.columns(4)
    metric_col_4.metric("MTM", format_money(mtm, base_currency))
    metric_col_5.metric("Net Deposits", format_money(deposits, base_currency))
    metric_col_6.metric(
        "Dividends + Interest", format_money(dividends + interest, base_currency)
    )
    metric_col_7.metric("Fees & Commissions", format_money(fees, base_currency))

    nav_table = get_table(
        report, "Allocation by Asset Class", required_columns=["Date", "NAV"]
    )
    if not nav_table.empty:
        nav_table["DateParsed"] = nav_table["Date"].map(parse_report_date)
        nav_table["NAV"] = to_numeric(nav_table["NAV"])
        nav_table["Equities"] = to_numeric(
            nav_table.get("Equities", pd.Series(dtype=float))
        )
        nav_table["Cash"] = to_numeric(nav_table.get("Cash", pd.Series(dtype=float)))
        nav_table = nav_table.dropna(subset=["DateParsed", "NAV"]).sort_values(
            "DateParsed"
        )

    chart_col_1, chart_col_2 = st.columns((1.4, 1.0))

    with chart_col_1:
        if nav_table.empty:
            st.info("NAV history is not available in this report.")
        else:
            nav_fig = go.Figure()
            nav_fig.add_trace(
                go.Scatter(
                    x=nav_table["DateParsed"],
                    y=nav_table["NAV"],
                    mode="lines",
                    fill="tozeroy",
                    line={"color": "#28d5b5", "width": 2.8},
                    name="NAV",
                )
            )
            nav_fig.update_layout(
                title="Portfolio NAV Over Time",
                template=PLOTLY_TEMPLATE,
                height=360,
                margin={"l": 12, "r": 12, "t": 48, "b": 8},
                xaxis_title="Date",
                yaxis_title=f"NAV ({base_currency})" if base_currency else "NAV",
            )
            st.plotly_chart(nav_fig, use_container_width=True)

    with chart_col_2:
        beginning_nav = parse_number(key_stats_row.get("BeginningNAV"))
        other = parse_number(key_stats_row.get("Other"))
        waterfall_labels = [
            "Beginning NAV",
            "Deposits",
            "MTM",
            "Dividends",
            "Interest",
            "Fees",
            "Other",
            "Ending NAV",
        ]
        waterfall_measure = [
            "absolute",
            "relative",
            "relative",
            "relative",
            "relative",
            "relative",
            "relative",
            "total",
        ]
        waterfall_values = [
            beginning_nav,
            deposits,
            mtm,
            dividends,
            interest,
            fees,
            other,
            0,
        ]

        nav_bridge = go.Figure(
            go.Waterfall(
                x=waterfall_labels,
                y=waterfall_values,
                measure=waterfall_measure,
                connector={"line": {"color": "rgba(180,194,220,0.4)"}},
                increasing={"marker": {"color": "#28d5b5"}},
                decreasing={"marker": {"color": "#ff5f8f"}},
                totals={"marker": {"color": "#5ca3ff"}},
            )
        )
        nav_bridge.update_layout(
            title="NAV Change Bridge",
            template=PLOTLY_TEMPLATE,
            height=360,
            margin={"l": 8, "r": 8, "t": 48, "b": 8},
            yaxis_title=f"Amount ({base_currency})" if base_currency else "Amount",
        )
        st.plotly_chart(nav_bridge, use_container_width=True)

    positions = get_table(
        report,
        "Open Position Summary",
        required_columns=[
            "Date",
            "Symbol",
            "Description",
            "Value",
            "UnrealizedP&L",
            "FinancialInstrument",
        ],
    )
    if positions.empty:
        st.info("Open position details are not available in this report.")
        return

    positions["Value"] = to_numeric(positions["Value"])
    positions["UnrealizedP&L"] = to_numeric(positions["UnrealizedP&L"])
    positions = sanitize_total_rows(positions, "Date")
    positions = sanitize_total_rows(positions, "Symbol", drop_blank=True)
    positions = positions.dropna(subset=["Value"])

    holdings = positions[positions["FinancialInstrument"].str.lower() != "cash"].copy()
    top_holdings = holdings.nlargest(8, "Value")

    if top_holdings.empty:
        st.info("No non-cash holdings were found in Open Position Summary.")
        return

    top_fig = px.bar(
        top_holdings.sort_values("Value"),
        x="Value",
        y="Symbol",
        color="UnrealizedP&L",
        orientation="h",
        color_continuous_scale=["#ff5f8f", "#5ca3ff", "#28d5b5"],
        template=PLOTLY_TEMPLATE,
        title="Top Holdings by Market Value",
        labels={
            "Value": f"Value ({base_currency})" if base_currency else "Value",
            "Symbol": "",
        },
    )
    top_fig.update_layout(
        height=390,
        margin={"l": 8, "r": 8, "t": 46, "b": 8},
        coloraxis_showscale=False,
    )
    st.plotly_chart(top_fig, use_container_width=True)


def render_performance_tab(
    report: ParsedIBKRReport, account_hint: str, performance_measure: str
) -> None:
    time_table = get_table(
        report,
        "Time Period Benchmark Comparison",
        required_columns=["Date", "BM1", "BM1Return", "BM2", "BM2Return", "BM3", "BM3Return"],
    )
    cumulative_table = get_table(
        report,
        "Cumulative Benchmark Comparison",
        required_columns=["Date", "BM1", "BM1Return", "BM2", "BM2Return", "BM3", "BM3Return"],
    )

    periodic_returns_long = build_benchmark_long(time_table)
    cumulative_returns_long = build_benchmark_long(cumulative_table)
    method_label, method_tip = return_method_label_and_tooltip(performance_measure)
    benchmark_names = []
    for benchmark_col in ("BM1", "BM2", "BM3"):
        if benchmark_col in time_table.columns:
            values = time_table[benchmark_col].replace("", np.nan).dropna()
            if not values.empty:
                benchmark_names.append(str(values.iloc[0]))

    portfolio_series_name = ""
    if not periodic_returns_long.empty:
        candidate_series = [
            series
            for series in periodic_returns_long["Series"].unique()
            if series not in benchmark_names
        ]
        portfolio_series_name = (
            account_hint
            if account_hint and account_hint in candidate_series
            else (
                candidate_series[0]
                if candidate_series
                else periodic_returns_long["Series"].iloc[0]
            )
        )

    performance_chart_col, drawdown_chart_col = st.columns((1.3, 1.0))

    with performance_chart_col:
        if periodic_returns_long.empty:
            st.info("Time period benchmark comparison was not found.")
        else:
            perf_fig = px.line(
                periodic_returns_long,
                x="Date",
                y="Return",
                color="Series",
                markers=True,
                template=PLOTLY_TEMPLATE,
                title="Periodic Return Comparison",
                color_discrete_sequence=CHART_COLORS,
            )
            perf_fig.update_layout(
                height=360,
                margin={"l": 12, "r": 12, "t": 48, "b": 8},
                yaxis_title="Return (%)",
                xaxis_title="Date",
                legend_title_text="Series",
            )
            st.plotly_chart(perf_fig, use_container_width=True)

    with drawdown_chart_col:
        if periodic_returns_long.empty:
            st.info("Drawdown chart requires periodic account returns.")
        else:
            portfolio_returns = periodic_returns_long.loc[
                periodic_returns_long["Series"] == portfolio_series_name
            ].sort_values("Date")

            if portfolio_returns.empty:
                st.info("Unable to identify the portfolio return series.")
            else:
                growth = (1 + (portfolio_returns["Return"] / 100.0)).cumprod()
                drawdown = ((growth / growth.cummax()) - 1) * 100
                drawdown_fig = go.Figure()
                drawdown_fig.add_trace(
                    go.Scatter(
                        x=portfolio_returns["Date"],
                        y=drawdown,
                        fill="tozeroy",
                        line={"color": "#ff5f8f", "width": 2.3},
                        name="Drawdown",
                    )
                )
                drawdown_fig.update_layout(
                    title=f"Drawdown ({portfolio_series_name})",
                    template=PLOTLY_TEMPLATE,
                    height=360,
                    margin={"l": 12, "r": 12, "t": 48, "b": 8},
                    yaxis_title="Drawdown (%)",
                    xaxis_title="Date",
                )
                st.plotly_chart(drawdown_fig, use_container_width=True)

    if not cumulative_returns_long.empty:
        st.markdown(
            (
                "<div class='method-tip'>Cumulative Return Comparison "
                f"<span class='hint-icon' title='{html.escape(method_tip)}'>i</span> "
                f"<span class='hint-value'>{html.escape(method_label)}</span></div>"
            ),
            unsafe_allow_html=True,
        )
        cumulative_fig = px.line(
            cumulative_returns_long,
            x="Date",
            y="Return",
            color="Series",
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=CHART_COLORS,
        )
        cumulative_fig.update_layout(
            height=360,
            margin={"l": 12, "r": 12, "t": 16, "b": 8},
            yaxis_title="Cumulative Return (%)",
            xaxis_title="Date",
            legend_title_text="Series",
        )
        st.plotly_chart(cumulative_fig, use_container_width=True)

        annualized_rows: list[dict[str, float | str]] = []
        for series_name, series_df in cumulative_returns_long.groupby("Series"):
            series_df = series_df.sort_values("Date")
            if series_df.empty:
                continue
            start_date = series_df["Date"].iloc[0]
            end_date = series_df["Date"].iloc[-1]
            years = period_years(start_date, end_date)
            cumulative_return = series_df["Return"].iloc[-1]
            annualized = annualize_return(cumulative_return, years)
            if pd.notna(annualized):
                annualized_rows.append(
                    {"Series": series_name, "AnnualizedReturn": annualized}
                )

        annualized_df = pd.DataFrame(annualized_rows)
        if not annualized_df.empty:
            annualized_df = annualized_df.sort_values("AnnualizedReturn", ascending=False)
        if not annualized_df.empty:
            st.markdown(
                (
                    "<div class='method-tip'>Annualized Return vs Benchmarks "
                    f"<span class='hint-icon' title='{html.escape(method_tip)}'>i</span> "
                    f"<span class='hint-value'>{html.escape(method_label)}</span></div>"
                ),
                unsafe_allow_html=True,
            )
            y_max = float(annualized_df["AnnualizedReturn"].max())
            y_min = float(annualized_df["AnnualizedReturn"].min())
            span = max(y_max - y_min, 1.0)
            upper_padding = max(span * 0.16, 1.2)
            lower_padding = max(span * 0.06, 0.6)
            y_start = min(0.0, y_min - lower_padding)
            y_end = y_max + upper_padding

            annualized_fig = px.bar(
                annualized_df,
                x="Series",
                y="AnnualizedReturn",
                color="Series",
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=CHART_COLORS,
                labels={"AnnualizedReturn": "Annualized return (%)", "Series": ""},
                text=annualized_df["AnnualizedReturn"].map(lambda value: f"{value:.2f}%"),
            )
            annualized_fig.update_layout(
                height=360,
                margin={"l": 12, "r": 12, "t": 22, "b": 8},
                showlegend=False,
                yaxis_range=[y_start, y_end],
            )
            annualized_fig.update_traces(
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
            )
            st.plotly_chart(annualized_fig, use_container_width=True)
    else:
        st.info("Cumulative benchmark comparison was not found.")

    if not periodic_returns_long.empty:
        portfolio_rows = periodic_returns_long.loc[
            periodic_returns_long["Series"] == portfolio_series_name
        ].copy()
        if portfolio_rows.empty:
            portfolio_rows = periodic_returns_long.copy()

        portfolio_rows = portfolio_rows.sort_values("Date")
        portfolio_rows["Year"] = portfolio_rows["Date"].dt.year.astype(str)
        portfolio_rows["MonthNumber"] = portfolio_rows["Date"].dt.month
        month_labels = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }
        portfolio_rows["Month"] = portfolio_rows["MonthNumber"].map(month_labels)
        heatmap_data = portfolio_rows.pivot_table(
            index="Year", columns="MonthNumber", values="Return", aggfunc="mean"
        )
        heatmap_data = heatmap_data.reindex(columns=list(month_labels.keys()))
        heatmap_data.columns = [month_labels[column] for column in heatmap_data.columns]

        if not heatmap_data.empty:
            heatmap_fig = px.imshow(
                heatmap_data,
                labels={"x": "Month", "y": "Year", "color": "Return (%)"},
                title="Portfolio Monthly Return Heatmap",
                color_continuous_scale=[
                    [0.0, "#8a2047"],
                    [0.45, "#1b3b63"],
                    [0.5, "#203f68"],
                    [0.55, "#1f6e73"],
                    [1.0, "#28d5b5"],
                ],
                aspect="auto",
                template=PLOTLY_TEMPLATE,
            )
            heatmap_fig.update_layout(
                height=320, margin={"l": 12, "r": 12, "t": 48, "b": 8}
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)

    symbol_perf = get_table(
        report,
        "Performance by Symbol",
        required_columns=["Symbol", "Description", "Contribution", "Return", "AvgWeight"],
    )
    if symbol_perf.empty:
        st.info("Performance by Symbol was not found in this report.")
        return

    symbol_perf = sanitize_total_rows(symbol_perf, "Symbol")
    symbol_perf = sanitize_total_rows(symbol_perf, "Description")
    symbol_perf["Contribution"] = to_numeric(symbol_perf["Contribution"])
    symbol_perf["Return"] = to_numeric(symbol_perf["Return"])
    symbol_perf["AvgWeight"] = to_numeric(symbol_perf["AvgWeight"])
    symbol_perf = symbol_perf.dropna(subset=["Contribution"])

    if symbol_perf.empty:
        st.info("No symbol-level contribution records were detected.")
        return

    best = symbol_perf.nlargest(7, "Contribution").sort_values("Contribution")
    worst = symbol_perf.nsmallest(7, "Contribution").sort_values("Contribution")

    winners_col, losers_col = st.columns(2)
    with winners_col:
        winners_fig = px.bar(
            best,
            x="Contribution",
            y="Symbol",
            orientation="h",
            color="Contribution",
            color_continuous_scale=["#1f6e73", "#28d5b5"],
            template=PLOTLY_TEMPLATE,
            title="Top Contributors",
            labels={"Contribution": "Contribution (%)", "Symbol": ""},
        )
        winners_fig.update_layout(
            height=340,
            margin={"l": 8, "r": 8, "t": 44, "b": 8},
            coloraxis_showscale=False,
        )
        st.plotly_chart(winners_fig, use_container_width=True)

    with losers_col:
        losers_fig = px.bar(
            worst,
            x="Contribution",
            y="Symbol",
            orientation="h",
            color="Contribution",
            color_continuous_scale=["#ff5f8f", "#8a2047"],
            template=PLOTLY_TEMPLATE,
            title="Bottom Contributors",
            labels={"Contribution": "Contribution (%)", "Symbol": ""},
        )
        losers_fig.update_layout(
            height=340,
            margin={"l": 8, "r": 8, "t": 44, "b": 8},
            coloraxis_showscale=False,
        )
        st.plotly_chart(losers_fig, use_container_width=True)


def render_holdings_tab(report: ParsedIBKRReport, base_currency: str) -> None:
    positions = get_table(
        report,
        "Open Position Summary",
        required_columns=[
            "Date",
            "FinancialInstrument",
            "Currency",
            "Symbol",
            "Description",
            "Sector",
            "Quantity",
            "ClosePrice",
            "Value",
            "Cost Basis",
            "UnrealizedP&L",
        ],
    )
    if positions.empty:
        st.info("Open Position Summary is not available.")
        return

    for numeric_col in ("Quantity", "ClosePrice", "Value", "Cost Basis", "UnrealizedP&L"):
        positions[numeric_col] = to_numeric(positions[numeric_col])

    positions = sanitize_total_rows(positions, "Date")
    positions = sanitize_total_rows(positions, "Symbol", drop_blank=True)
    positions = positions.dropna(subset=["Value"])

    if positions.empty:
        st.info("No open positions were found.")
        return

    cash_positions = positions[positions["FinancialInstrument"].str.lower() == "cash"].copy()
    non_cash_positions = positions[
        positions["FinancialInstrument"].str.lower() != "cash"
    ].copy()

    total_value = positions["Value"].sum()
    cash_value = cash_positions["Value"].sum()
    holdings_value = non_cash_positions["Value"].sum()
    total_unrealized = positions["UnrealizedP&L"].sum()

    summary_col_1, summary_col_2, summary_col_3, summary_col_4 = st.columns(4)
    summary_col_1.metric("Total Market Value", format_money(total_value, base_currency))
    summary_col_2.metric("Holdings Value", format_money(holdings_value, base_currency))
    summary_col_3.metric("Cash Value", format_money(cash_value, base_currency))
    summary_col_4.metric("Unrealized P&L", format_money(total_unrealized, base_currency))

    composition_col_1, composition_col_2 = st.columns(2)
    with composition_col_1:
        sector_chart_data = (
            non_cash_positions.groupby("Sector", dropna=False)["Value"]
            .sum()
            .reset_index()
            .sort_values("Value", ascending=False)
        )
        if sector_chart_data.empty:
            st.info("No sector allocation records detected.")
        else:
            sector_fig = px.pie(
                sector_chart_data,
                names="Sector",
                values="Value",
                title="Holdings by Sector",
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=CHART_COLORS,
                hole=0.52,
            )
            sector_fig.update_layout(
                height=360, margin={"l": 8, "r": 8, "t": 48, "b": 8}
            )
            st.plotly_chart(sector_fig, use_container_width=True)

    with composition_col_2:
        currency_chart_data = (
            positions.groupby("Currency", dropna=False)["Value"]
            .sum()
            .reset_index()
            .sort_values("Value", ascending=False)
        )
        currency_fig = px.pie(
            currency_chart_data,
            names="Currency",
            values="Value",
            title="Exposure by Currency",
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=CHART_COLORS,
            hole=0.52,
        )
        currency_fig.update_layout(height=360, margin={"l": 8, "r": 8, "t": 48, "b": 8})
        st.plotly_chart(currency_fig, use_container_width=True)

    top_positions = non_cash_positions.nlargest(12, "Value").copy()
    display_columns = [
        "Symbol",
        "Description",
        "Sector",
        "Currency",
        "Quantity",
        "Value",
        "Cost Basis",
        "UnrealizedP&L",
    ]
    if not top_positions.empty:
        display_table = top_positions[display_columns].copy()
        for value_col in ("Value", "Cost Basis", "UnrealizedP&L"):
            display_table[value_col] = display_table[value_col].map(
                lambda value: format_money(value, base_currency)
            )
        display_table["Quantity"] = display_table["Quantity"].map(
            lambda value: "-" if pd.isna(value) else f"{value:,.4f}"
        )
        st.subheader("Largest Holdings")
        st.dataframe(display_table, use_container_width=True, hide_index=True)

    trade_summary = get_table(
        report,
        "Trade Summary",
        required_columns=[
            "Symbol",
            "Description",
            "Proceeds Bought in Base",
            "Proceeds Sold in Base",
            "Financial Instrument",
        ],
    )
    if trade_summary.empty:
        st.info("Trade Summary was not found.")
        return

    trade_summary = sanitize_total_rows(trade_summary, "Symbol", drop_blank=True)
    trade_summary["Proceeds Bought in Base"] = to_numeric(trade_summary["Proceeds Bought in Base"])
    trade_summary["Proceeds Sold in Base"] = to_numeric(trade_summary["Proceeds Sold in Base"])
    trade_summary["NetInvested"] = -(
        trade_summary["Proceeds Bought in Base"] + trade_summary["Proceeds Sold in Base"]
    )
    trade_summary = trade_summary.dropna(subset=["NetInvested"])
    trade_summary = trade_summary[
        trade_summary["Financial Instrument"].str.lower() != "forex"
    ]

    if not trade_summary.empty:
        traded_fig = px.bar(
            trade_summary.sort_values("NetInvested"),
            x="NetInvested",
            y="Symbol",
            orientation="h",
            color="NetInvested",
            color_continuous_scale=["#5ca3ff", "#28d5b5"],
            template=PLOTLY_TEMPLATE,
            title="Net Capital Deployed by Symbol (Base Currency)",
            labels={"NetInvested": "Net Invested", "Symbol": ""},
        )
        traded_fig.update_layout(
            height=340,
            margin={"l": 8, "r": 8, "t": 44, "b": 8},
            coloraxis_showscale=False,
        )
        st.plotly_chart(traded_fig, use_container_width=True)


def render_concentration_tab(report: ParsedIBKRReport) -> None:
    concentration = get_table(
        report,
        "Concentration",
        required_columns=[
            "SubSection",
            "Symbol",
            "Description",
            "LongParsedWeight",
            "ShortParsedWeight",
            "NetParsedWeight",
        ],
    )
    if concentration.empty:
        st.info("Concentration table was not found in this report.")
        return

    concentration = concentration.copy()
    concentration["SubSection"] = concentration["SubSection"].astype(str).str.strip()
    concentration["Symbol"] = concentration["Symbol"].astype(str).str.strip()
    concentration["Description"] = concentration["Description"].astype(str).str.strip()
    concentration = concentration[
        concentration["SubSection"].str.lower() == "holdings"
    ].copy()

    long_weight = to_numeric(concentration["LongParsedWeight"])
    short_weight = to_numeric(concentration["ShortParsedWeight"])
    concentration["NetWeight"] = to_numeric(concentration["NetParsedWeight"])

    # Keep top-level underlying stock rows and drop ETF decomposition rows.
    concentration = concentration[(long_weight.notna()) | (short_weight.notna())]
    concentration = sanitize_total_rows(concentration, "Symbol", drop_blank=True)
    concentration = sanitize_total_rows(concentration, "Description")
    concentration = concentration.dropna(subset=["NetWeight"])
    concentration = concentration[concentration["NetWeight"] > 0]

    if concentration.empty:
        st.info("No positive underlying concentration weights were detected.")
        return

    stock_exposure = (
        concentration.groupby(["Symbol", "Description"], as_index=False)["NetWeight"]
        .sum()
        .sort_values("NetWeight", ascending=False)
    )
    total_weight = stock_exposure["NetWeight"].sum()
    max_top = max(1, min(20, len(stock_exposure)))
    default_top = min(12, max_top)
    top_n = st.slider(
        "Top stocks in donut",
        min_value=1,
        max_value=max_top,
        value=default_top,
        step=1,
    )

    top = stock_exposure.head(top_n).copy()
    others_weight = total_weight - top["NetWeight"].sum()
    top_coverage = (
        (top["NetWeight"].sum() / total_weight) * 100 if total_weight > 0 else np.nan
    )
    donut_data = top[["Symbol", "NetWeight"]].rename(
        columns={"Symbol": "Bucket", "NetWeight": "Weight"}
    )
    if others_weight > 0.00001:
        donut_data = pd.concat(
            [donut_data, pd.DataFrame([{"Bucket": "Others", "Weight": others_weight}])],
            ignore_index=True,
        )

    metrics_col_1, metrics_col_2, metrics_col_3 = st.columns(3)
    top_row = stock_exposure.iloc[0]
    metrics_col_1.metric("Top Underlying", str(top_row["Symbol"]))
    metrics_col_2.metric("Top Weight", format_pct(top_row["NetWeight"]))
    metrics_col_3.metric("Top-N Coverage", format_pct(top_coverage))

    holdings_count = len(stock_exposure)
    milestone_metrics = []
    for point in (5, 10, 20, 50):
        top_count = min(point, holdings_count)
        if not milestone_metrics or milestone_metrics[-1][0] != top_count:
            label = f"Top {top_count}"
            if top_count == holdings_count and holdings_count < 50:
                label = f"Top {top_count} (All)"
            covered = stock_exposure.head(top_count)["NetWeight"].sum()
            coverage_pct = (covered / total_weight) * 100 if total_weight > 0 else np.nan
            milestone_metrics.append((label, coverage_pct))

    milestone_columns = st.columns(len(milestone_metrics))
    for metric_col, (label, value) in zip(milestone_columns, milestone_metrics):
        metric_col.metric(label, format_pct(value))

    concentration_col_1, concentration_col_2 = st.columns((1.15, 1.0))
    with concentration_col_1:
        donut_fig = px.pie(
            donut_data,
            names="Bucket",
            values="Weight",
            hole=0.58,
            template=PLOTLY_TEMPLATE,
            title="Underlying Stock Concentration",
            color_discrete_sequence=CHART_COLORS,
        )
        donut_fig.update_traces(
            textposition="inside",
            texttemplate="%{label}<br>%{percent:.2%}",
            hovertemplate="%{label}<br>%{value:.2f}%<extra></extra>",
        )
        donut_fig.update_layout(height=390, margin={"l": 8, "r": 8, "t": 48, "b": 8})
        st.plotly_chart(donut_fig, use_container_width=True)

    with concentration_col_2:
        focus_limit = min(50, holdings_count)
        milestone_candidates = [5, 10, 20, 30, 40, 50]
        checkpoints = [point for point in milestone_candidates if point <= focus_limit]
        if not checkpoints:
            checkpoints = [focus_limit]
        elif checkpoints[-1] != focus_limit:
            checkpoints.append(focus_limit)

        coverage_rows: list[dict[str, float | str]] = []
        for checkpoint in checkpoints:
            covered_weight = stock_exposure.head(checkpoint)["NetWeight"].sum()
            coverage_pct = (covered_weight / total_weight) * 100 if total_weight > 0 else np.nan
            label = f"Top {checkpoint}"
            if checkpoint == holdings_count and holdings_count < 50:
                label = f"Top {checkpoint} (All)"
            coverage_rows.append({"Bucket": label, "CoveragePct": coverage_pct})

        coverage_df = pd.DataFrame(coverage_rows)
        coverage_fig = px.bar(
            coverage_df,
            x="Bucket",
            y="CoveragePct",
            template=PLOTLY_TEMPLATE,
            title="Cumulative Coverage Milestones (Top 50 Max)",
            color="CoveragePct",
            color_continuous_scale=["#1f6e73", "#28d5b5"],
            labels={"CoveragePct": "Cumulative coverage (%)", "Bucket": ""},
            text=coverage_df["CoveragePct"].map(lambda value: f"{value:.2f}%"),
        )
        coverage_fig.update_layout(
            height=390,
            margin={"l": 8, "r": 8, "t": 48, "b": 8},
            coloraxis_showscale=False,
            yaxis_range=[0, 100],
        )
        coverage_fig.update_traces(
            textposition="outside",
            hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
        )
        st.plotly_chart(coverage_fig, use_container_width=True)

    stock_table = stock_exposure.head(50).copy()
    stock_table["CumulativeCoverage"] = (
        stock_table["NetWeight"].cumsum() / total_weight * 100 if total_weight > 0 else np.nan
    )
    stock_table["NetWeight"] = stock_table["NetWeight"].map(format_pct)
    stock_table["CumulativeCoverage"] = stock_table["CumulativeCoverage"].map(format_pct)
    st.subheader("Underlying Stock Weights")
    st.dataframe(
        stock_table[["Symbol", "Description", "NetWeight", "CumulativeCoverage"]],
        use_container_width=True,
        hide_index=True,
    )


def render_cashflow_income_tab(report: ParsedIBKRReport, base_currency: str) -> None:
    cashflows = get_table(
        report,
        "Deposits And Withdrawals",
        required_columns=["Date", "Type", "Description", "Amount"],
    )
    dividends = get_table(
        report, "Dividends", required_columns=["PayDate", "Symbol", "Amount"]
    )
    fees = get_table(report, "Fee Summary", required_columns=["Date", "Amount"])
    interest = get_table(report, "Interest Details", required_columns=["Date", "Amount"])
    projected_income = get_table(
        report,
        "Projected Income",
        required_columns=["Estimated Annual Income", "Estimated 2026 Remaining Income"],
    )

    if not cashflows.empty:
        cashflows["DateParsed"] = cashflows["Date"].map(parse_report_date)
        cashflows["Amount"] = to_numeric(cashflows["Amount"])
        cashflows = cashflows.dropna(subset=["DateParsed", "Amount"])

    if not dividends.empty:
        dividends["DateParsed"] = dividends["PayDate"].map(parse_report_date)
        dividends["Amount"] = to_numeric(dividends["Amount"])
        dividends = dividends.dropna(subset=["DateParsed", "Amount"])

    if not fees.empty:
        fees["DateParsed"] = fees["Date"].map(parse_report_date)
        fees["Amount"] = to_numeric(fees["Amount"])
        fees = fees.dropna(subset=["DateParsed", "Amount"])

    if not interest.empty:
        interest["DateParsed"] = interest["Date"].map(parse_report_date)
        interest["Amount"] = to_numeric(interest["Amount"])
        interest = interest.dropna(subset=["DateParsed", "Amount"])

    deposits_total = (
        cashflows.loc[cashflows["Amount"] > 0, "Amount"].sum() if not cashflows.empty else np.nan
    )
    withdrawals_total = (
        cashflows.loc[cashflows["Amount"] < 0, "Amount"].sum() if not cashflows.empty else np.nan
    )
    dividend_total = dividends["Amount"].sum() if not dividends.empty else np.nan
    interest_total = interest["Amount"].sum() if not interest.empty else np.nan
    fee_total = fees["Amount"].sum() if not fees.empty else np.nan

    cf_col_1, cf_col_2, cf_col_3, cf_col_4 = st.columns(4)
    cf_col_1.metric("Total Deposits", format_money(deposits_total, base_currency))
    cf_col_2.metric("Total Withdrawals", format_money(withdrawals_total, base_currency))
    cf_col_3.metric("Dividends Received", format_money(dividend_total, base_currency))
    cf_col_4.metric(
        "Net Interest + Fees",
        format_money((interest_total if pd.notna(interest_total) else 0) + (fee_total if pd.notna(fee_total) else 0), base_currency),
    )

    chart_col_1, chart_col_2 = st.columns(2)

    with chart_col_1:
        if cashflows.empty:
            st.info("No deposit/withdrawal records found.")
        else:
            monthly_cashflows = (
                cashflows.assign(Month=cashflows["DateParsed"].dt.to_period("M").dt.to_timestamp())
                .groupby("Month", as_index=False)["Amount"]
                .sum()
            )
            cf_fig = px.bar(
                monthly_cashflows,
                x="Month",
                y="Amount",
                color="Amount",
                color_continuous_scale=["#ff5f8f", "#1f6e73", "#28d5b5"],
                template=PLOTLY_TEMPLATE,
                title="Net Deposits / Withdrawals by Month",
                labels={
                    "Amount": f"Amount ({base_currency})" if base_currency else "Amount",
                    "Month": "Month",
                },
            )
            cf_fig.update_layout(
                height=340,
                margin={"l": 12, "r": 12, "t": 46, "b": 8},
                coloraxis_showscale=False,
            )
            st.plotly_chart(cf_fig, use_container_width=True)

    with chart_col_2:
        income_components = pd.DataFrame(
            {
                "Category": ["Dividends", "Interest", "Fees"],
                "Amount": [dividend_total, interest_total, fee_total],
            }
        )
        income_components["Amount"] = income_components["Amount"].fillna(0.0)
        income_fig = px.bar(
            income_components,
            x="Category",
            y="Amount",
            color="Category",
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=CHART_COLORS,
            title="Income and Cost Components",
            labels={"Amount": f"Amount ({base_currency})" if base_currency else "Amount"},
        )
        income_fig.update_layout(
            height=340,
            margin={"l": 12, "r": 12, "t": 46, "b": 8},
            showlegend=False,
        )
        st.plotly_chart(income_fig, use_container_width=True)

    if not dividends.empty:
        dividends = dividends.sort_values("DateParsed")
        dividends["CumulativeDividends"] = dividends["Amount"].cumsum()
        dividend_fig = px.area(
            dividends,
            x="DateParsed",
            y="CumulativeDividends",
            template=PLOTLY_TEMPLATE,
            title="Cumulative Dividends",
            labels={
                "DateParsed": "Pay Date",
                "CumulativeDividends": f"Cumulative ({base_currency})"
                if base_currency
                else "Cumulative",
            },
        )
        dividend_fig.update_traces(line={"color": "#28d5b5", "width": 2.2})
        dividend_fig.update_layout(height=320, margin={"l": 12, "r": 12, "t": 46, "b": 8})
        st.plotly_chart(dividend_fig, use_container_width=True)

    if not projected_income.empty:
        projected_income_total = projected_income.copy()
        if "Symbol" in projected_income_total.columns:
            preferred_rows = projected_income_total.loc[
                projected_income_total["Symbol"].astype(str).str.strip().str.lower() == "total"
            ]
            projected_row = (
                preferred_rows.iloc[0]
                if not preferred_rows.empty
                else projected_income_total.iloc[-1]
            )
        else:
            projected_row = projected_income_total.iloc[-1]

        annual_income = parse_number(projected_row.get("Estimated Annual Income"))
        remaining_income = parse_number(projected_row.get("Estimated 2026 Remaining Income"))
        yield_value = parse_number(projected_row.get("Current Yield %"))

        project_col_1, project_col_2, project_col_3 = st.columns(3)
        project_col_1.metric(
            "Projected Annual Income", format_money(annual_income, base_currency)
        )
        project_col_2.metric(
            "Remaining 2026 Income", format_money(remaining_income, base_currency)
        )
        project_col_3.metric("Current Yield", format_pct(yield_value))


def render_risk_esg_tab(report: ParsedIBKRReport) -> None:
    risk_absolute = get_table(
        report,
        "Risk Measures Benchmark Comparison",
        required_columns=[
            "Risk Measure",
            "BM1",
            "BM1 Value",
            "BM2",
            "BM2 Value",
            "BM3",
            "BM3 Value",
            "Account",
            "Account Value",
        ],
    )
    risk_relative = get_table(
        report,
        "Risk Measures Benchmark Comparison",
        required_columns=[
            "Risk Measure Relative to Benchmark",
            "BM1",
            "BM1 Value",
            "BM2",
            "BM2 Value",
            "BM3",
            "BM3 Value",
        ],
    )

    if not risk_absolute.empty:
        risk_absolute = risk_absolute.copy()
        risk_absolute["Metric"] = (
            risk_absolute["Risk Measure"]
            .astype(str)
            .str.replace(":", "", regex=False)
            .str.strip()
        )

        benchmark_names = {}
        for key in ("BM1", "BM2", "BM3", "Account"):
            values = risk_absolute[key].replace("", np.nan).dropna()
            benchmark_names[key] = str(values.iloc[0]) if not values.empty else key

        metric_subset = [
            "Sharpe Ratio",
            "Sortino Ratio",
            "Calmar Ratio",
            "Standard Deviation",
            "Max Drawdown",
            "1 Month VaR 95",
        ]
        chart_rows = risk_absolute[risk_absolute["Metric"].isin(metric_subset)].copy()

        melted_rows: list[dict[str, object]] = []
        for _, row in chart_rows.iterrows():
            for benchmark, value_column in (
                ("BM1", "BM1 Value"),
                ("BM2", "BM2 Value"),
                ("BM3", "BM3 Value"),
                ("Account", "Account Value"),
            ):
                value = parse_number(row.get(value_column))
                if pd.notna(value):
                    melted_rows.append(
                        {
                            "Metric": row["Metric"],
                            "Series": benchmark_names[benchmark],
                            "Value": value,
                        }
                    )

        if melted_rows:
            risk_bar = px.bar(
                pd.DataFrame(melted_rows),
                x="Metric",
                y="Value",
                color="Series",
                barmode="group",
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=CHART_COLORS,
                title="Risk Measure Comparison",
            )
            risk_bar.update_layout(height=360, margin={"l": 12, "r": 12, "t": 48, "b": 8})
            st.plotly_chart(risk_bar, use_container_width=True)

        st.subheader("Absolute Risk Measures")
        st.dataframe(
            risk_absolute.drop(columns=["Metric"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Absolute risk measures section not found.")

    if not risk_relative.empty:
        st.subheader("Risk Measures Relative to Benchmarks")
        st.dataframe(risk_relative, use_container_width=True, hide_index=True)

    esg_summary = get_table(report, "ESG", required_columns=["SubSection", "Category", "Score"])
    esg_holdings = get_table(
        report,
        "ESG",
        required_columns=["SubSection", "Symbol", "Description", "Weight (%)", "ESG", "Combined"],
    )

    if esg_summary.empty and esg_holdings.empty:
        st.info("No ESG section found in this report.")
        return

    if not esg_summary.empty:
        esg_summary = esg_summary.copy()
        esg_summary["Score"] = to_numeric(esg_summary["Score"])
        esg_summary = esg_summary.dropna(subset=["Score"])
        if not esg_summary.empty:
            esg_fig = px.bar(
                esg_summary,
                x="Category",
                y="Score",
                color="Category",
                template=PLOTLY_TEMPLATE,
                title="Portfolio-Level ESG Snapshot",
                color_discrete_sequence=CHART_COLORS,
            )
            esg_fig.update_layout(
                height=320,
                margin={"l": 12, "r": 12, "t": 46, "b": 8},
                showlegend=False,
            )
            st.plotly_chart(esg_fig, use_container_width=True)

    if not esg_holdings.empty:
        esg_holdings = esg_holdings.copy()
        esg_holdings["Weight (%)"] = to_numeric(esg_holdings["Weight (%)"])
        esg_holdings["ESG"] = to_numeric(esg_holdings["ESG"])
        esg_holdings["Combined"] = to_numeric(esg_holdings["Combined"])
        esg_holdings = sanitize_total_rows(esg_holdings, "Symbol", drop_blank=True)
        esg_holdings = esg_holdings.dropna(subset=["Weight (%)", "ESG"])
        esg_holdings = esg_holdings.sort_values("Weight (%)", ascending=False).head(20)

        if not esg_holdings.empty:
            st.subheader("Top Weighted ESG Constituents")
            scatter_fig = px.scatter(
                esg_holdings,
                x="Weight (%)",
                y="ESG",
                size="Weight (%)",
                hover_name="Symbol",
                hover_data={"Description": True, "Combined": True},
                color="Combined",
                template=PLOTLY_TEMPLATE,
                title="Weight vs ESG Score",
                color_continuous_scale=["#ff5f8f", "#5ca3ff", "#28d5b5"],
            )
            scatter_fig.update_layout(height=340, margin={"l": 12, "r": 12, "t": 46, "b": 8})
            st.plotly_chart(scatter_fig, use_container_width=True)

            table_view = esg_holdings[
                ["Symbol", "Description", "Weight (%)", "ESG", "Combined"]
            ].copy()
            st.dataframe(table_view, use_container_width=True, hide_index=True)


def render_raw_tables_tab(report: ParsedIBKRReport) -> None:
    section_names = sorted(report.tables.keys())
    if not section_names:
        st.info("No tables were parsed from this file.")
        return

    section = st.selectbox("Report section", section_names, key="raw_section")
    tables = report.tables[section]
    table_options = []
    for table_index, table in enumerate(tables, start=1):
        preview_columns = ", ".join(table.columns[:4])
        label = f"Table {table_index} ({table.shape[0]} rows, {table.shape[1]} cols) - {preview_columns}"
        table_options.append(label)

    selected_label = st.selectbox("Table", table_options, key="raw_table")
    selected_index = table_options.index(selected_label)
    selected_table = tables[selected_index]

    st.dataframe(selected_table, use_container_width=True, height=480, hide_index=True)
    st.caption(f"Metadata rows in section: {len(report.metadata.get(section, []))}")


def streamlit_app() -> None:
    st.set_page_config(
        page_title="IBKR Portfolio Analyzer",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">IBKR Portfolio Analyzer</div>
            <p class="hero-sub">
                Upload an Interactive Brokers Portfolio Analyst CSV and get a full dashboard:
                performance, holdings, cashflows, risk, ESG, and benchmark comparison.
            </p>
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
        report = parse_ibkr_report(report_bytes)
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
        f"""
        <div class="panel">
            <b>{report_source}</b><br/>
            Account: <b>{account_name}</b> ({account_id})<br/>
            Base Currency: <b>{base_currency or "-"}</b><br/>
            Return Measure: <b>{performance_measure or "-"}</b><br/>
            Analysis Period: <b>{analysis_period or "-"}</b><br/>
            Period Length: <b>{period_length_display}</b><br/>
            Parsed Sections: <b>{len(report.tables)}</b>
        </div>
        """,
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
