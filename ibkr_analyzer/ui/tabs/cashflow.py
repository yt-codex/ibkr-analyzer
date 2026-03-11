import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ibkr_analyzer.report_utils import (
    ParsedIBKRReport,
    find_projected_remaining_income_column,
    format_money,
    format_pct,
    get_table,
    parse_number,
    parse_report_date,
    remaining_income_metric_label,
    to_numeric,
)
from ibkr_analyzer.ui.constants import CHART_COLORS, PLOTLY_TEMPLATE, get_chart_theme


def render_cashflow_income_tab(report: ParsedIBKRReport, base_currency: str) -> None:
    chart_theme = get_chart_theme()
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
        required_columns=["Estimated Annual Income"],
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
        format_money(
            (interest_total if pd.notna(interest_total) else 0)
            + (fee_total if pd.notna(fee_total) else 0),
            base_currency,
        ),
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
                color_continuous_scale=list(chart_theme["cashflow_scale"]),
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
        dividend_fig.update_traces(line={"color": CHART_COLORS[0], "width": 2.2})
        dividend_fig.update_layout(height=320, margin={"l": 12, "r": 12, "t": 46, "b": 8})
        st.plotly_chart(dividend_fig, use_container_width=True)

    if not projected_income.empty:
        projected_income_total = projected_income.copy()
        remaining_income_column = find_projected_remaining_income_column(
            projected_income_total.columns
        )
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
        remaining_income = (
            parse_number(projected_row.get(remaining_income_column))
            if remaining_income_column
            else np.nan
        )
        yield_value = parse_number(projected_row.get("Current Yield %"))
        remaining_income_label = remaining_income_metric_label(remaining_income_column)

        project_col_1, project_col_2, project_col_3 = st.columns(3)
        project_col_1.metric(
            "Projected Annual Income", format_money(annual_income, base_currency)
        )
        project_col_2.metric(
            remaining_income_label, format_money(remaining_income, base_currency)
        )
        project_col_3.metric("Current Yield", format_pct(yield_value))
