import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ibkr_analyzer.report_utils import (
    ParsedIBKRReport,
    annualize_return,
    format_money,
    format_pct,
    get_table,
    parse_number,
    parse_report_date,
    return_method_label_and_tooltip,
    sanitize_total_rows,
    to_numeric,
    value_or_zero,
)
from ibkr_analyzer.ui.constants import PLOTLY_TEMPLATE


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
        "Dividends + Interest",
        format_money(value_or_zero(dividends) + value_or_zero(interest), base_currency),
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
