import pandas as pd
import plotly.express as px
import streamlit as st

from ibkr_analyzer.report_utils import (
    ParsedIBKRReport,
    format_money,
    get_table,
    sanitize_total_rows,
    to_numeric,
)
from ibkr_analyzer.ui.constants import CHART_COLORS, get_chart_theme, get_plotly_template
from ibkr_analyzer.ui.tables import render_dataframe


def render_holdings_tab(report: ParsedIBKRReport, base_currency: str) -> None:
    chart_theme = get_chart_theme()
    plotly_template = get_plotly_template()
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
                template=plotly_template,
                color_discrete_sequence=CHART_COLORS,
                hole=0.52,
            )
            sector_fig.update_layout(
                height=360,
                margin={"l": 8, "r": 8, "t": 48, "b": 8},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor=str(chart_theme["plot_bg"]),
                font={"color": str(chart_theme["font_color"])},
                title={"x": 0, "xanchor": "left"},
            )
            sector_fig.update_traces(marker_line={"width": 1.2, "color": str(chart_theme["marker_line"])})
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
            template=plotly_template,
            color_discrete_sequence=CHART_COLORS,
            hole=0.52,
        )
        currency_fig.update_layout(
            height=360,
            margin={"l": 8, "r": 8, "t": 48, "b": 8},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=str(chart_theme["plot_bg"]),
            font={"color": str(chart_theme["font_color"])},
            title={"x": 0, "xanchor": "left"},
        )
        currency_fig.update_traces(marker_line={"width": 1.2, "color": str(chart_theme["marker_line"])})
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
        render_dataframe(display_table, use_container_width=True, hide_index=True)

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
            color_continuous_scale=list(chart_theme["capital_scale"]),
            template=plotly_template,
            title="Net Capital Deployed by Symbol (Base Currency)",
            labels={"NetInvested": "Net Invested", "Symbol": ""},
        )
        traded_fig.update_layout(
            height=340,
            margin={"l": 8, "r": 8, "t": 44, "b": 8},
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=str(chart_theme["plot_bg"]),
            font={"color": str(chart_theme["font_color"])},
            title={"x": 0, "xanchor": "left"},
        )
        traded_fig.update_xaxes(showgrid=True, gridcolor=str(chart_theme["grid_color"]), zeroline=False)
        traded_fig.update_yaxes(showgrid=False, zeroline=False)
        traded_fig.update_traces(marker_line={"width": 1.1, "color": str(chart_theme["marker_line"])})
        st.plotly_chart(traded_fig, use_container_width=True)
