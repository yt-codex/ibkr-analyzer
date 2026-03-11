import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ibkr_analyzer.report_utils import (
    ParsedIBKRReport,
    get_table,
    parse_number,
    sanitize_total_rows,
    to_numeric,
)
from ibkr_analyzer.ui.constants import CHART_COLORS, get_chart_theme, get_plotly_template


def render_risk_esg_tab(report: ParsedIBKRReport) -> None:
    chart_theme = get_chart_theme()
    plotly_template = get_plotly_template()
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
                template=plotly_template,
                color_discrete_sequence=CHART_COLORS,
                title="Risk Measure Comparison",
            )
            risk_bar.update_layout(
                height=360,
                margin={"l": 12, "r": 12, "t": 48, "b": 8},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor=str(chart_theme["plot_bg"]),
                font={"color": str(chart_theme["font_color"])},
                title={"x": 0, "xanchor": "left"},
            )
            risk_bar.update_xaxes(showgrid=False, zeroline=False)
            risk_bar.update_yaxes(showgrid=True, gridcolor=str(chart_theme["grid_color"]), zeroline=False)
            risk_bar.update_traces(marker_line={"width": 1.1, "color": str(chart_theme["marker_line"])})
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
                template=plotly_template,
                title="Portfolio-Level ESG Snapshot",
                color_discrete_sequence=CHART_COLORS,
            )
            esg_fig.update_layout(
                height=320,
                margin={"l": 12, "r": 12, "t": 46, "b": 8},
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor=str(chart_theme["plot_bg"]),
                font={"color": str(chart_theme["font_color"])},
                title={"x": 0, "xanchor": "left"},
            )
            esg_fig.update_xaxes(showgrid=False, zeroline=False)
            esg_fig.update_yaxes(showgrid=True, gridcolor=str(chart_theme["grid_color"]), zeroline=False)
            esg_fig.update_traces(marker_line={"width": 1.1, "color": str(chart_theme["marker_line"])})
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
                template=plotly_template,
                title="Weight vs ESG Score",
                color_continuous_scale=list(chart_theme["risk_scale"]),
            )
            scatter_fig.update_layout(
                height=340,
                margin={"l": 12, "r": 12, "t": 46, "b": 8},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor=str(chart_theme["plot_bg"]),
                font={"color": str(chart_theme["font_color"])},
                title={"x": 0, "xanchor": "left"},
            )
            scatter_fig.update_xaxes(showgrid=True, gridcolor=str(chart_theme["grid_color"]), zeroline=False)
            scatter_fig.update_yaxes(showgrid=True, gridcolor=str(chart_theme["grid_color"]), zeroline=False)
            scatter_fig.update_traces(marker_line={"width": 1.0, "color": str(chart_theme["marker_line"])})
            st.plotly_chart(scatter_fig, use_container_width=True)

            table_view = esg_holdings[
                ["Symbol", "Description", "Weight (%)", "ESG", "Combined"]
            ].copy()
            st.dataframe(table_view, use_container_width=True, hide_index=True)
