import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ibkr_analyzer.report_utils import (
    ParsedIBKRReport,
    format_pct,
    get_table,
    sanitize_total_rows,
    to_numeric,
)
from ibkr_analyzer.ui.constants import CHART_COLORS, get_chart_theme, get_plotly_template


def render_concentration_tab(report: ParsedIBKRReport) -> None:
    chart_theme = get_chart_theme()
    plotly_template = get_plotly_template()
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
            template=plotly_template,
            title="Underlying Stock Concentration",
            color_discrete_sequence=CHART_COLORS,
        )
        donut_fig.update_traces(
            textposition="inside",
            texttemplate="%{label}<br>%{percent:.2%}",
            hovertemplate="%{label}<br>%{value:.2f}%<extra></extra>",
        )
        donut_fig.update_layout(
            height=390,
            margin={"l": 8, "r": 8, "t": 48, "b": 8},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=str(chart_theme["plot_bg"]),
            font={"color": str(chart_theme["font_color"])},
            title={"x": 0, "xanchor": "left"},
        )
        donut_fig.update_traces(marker_line={"width": 1.2, "color": str(chart_theme["marker_line"])})
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
            template=plotly_template,
            title="Cumulative Coverage Milestones (Top 50 Max)",
            color="CoveragePct",
            color_continuous_scale=list(chart_theme["positive_scale"]),
            labels={"CoveragePct": "Cumulative coverage (%)", "Bucket": ""},
            text=coverage_df["CoveragePct"].map(lambda value: f"{value:.2f}%"),
        )
        coverage_fig.update_layout(
            height=390,
            margin={"l": 8, "r": 8, "t": 48, "b": 8},
            coloraxis_showscale=False,
            yaxis_range=[0, 100],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=str(chart_theme["plot_bg"]),
            font={"color": str(chart_theme["font_color"])},
            title={"x": 0, "xanchor": "left"},
        )
        coverage_fig.update_xaxes(showgrid=False, zeroline=False)
        coverage_fig.update_yaxes(showgrid=True, gridcolor=str(chart_theme["grid_color"]), zeroline=False)
        coverage_fig.update_traces(
            textposition="outside",
            marker_line={"width": 1.1, "color": str(chart_theme["marker_line"])},
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
