import html

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ibkr_analyzer.report_utils import (
    ParsedIBKRReport,
    annualize_return,
    build_benchmark_long,
    get_table,
    period_years,
    return_method_label_and_tooltip,
    sanitize_total_rows,
    to_numeric,
)
from ibkr_analyzer.ui.chrome import render_section_intro
from ibkr_analyzer.ui.constants import CHART_COLORS, PLOTLY_TEMPLATE


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

    render_section_intro(
        eyebrow="Performance Lens",
        title="Returns Versus Benchmarks",
        subtitle="Compare annual results, cumulative compounding, drawdowns, and symbol-level contribution without leaving the dashboard.",
        badge=f"Method: {method_label}",
    )

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

    annual_returns_long = pd.DataFrame()
    if not periodic_returns_long.empty:
        annual_returns_long = periodic_returns_long.copy()
        annual_returns_long = annual_returns_long.dropna(subset=["Date", "Return"])
        annual_returns_long["Year"] = annual_returns_long["Date"].dt.year
        annual_returns_long = (
            annual_returns_long.groupby(["Series", "Year"], as_index=False)
            .agg(
                AnnualReturn=(
                    "Return",
                    lambda values: ((1 + (values / 100.0)).prod() - 1) * 100,
                )
            )
            .sort_values(["Year", "Series"])
        )
        current_year = pd.Timestamp.today().year
        annual_returns_long["YearLabel"] = annual_returns_long["Year"].astype(int).astype(str)
        annual_returns_long.loc[
            annual_returns_long["Year"] == current_year, "YearLabel"
        ] = (
            annual_returns_long.loc[
                annual_returns_long["Year"] == current_year, "YearLabel"
            ]
            + " YTD"
        )

    performance_chart_col, drawdown_chart_col = st.columns((1.3, 1.0))

    with performance_chart_col:
        if annual_returns_long.empty:
            st.info("Annual return comparison data was not found.")
        else:
            year_order = (
                annual_returns_long[["Year", "YearLabel"]]
                .drop_duplicates()
                .sort_values("Year")["YearLabel"]
                .tolist()
            )
            perf_fig = px.bar(
                annual_returns_long,
                x="YearLabel",
                y="AnnualReturn",
                color="Series",
                barmode="group",
                template=PLOTLY_TEMPLATE,
                title="Annual Return Comparison",
                color_discrete_sequence=CHART_COLORS,
                category_orders={"YearLabel": year_order},
            )
            perf_fig.update_layout(
                height=360,
                margin={"l": 12, "r": 12, "t": 56, "b": 12},
                yaxis_title="Return (%)",
                xaxis_title="Year",
                legend_title_text="Series",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 27, 45, 0.24)",
                hovermode="x unified",
                font={"color": "#f3f7fb"},
                title={"x": 0, "xanchor": "left"},
                legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
            )
            perf_fig.update_xaxes(showgrid=False, zeroline=False)
            perf_fig.update_yaxes(
                showgrid=True,
                gridcolor="rgba(138, 160, 199, 0.10)",
                zeroline=False,
            )
            perf_fig.update_traces(
                marker_line={"width": 1.1, "color": "rgba(15, 27, 45, 0.95)"},
                hovertemplate="%{x}<br>%{fullData.name}: %{y:.2f}%<extra></extra>"
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
                        fillcolor="rgba(255, 107, 107, 0.14)",
                        line={"color": CHART_COLORS[3], "width": 2.6},
                        name="Drawdown",
                        hovertemplate="%{x|%b %Y}<br>Drawdown: %{y:.2f}%<extra></extra>",
                    )
                )
                drawdown_fig.update_layout(
                    template=PLOTLY_TEMPLATE,
                    height=360,
                    margin={"l": 12, "r": 12, "t": 56, "b": 12},
                    yaxis_title="Drawdown (%)",
                    xaxis_title="Date",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15, 27, 45, 0.24)",
                    hovermode="x unified",
                    font={"color": "#f3f7fb"},
                    title={
                        "text": f"Drawdown ({portfolio_series_name})",
                        "x": 0,
                        "xanchor": "left",
                    },
                )
                drawdown_fig.update_xaxes(
                    showgrid=True,
                    gridcolor="rgba(138, 160, 199, 0.10)",
                    zeroline=False,
                )
                drawdown_fig.update_yaxes(
                    showgrid=True,
                    gridcolor="rgba(138, 160, 199, 0.10)",
                    zeroline=False,
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
            margin={"l": 12, "r": 12, "t": 20, "b": 12},
            yaxis_title="Cumulative Return (%)",
            xaxis_title="Date",
            legend_title_text="Series",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 27, 45, 0.24)",
            hovermode="x unified",
            font={"color": "#f3f7fb"},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
        )
        cumulative_fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(138, 160, 199, 0.10)",
            zeroline=False,
        )
        cumulative_fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(138, 160, 199, 0.10)",
            zeroline=False,
        )
        cumulative_fig.update_traces(line={"width": 2.5})
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
                margin={"l": 12, "r": 12, "t": 26, "b": 12},
                showlegend=False,
                yaxis_range=[y_start, y_end],
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 27, 45, 0.24)",
                font={"color": "#f3f7fb"},
                title={"x": 0, "xanchor": "left"},
            )
            annualized_fig.update_xaxes(showgrid=False, zeroline=False)
            annualized_fig.update_yaxes(
                showgrid=True,
                gridcolor="rgba(138, 160, 199, 0.10)",
                zeroline=False,
            )
            annualized_fig.update_traces(
                textposition="outside",
                cliponaxis=False,
                marker_line={"width": 1.1, "color": "rgba(15, 27, 45, 0.95)"},
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
                    [0.0, "#7a3131"],
                    [0.45, "#22354d"],
                    [0.5, "#2b3f58"],
                    [0.55, "#396c75"],
                    [1.0, "#63e6be"],
                ],
                aspect="auto",
                template=PLOTLY_TEMPLATE,
            )
            heatmap_fig.update_layout(
                height=320,
                margin={"l": 12, "r": 12, "t": 56, "b": 12},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 27, 45, 0.24)",
                font={"color": "#f3f7fb"},
                title={"x": 0, "xanchor": "left"},
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
            color_continuous_scale=["#9fb0c7", "#63e6be"],
            template=PLOTLY_TEMPLATE,
            title="Top Contributors",
            labels={"Contribution": "Contribution (%)", "Symbol": ""},
        )
        winners_fig.update_layout(
            height=340,
            margin={"l": 8, "r": 8, "t": 54, "b": 10},
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 27, 45, 0.24)",
            font={"color": "#f3f7fb"},
            title={"x": 0, "xanchor": "left"},
        )
        winners_fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(138, 160, 199, 0.10)",
            zeroline=False,
        )
        winners_fig.update_yaxes(showgrid=False, zeroline=False)
        winners_fig.update_traces(marker_line={"width": 1.1, "color": "rgba(15, 27, 45, 0.95)"})
        st.plotly_chart(winners_fig, use_container_width=True)

    with losers_col:
        losers_fig = px.bar(
            worst,
            x="Contribution",
            y="Symbol",
            orientation="h",
            color="Contribution",
            color_continuous_scale=["#ffc2c2", "#ff6b6b"],
            template=PLOTLY_TEMPLATE,
            title="Bottom Contributors",
            labels={"Contribution": "Contribution (%)", "Symbol": ""},
        )
        losers_fig.update_layout(
            height=340,
            margin={"l": 8, "r": 8, "t": 54, "b": 10},
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 27, 45, 0.24)",
            font={"color": "#f3f7fb"},
            title={"x": 0, "xanchor": "left"},
        )
        losers_fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(138, 160, 199, 0.10)",
            zeroline=False,
        )
        losers_fig.update_yaxes(showgrid=False, zeroline=False)
        losers_fig.update_traces(marker_line={"width": 1.1, "color": "rgba(15, 27, 45, 0.95)"})
        st.plotly_chart(losers_fig, use_container_width=True)
