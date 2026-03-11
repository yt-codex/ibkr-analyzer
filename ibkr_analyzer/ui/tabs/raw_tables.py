import streamlit as st

from ibkr_analyzer.report_utils import ParsedIBKRReport
from ibkr_analyzer.ui.tables import render_dataframe


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

    render_dataframe(
        selected_table,
        use_container_width=True,
        height=480,
        hide_index=True,
    )
    st.caption(f"Metadata rows in section: {len(report.metadata.get(section, []))}")
