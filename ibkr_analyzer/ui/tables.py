from __future__ import annotations

import pandas as pd
import streamlit as st

from .constants import get_active_theme


def render_dataframe(
    dataframe: pd.DataFrame,
    *,
    use_container_width: bool = True,
    hide_index: bool = True,
    height: int | None = None,
) -> None:
    if get_active_theme() != "editorial":
        st.dataframe(
            dataframe,
            use_container_width=use_container_width,
            hide_index=hide_index,
            height=height,
        )
        return

    safe_dataframe = dataframe.copy().where(pd.notna(dataframe), "")
    table_html = safe_dataframe.to_html(
        index=not hide_index,
        escape=True,
        border=0,
        classes="editorial-data-table",
    )
    max_height = f"max-height: {height}px;" if height else ""
    st.markdown(
        f'<div class="editorial-table-shell" style="{max_height}">{table_html}</div>',
        unsafe_allow_html=True,
    )
