from __future__ import annotations

import html

import streamlit as st


def render_section_intro(
    *,
    eyebrow: str,
    title: str,
    subtitle: str,
    badge: str | None = None,
) -> None:
    badge_markup = (
        f"<div class='section-badge'>{html.escape(badge)}</div>" if badge else ""
    )
    st.markdown(
        f"""
        <div class="section-intro">
            <div>
                <div class="section-eyebrow">{html.escape(eyebrow)}</div>
                <div class="section-title">{html.escape(title)}</div>
                <p class="section-sub">{html.escape(subtitle)}</p>
            </div>
            {badge_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )
