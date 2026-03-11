from __future__ import annotations

import streamlit as st

_CSS_PRESETS = {
    "slate_mint": {
        "body_font": '"Manrope", "Segoe UI", sans-serif',
        "heading_font": '"Space Grotesk", "Segoe UI", sans-serif',
        "bg_1": "#071018",
        "bg_2": "#0d1724",
        "bg_3": "#15243a",
        "panel_bg": "rgba(15, 27, 45, 0.9)",
        "panel_bg_strong": "rgba(21, 36, 58, 0.96)",
        "panel_border": "rgba(159, 176, 199, 0.18)",
        "panel_border_strong": "rgba(159, 176, 199, 0.28)",
        "text_main": "#f3f7ff",
        "text_soft": "#9fb0c7",
        "text_muted": "#74859c",
        "accent": "#63e6be",
        "accent_2": "#7cb8ff",
        "accent_3": "#ffb454",
        "danger": "#ff6b6b",
        "shadow_soft": "0 16px 38px rgba(2, 8, 19, 0.24)",
        "shadow_strong": "0 26px 60px rgba(2, 8, 19, 0.32)",
        "bg_glow_a": "rgba(99, 230, 190, 0.18)",
        "bg_glow_b": "rgba(255, 180, 84, 0.14)",
        "bg_glow_c": "rgba(124, 184, 255, 0.08)",
        "sidebar_glow_a": "rgba(99, 230, 190, 0.08)",
        "sidebar_glow_b": "rgba(255, 180, 84, 0.06)",
        "sidebar_bg_top": "rgba(15, 27, 45, 0.99)",
        "sidebar_bg_bottom": "rgba(18, 31, 48, 0.97)",
        "hero_overlay_a": "rgba(99, 230, 190, 0.16)",
        "hero_overlay_b": "rgba(124, 184, 255, 0.08)",
        "hero_overlay_c": "rgba(255, 180, 84, 0.16)",
        "hero_eyebrow_border": "rgba(99, 230, 190, 0.3)",
        "hero_eyebrow_bg": "rgba(99, 230, 190, 0.12)",
        "hero_eyebrow_text": "#dcfff4",
        "hero_sub": "#bfccdd",
        "hero_badge_border": "rgba(99, 230, 190, 0.18)",
        "hero_badge_bg": "rgba(99, 230, 190, 0.08)",
        "hero_badge_text": "#edf4fc",
        "hero_badge_alt_border": "rgba(124, 184, 255, 0.2)",
        "hero_badge_alt_bg": "rgba(124, 184, 255, 0.1)",
        "hero_badge_emphasis_border": "rgba(255, 180, 84, 0.24)",
        "hero_badge_emphasis_bg": "rgba(255, 180, 84, 0.1)",
        "hero_badge_emphasis_text": "#fff2dd",
        "section_bg_a": "rgba(99, 230, 190, 0.1)",
        "section_bg_b": "rgba(255, 180, 84, 0.05)",
        "section_eyebrow": "#9de8d1",
        "section_badge_bg": "rgba(7, 16, 24, 0.48)",
        "surface_tint_a": "rgba(99, 230, 190, 0.22)",
        "surface_tint_b": "rgba(124, 184, 255, 0.18)",
        "surface_tint_c": "rgba(255, 180, 84, 0.22)",
        "tabs_bg": "rgba(7, 16, 24, 0.46)",
        "tabs_active_a": "rgba(99, 230, 190, 0.2)",
        "tabs_active_b": "rgba(255, 180, 84, 0.16)",
        "tabs_active_border": "rgba(99, 230, 190, 0.18)",
        "chart_frame_bg": "rgba(15, 27, 45, 0.34)",
        "uploader_border": "rgba(99, 230, 190, 0.34)",
        "uploader_bg_a": "rgba(99, 230, 190, 0.08)",
        "uploader_bg_b": "rgba(255, 180, 84, 0.04)",
        "button_border": "rgba(99, 230, 190, 0.26)",
        "button_bg_a": "rgba(99, 230, 190, 0.18)",
        "button_bg_b": "rgba(255, 180, 84, 0.16)",
        "alert_border": "rgba(255, 180, 84, 0.26)",
        "alert_bg_a": "rgba(255, 180, 84, 0.14)",
        "alert_bg_b": "rgba(99, 230, 190, 0.08)",
        "hint_border": "rgba(159, 176, 199, 0.38)",
        "hint_icon": "#b8c6da",
        "hint_value_bg": "rgba(124, 184, 255, 0.14)",
        "hint_value_border": "rgba(159, 176, 199, 0.22)",
        "hint_value_text": "#e6eef8",
    },
    "editorial": {
        "body_font": '"Manrope", "Segoe UI", sans-serif',
        "heading_font": '"Fraunces", Georgia, serif',
        "bg_1": "#111214",
        "bg_2": "#17181b",
        "bg_3": "#23262c",
        "panel_bg": "rgba(26, 28, 32, 0.92)",
        "panel_bg_strong": "rgba(35, 38, 44, 0.97)",
        "panel_border": "rgba(185, 174, 161, 0.18)",
        "panel_border_strong": "rgba(230, 161, 90, 0.24)",
        "text_main": "#f5f1ea",
        "text_soft": "#b9aea1",
        "text_muted": "#8f857a",
        "accent": "#e6a15a",
        "accent_2": "#8ab4f8",
        "accent_3": "#f2c14e",
        "danger": "#f87171",
        "shadow_soft": "0 16px 38px rgba(6, 6, 7, 0.26)",
        "shadow_strong": "0 26px 60px rgba(6, 6, 7, 0.34)",
        "bg_glow_a": "rgba(230, 161, 90, 0.16)",
        "bg_glow_b": "rgba(138, 180, 248, 0.08)",
        "bg_glow_c": "rgba(242, 193, 78, 0.08)",
        "sidebar_glow_a": "rgba(230, 161, 90, 0.08)",
        "sidebar_glow_b": "rgba(138, 180, 248, 0.06)",
        "sidebar_bg_top": "rgba(26, 28, 32, 0.99)",
        "sidebar_bg_bottom": "rgba(35, 38, 44, 0.97)",
        "hero_overlay_a": "rgba(230, 161, 90, 0.18)",
        "hero_overlay_b": "rgba(138, 180, 248, 0.08)",
        "hero_overlay_c": "rgba(242, 193, 78, 0.12)",
        "hero_eyebrow_border": "rgba(230, 161, 90, 0.34)",
        "hero_eyebrow_bg": "rgba(230, 161, 90, 0.12)",
        "hero_eyebrow_text": "#fff0df",
        "hero_sub": "#d4c8ba",
        "hero_badge_border": "rgba(230, 161, 90, 0.22)",
        "hero_badge_bg": "rgba(230, 161, 90, 0.1)",
        "hero_badge_text": "#f8eee4",
        "hero_badge_alt_border": "rgba(138, 180, 248, 0.22)",
        "hero_badge_alt_bg": "rgba(138, 180, 248, 0.12)",
        "hero_badge_emphasis_border": "rgba(242, 193, 78, 0.28)",
        "hero_badge_emphasis_bg": "rgba(242, 193, 78, 0.12)",
        "hero_badge_emphasis_text": "#fff3ce",
        "section_bg_a": "rgba(230, 161, 90, 0.1)",
        "section_bg_b": "rgba(138, 180, 248, 0.05)",
        "section_eyebrow": "#f0c79f",
        "section_badge_bg": "rgba(17, 18, 20, 0.5)",
        "surface_tint_a": "rgba(230, 161, 90, 0.2)",
        "surface_tint_b": "rgba(138, 180, 248, 0.16)",
        "surface_tint_c": "rgba(242, 193, 78, 0.2)",
        "tabs_bg": "rgba(17, 18, 20, 0.46)",
        "tabs_active_a": "rgba(230, 161, 90, 0.18)",
        "tabs_active_b": "rgba(242, 193, 78, 0.16)",
        "tabs_active_border": "rgba(230, 161, 90, 0.2)",
        "chart_frame_bg": "rgba(35, 38, 44, 0.36)",
        "uploader_border": "rgba(230, 161, 90, 0.36)",
        "uploader_bg_a": "rgba(230, 161, 90, 0.08)",
        "uploader_bg_b": "rgba(242, 193, 78, 0.04)",
        "button_border": "rgba(230, 161, 90, 0.28)",
        "button_bg_a": "rgba(230, 161, 90, 0.2)",
        "button_bg_b": "rgba(242, 193, 78, 0.14)",
        "alert_border": "rgba(230, 161, 90, 0.28)",
        "alert_bg_a": "rgba(230, 161, 90, 0.14)",
        "alert_bg_b": "rgba(138, 180, 248, 0.08)",
        "hint_border": "rgba(185, 174, 161, 0.38)",
        "hint_icon": "#d7c9b9",
        "hint_value_bg": "rgba(230, 161, 90, 0.14)",
        "hint_value_border": "rgba(185, 174, 161, 0.24)",
        "hint_value_text": "#f7ede1",
    },
}


def inject_custom_css(theme_name: str = "slate_mint") -> None:
    theme = _CSS_PRESETS.get(theme_name, _CSS_PRESETS["slate_mint"])

    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Manrope:wght@400;500;600;700;800&display=swap');

            :root {{
                --bg-1: {theme["bg_1"]};
                --bg-2: {theme["bg_2"]};
                --bg-3: {theme["bg_3"]};
                --panel-bg: {theme["panel_bg"]};
                --panel-bg-strong: {theme["panel_bg_strong"]};
                --panel-border: {theme["panel_border"]};
                --panel-border-strong: {theme["panel_border_strong"]};
                --text-main: {theme["text_main"]};
                --text-soft: {theme["text_soft"]};
                --text-muted: {theme["text_muted"]};
                --accent: {theme["accent"]};
                --accent-2: {theme["accent_2"]};
                --accent-3: {theme["accent_3"]};
                --danger: {theme["danger"]};
                --shadow-soft: {theme["shadow_soft"]};
                --shadow-strong: {theme["shadow_strong"]};
            }}

            html, body {{
                font-family: {theme["body_font"]};
                color: var(--text-main);
            }}

            .stApp {{
                background:
                    radial-gradient(circle at 14% 14%, {theme["bg_glow_a"]}, transparent 28%),
                    radial-gradient(circle at 84% 10%, {theme["bg_glow_b"]}, transparent 30%),
                    radial-gradient(circle at 55% 100%, {theme["bg_glow_c"]}, transparent 26%),
                    linear-gradient(160deg, var(--bg-1) 0%, var(--bg-2) 48%, var(--bg-3) 100%);
                color: var(--text-main);
            }}

            .block-container {{
                max-width: 1360px;
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }}

            section[data-testid="stSidebar"] > div {{
                background:
                    radial-gradient(circle at 0% 0%, {theme["sidebar_glow_a"]}, transparent 26%),
                    radial-gradient(circle at 100% 100%, {theme["sidebar_glow_b"]}, transparent 28%),
                    linear-gradient(180deg, {theme["sidebar_bg_top"]} 0%, {theme["sidebar_bg_bottom"]} 100%);
                border-right: 1px solid var(--panel-border);
            }}

            section[data-testid="stSidebar"] .block-container {{
                padding-top: 1rem;
            }}

            h1, h2, h3 {{
                font-family: {theme["heading_font"]};
                letter-spacing: -0.025em;
                color: var(--text-main);
            }}

            p, li, label, .stCaption {{
                color: var(--text-soft);
            }}

            .hero-card {{
                display: grid;
                grid-template-columns: minmax(0, 1.8fr) minmax(240px, 0.9fr);
                gap: 1rem;
                align-items: end;
                padding: 1.45rem 1.5rem;
                margin: 0.1rem 0 1rem 0;
                border-radius: 24px;
                border: 1px solid var(--panel-border-strong);
                background:
                    linear-gradient(135deg, {theme["hero_overlay_a"]}, {theme["hero_overlay_b"]} 38%, {theme["hero_overlay_c"]}),
                    linear-gradient(180deg, var(--panel-bg-strong), var(--panel-bg));
                box-shadow: var(--shadow-strong);
            }}

            .hero-copy {{
                max-width: 52rem;
            }}

            .hero-eyebrow {{
                display: inline-flex;
                align-items: center;
                margin-bottom: 0.55rem;
                padding: 0.28rem 0.62rem;
                border-radius: 999px;
                border: 1px solid {theme["hero_eyebrow_border"]};
                background: {theme["hero_eyebrow_bg"]};
                font-family: {theme["heading_font"]};
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: {theme["hero_eyebrow_text"]};
            }}

            .hero-title {{
                font-family: {theme["heading_font"]};
                font-size: clamp(2rem, 4vw, 3rem);
                line-height: 0.95;
                font-weight: 700;
                margin-bottom: 0.45rem;
            }}

            .hero-sub {{
                max-width: 48rem;
                margin-bottom: 0;
                font-size: 1rem;
                line-height: 1.65;
                color: {theme["hero_sub"]};
            }}

            .hero-badges {{
                display: flex;
                flex-wrap: wrap;
                justify-content: flex-end;
                align-content: start;
                gap: 0.55rem;
            }}

            .hero-badge {{
                padding: 0.48rem 0.72rem;
                border-radius: 999px;
                border: 1px solid {theme["hero_badge_border"]};
                background: {theme["hero_badge_bg"]};
                font-size: 0.79rem;
                font-weight: 700;
                color: {theme["hero_badge_text"]};
            }}

            .hero-badge:nth-child(2) {{
                border-color: {theme["hero_badge_alt_border"]};
                background: {theme["hero_badge_alt_bg"]};
            }}

            .hero-badge:nth-child(3) {{
                border-color: {theme["hero_badge_emphasis_border"]};
                background: {theme["hero_badge_emphasis_bg"]};
                color: {theme["hero_badge_emphasis_text"]};
            }}

            .panel {{
                background: var(--panel-bg);
                border: 1px solid var(--panel-border);
                border-radius: 18px;
                padding: 0.95rem 1rem;
                margin-bottom: 0.95rem;
                box-shadow: var(--shadow-soft);
            }}

            .summary-strip {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 0.75rem;
                margin-bottom: 1.1rem;
            }}

            .summary-item {{
                min-height: 92px;
                padding: 0.9rem 0.95rem;
                border-radius: 18px;
                border: 1px solid var(--panel-border);
                background: var(--panel-bg);
                box-shadow: var(--shadow-soft);
            }}

            .summary-item-primary {{
                grid-column: span 2;
            }}

            .summary-label {{
                display: block;
                margin-bottom: 0.42rem;
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: var(--text-muted);
            }}

            .summary-value {{
                display: block;
                font-family: {theme["heading_font"]};
                font-size: 1rem;
                font-weight: 700;
                line-height: 1.15;
                color: var(--text-main);
            }}

            .summary-meta {{
                display: block;
                margin-top: 0.35rem;
                font-size: 0.78rem;
                color: var(--text-soft);
            }}

            .section-intro {{
                display: flex;
                align-items: end;
                justify-content: space-between;
                gap: 1rem;
                margin: 0.1rem 0 1rem 0;
                padding: 0.95rem 1rem;
                border-radius: 18px;
                border: 1px solid var(--panel-border);
                background: linear-gradient(90deg, {theme["section_bg_a"]}, {theme["section_bg_b"]} 44%, transparent 100%);
            }}

            .section-eyebrow {{
                margin-bottom: 0.28rem;
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: {theme["section_eyebrow"]};
            }}

            .section-title {{
                font-family: {theme["heading_font"]};
                font-size: 1.26rem;
                font-weight: 700;
                color: var(--text-main);
            }}

            .section-sub {{
                margin: 0.28rem 0 0 0;
                max-width: 40rem;
                font-size: 0.94rem;
                line-height: 1.6;
                color: var(--text-soft);
            }}

            .section-badge {{
                flex-shrink: 0;
                padding: 0.48rem 0.72rem;
                border-radius: 999px;
                border: 1px solid var(--panel-border);
                background: {theme["section_badge_bg"]};
                font-size: 0.8rem;
                font-weight: 700;
                color: var(--text-main);
            }}

            div[data-testid="stMetric"] {{
                background: var(--panel-bg);
                border: 1px solid var(--panel-border);
                padding: 0.82rem 0.9rem;
                border-radius: 18px;
                min-height: 118px;
                box-shadow: var(--shadow-soft);
            }}

            div[data-testid="stMetric"]:nth-of-type(3n + 1) {{
                box-shadow: inset 0 1px 0 {theme["surface_tint_a"]}, var(--shadow-soft);
            }}

            div[data-testid="stMetric"]:nth-of-type(3n + 2) {{
                box-shadow: inset 0 1px 0 {theme["surface_tint_b"]}, var(--shadow-soft);
            }}

            div[data-testid="stMetric"]:nth-of-type(3n) {{
                box-shadow: inset 0 1px 0 {theme["surface_tint_c"]}, var(--shadow-soft);
            }}

            div[data-testid="stMetricLabel"] {{
                color: var(--text-soft);
                font-weight: 600;
            }}

            div[data-testid="stMetricValue"] {{
                color: var(--text-main);
                font-family: {theme["heading_font"]};
                letter-spacing: -0.03em;
            }}

            div[data-testid="stMetricDelta"] {{
                color: var(--accent);
            }}

            div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
                gap: 0.38rem;
                flex-wrap: wrap;
                margin-bottom: 1rem;
                padding: 0.42rem;
                border-radius: 18px;
                background: {theme["tabs_bg"]};
                border: 1px solid var(--panel-border);
            }}

            div[data-testid="stTabs"] [data-baseweb="tab"] {{
                height: auto;
                padding: 0.52rem 0.82rem;
                border-radius: 12px;
                color: var(--text-soft);
                font-family: {theme["heading_font"]};
                font-size: 0.84rem;
                font-weight: 700;
            }}

            div[data-testid="stTabs"] [aria-selected="true"] {{
                background: linear-gradient(135deg, {theme["tabs_active_a"]}, {theme["tabs_active_b"]});
                color: var(--text-main);
                box-shadow: inset 0 0 0 1px {theme["tabs_active_border"]};
            }}

            div[data-testid="stPlotlyChart"],
            div[data-testid="stDataFrame"] {{
                border-radius: 20px;
                overflow: hidden;
                background: {theme["chart_frame_bg"]};
                border: 1px solid var(--panel-border);
                box-shadow: var(--shadow-soft);
            }}

            [data-testid="stFileUploaderDropzone"] {{
                border-radius: 18px;
                border: 1px dashed {theme["uploader_border"]};
                background:
                    linear-gradient(180deg, {theme["uploader_bg_a"]}, {theme["uploader_bg_b"]}),
                    var(--panel-bg);
            }}

            [data-testid="stFileUploaderDropzone"] * {{
                color: var(--text-soft);
            }}

            .stButton > button,
            button[kind="secondary"],
            [data-testid="stBaseButton-secondary"] {{
                border-radius: 12px;
                border: 1px solid {theme["button_border"]};
                background: linear-gradient(135deg, {theme["button_bg_a"]}, {theme["button_bg_b"]});
                color: var(--text-main);
                font-weight: 700;
                box-shadow: var(--shadow-soft);
            }}

            [data-testid="stAlert"],
            div[data-baseweb="notification"] {{
                border-radius: 18px;
                border: 1px solid {theme["alert_border"]};
                background:
                    linear-gradient(90deg, {theme["alert_bg_a"]}, {theme["alert_bg_b"]}),
                    var(--panel-bg-strong);
                box-shadow: var(--shadow-soft);
            }}

            [data-testid="stAlert"] p,
            [data-testid="stAlert"] div,
            div[data-baseweb="notification"] p,
            div[data-baseweb="notification"] div {{
                color: var(--text-main);
            }}

            .method-tip {{
                display: inline-flex;
                align-items: center;
                gap: 0.48rem;
                margin: 0.15rem 0 0.45rem 0;
                padding: 0.34rem 0.55rem;
                border-radius: 999px;
                border: 1px solid var(--panel-border);
                background: var(--panel-bg);
                font-family: {theme["heading_font"]};
                font-size: 0.88rem;
                font-weight: 700;
                color: var(--text-main);
            }}

            .hint-icon {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 1rem;
                height: 1rem;
                border-radius: 999px;
                border: 1px solid {theme["hint_border"]};
                color: {theme["hint_icon"]};
                font-size: 0.68rem;
                cursor: help;
                line-height: 1rem;
            }}

            .hint-value {{
                font-family: {theme["body_font"]};
                font-size: 0.72rem;
                font-weight: 700;
                color: {theme["hint_value_text"]};
                background: {theme["hint_value_bg"]};
                border: 1px solid {theme["hint_value_border"]};
                border-radius: 999px;
                padding: 0.1rem 0.44rem;
            }}

            @media (max-width: 900px) {{
                .hero-card {{
                    grid-template-columns: 1fr;
                    padding: 1.2rem 1.1rem;
                }}

                .hero-badges {{
                    justify-content: flex-start;
                }}

                .summary-item-primary {{
                    grid-column: span 1;
                }}

                .section-intro {{
                    flex-direction: column;
                    align-items: flex-start;
                }}

                .block-container {{
                    padding-top: 1rem;
                    padding-bottom: 1.4rem;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
