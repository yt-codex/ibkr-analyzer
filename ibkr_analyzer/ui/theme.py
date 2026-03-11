import streamlit as st


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Manrope:wght@400;500;600;700;800&display=swap');

            :root {
                --bg-1: #071018;
                --bg-2: #0d1724;
                --bg-3: #15243a;
                --panel-bg: rgba(15, 27, 45, 0.9);
                --panel-bg-strong: rgba(21, 36, 58, 0.96);
                --panel-border: rgba(159, 176, 199, 0.18);
                --panel-border-strong: rgba(159, 176, 199, 0.28);
                --text-main: #f3f7ff;
                --text-soft: #9fb0c7;
                --text-muted: #74859c;
                --accent: #63e6be;
                --accent-2: #7cb8ff;
                --accent-3: #ffb454;
                --danger: #ff6b6b;
                --shadow-soft: 0 16px 38px rgba(2, 8, 19, 0.24);
                --shadow-strong: 0 26px 60px rgba(2, 8, 19, 0.32);
            }

            html, body {
                font-family: "Manrope", "Segoe UI", sans-serif;
                color: var(--text-main);
            }

            .stApp {
                background:
                    radial-gradient(circle at 14% 14%, rgba(99, 230, 190, 0.18), transparent 28%),
                    radial-gradient(circle at 84% 10%, rgba(255, 180, 84, 0.14), transparent 30%),
                    radial-gradient(circle at 55% 100%, rgba(124, 184, 255, 0.08), transparent 26%),
                    linear-gradient(160deg, var(--bg-1) 0%, var(--bg-2) 48%, var(--bg-3) 100%);
                color: var(--text-main);
            }

            .block-container {
                max-width: 1360px;
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }

            section[data-testid="stSidebar"] > div {
                background:
                    radial-gradient(circle at 0% 0%, rgba(99, 230, 190, 0.08), transparent 26%),
                    radial-gradient(circle at 100% 100%, rgba(255, 180, 84, 0.06), transparent 28%),
                    linear-gradient(180deg, rgba(15, 27, 45, 0.99) 0%, rgba(18, 31, 48, 0.97) 100%);
                border-right: 1px solid rgba(159, 176, 199, 0.12);
            }

            section[data-testid="stSidebar"] .block-container {
                padding-top: 1rem;
            }

            h1, h2, h3 {
                font-family: "Space Grotesk", "Segoe UI", sans-serif;
                letter-spacing: -0.025em;
                color: var(--text-main);
            }

            p, li, label, .stCaption {
                color: var(--text-soft);
            }

            .hero-card {
                display: grid;
                grid-template-columns: minmax(0, 1.8fr) minmax(240px, 0.9fr);
                gap: 1rem;
                align-items: end;
                padding: 1.45rem 1.5rem;
                margin: 0.1rem 0 1rem 0;
                border-radius: 24px;
                border: 1px solid var(--panel-border-strong);
                background:
                    linear-gradient(135deg, rgba(99, 230, 190, 0.16), rgba(124, 184, 255, 0.08) 38%, rgba(255, 180, 84, 0.16)),
                    linear-gradient(180deg, rgba(15, 27, 45, 0.99), rgba(21, 36, 58, 0.96));
                box-shadow: var(--shadow-strong);
            }

            .hero-copy {
                max-width: 52rem;
            }

            .hero-eyebrow {
                display: inline-flex;
                align-items: center;
                margin-bottom: 0.55rem;
                padding: 0.28rem 0.62rem;
                border-radius: 999px;
                border: 1px solid rgba(99, 230, 190, 0.3);
                background: rgba(99, 230, 190, 0.12);
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #dcfff4;
            }

            .hero-title {
                font-size: clamp(2rem, 4vw, 3rem);
                line-height: 0.95;
                font-weight: 700;
                margin-bottom: 0.45rem;
            }

            .hero-sub {
                max-width: 48rem;
                margin-bottom: 0;
                font-size: 1rem;
                line-height: 1.65;
                color: #bfccdd;
            }

            .hero-badges {
                display: flex;
                flex-wrap: wrap;
                justify-content: flex-end;
                align-content: start;
                gap: 0.55rem;
            }

            .hero-badge {
                padding: 0.48rem 0.72rem;
                border-radius: 999px;
                border: 1px solid rgba(99, 230, 190, 0.18);
                background: rgba(99, 230, 190, 0.08);
                font-size: 0.79rem;
                font-weight: 700;
                color: #edf4fc;
            }

            .hero-badge:nth-child(2) {
                border-color: rgba(124, 184, 255, 0.2);
                background: rgba(124, 184, 255, 0.1);
            }

            .hero-badge:nth-child(3) {
                border-color: rgba(255, 180, 84, 0.24);
                background: rgba(255, 180, 84, 0.1);
                color: #fff2dd;
            }

            .panel {
                background: var(--panel-bg);
                border: 1px solid var(--panel-border);
                border-radius: 18px;
                padding: 0.95rem 1rem;
                margin-bottom: 0.95rem;
                box-shadow: var(--shadow-soft);
            }

            .summary-strip {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 0.75rem;
                margin-bottom: 1.1rem;
            }

            .summary-item {
                min-height: 92px;
                padding: 0.9rem 0.95rem;
                border-radius: 18px;
                border: 1px solid rgba(159, 176, 199, 0.16);
                background: rgba(15, 27, 45, 0.76);
                box-shadow: var(--shadow-soft);
            }

            .summary-item-primary {
                grid-column: span 2;
            }

            .summary-label {
                display: block;
                margin-bottom: 0.42rem;
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: var(--text-muted);
            }

            .summary-value {
                display: block;
                font-family: "Space Grotesk", "Segoe UI", sans-serif;
                font-size: 1rem;
                font-weight: 700;
                line-height: 1.15;
                color: var(--text-main);
            }

            .summary-meta {
                display: block;
                margin-top: 0.35rem;
                font-size: 0.78rem;
                color: var(--text-soft);
            }

            .section-intro {
                display: flex;
                align-items: end;
                justify-content: space-between;
                gap: 1rem;
                margin: 0.1rem 0 1rem 0;
                padding: 0.95rem 1rem;
                border-radius: 18px;
                border: 1px solid rgba(159, 176, 199, 0.14);
                background: linear-gradient(90deg, rgba(99, 230, 190, 0.1), rgba(255, 180, 84, 0.05) 44%, transparent 100%);
            }

            .section-eyebrow {
                margin-bottom: 0.28rem;
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #9de8d1;
            }

            .section-title {
                font-family: "Space Grotesk", "Segoe UI", sans-serif;
                font-size: 1.26rem;
                font-weight: 700;
                color: var(--text-main);
            }

            .section-sub {
                margin: 0.28rem 0 0 0;
                max-width: 40rem;
                font-size: 0.94rem;
                line-height: 1.6;
                color: var(--text-soft);
            }

            .section-badge {
                flex-shrink: 0;
                padding: 0.48rem 0.72rem;
                border-radius: 999px;
                border: 1px solid rgba(159, 176, 199, 0.2);
                background: rgba(7, 16, 24, 0.48);
                font-size: 0.8rem;
                font-weight: 700;
                color: #edf4fc;
            }

            div[data-testid="stMetric"] {
                background: rgba(15, 27, 45, 0.78);
                border: 1px solid rgba(159, 176, 199, 0.16);
                padding: 0.82rem 0.9rem;
                border-radius: 18px;
                min-height: 118px;
                box-shadow: var(--shadow-soft);
            }

            div[data-testid="stMetric"]:nth-of-type(3n + 1) {
                box-shadow: inset 0 1px 0 rgba(99, 230, 190, 0.22), var(--shadow-soft);
            }

            div[data-testid="stMetric"]:nth-of-type(3n + 2) {
                box-shadow: inset 0 1px 0 rgba(124, 184, 255, 0.18), var(--shadow-soft);
            }

            div[data-testid="stMetric"]:nth-of-type(3n) {
                box-shadow: inset 0 1px 0 rgba(255, 180, 84, 0.22), var(--shadow-soft);
            }

            div[data-testid="stMetricLabel"] {
                color: var(--text-soft);
                font-weight: 600;
            }

            div[data-testid="stMetricValue"] {
                color: var(--text-main);
                font-family: "Space Grotesk", "Segoe UI", sans-serif;
                letter-spacing: -0.03em;
            }

            div[data-testid="stMetricDelta"] {
                color: #d8fff3;
            }

            div[data-testid="stTabs"] [data-baseweb="tab-list"] {
                gap: 0.38rem;
                flex-wrap: wrap;
                margin-bottom: 1rem;
                padding: 0.42rem;
                border-radius: 18px;
                background: rgba(7, 16, 24, 0.46);
                border: 1px solid rgba(159, 176, 199, 0.12);
            }

            div[data-testid="stTabs"] [data-baseweb="tab"] {
                height: auto;
                padding: 0.52rem 0.82rem;
                border-radius: 12px;
                color: var(--text-soft);
                font-family: "Space Grotesk", "Segoe UI", sans-serif;
                font-size: 0.84rem;
                font-weight: 700;
            }

            div[data-testid="stTabs"] [aria-selected="true"] {
                background: linear-gradient(135deg, rgba(99, 230, 190, 0.2), rgba(255, 180, 84, 0.16));
                color: var(--text-main);
                box-shadow: inset 0 0 0 1px rgba(99, 230, 190, 0.18);
            }

            div[data-testid="stPlotlyChart"],
            div[data-testid="stDataFrame"] {
                border-radius: 20px;
                overflow: hidden;
                background: rgba(15, 27, 45, 0.34);
                border: 1px solid rgba(159, 176, 199, 0.12);
                box-shadow: var(--shadow-soft);
            }

            [data-testid="stFileUploaderDropzone"] {
                border-radius: 18px;
                border: 1px dashed rgba(99, 230, 190, 0.34);
                background:
                    linear-gradient(180deg, rgba(99, 230, 190, 0.08), rgba(255, 180, 84, 0.04)),
                    rgba(15, 27, 45, 0.56);
            }

            [data-testid="stFileUploaderDropzone"] * {
                color: var(--text-soft);
            }

            .stButton > button,
            button[kind="secondary"],
            [data-testid="stBaseButton-secondary"] {
                border-radius: 12px;
                border: 1px solid rgba(99, 230, 190, 0.26);
                background: linear-gradient(135deg, rgba(99, 230, 190, 0.18), rgba(255, 180, 84, 0.16));
                color: #f3f7fb;
                font-weight: 700;
                box-shadow: var(--shadow-soft);
            }

            [data-testid="stAlert"],
            div[data-baseweb="notification"] {
                border-radius: 18px;
                border: 1px solid rgba(255, 180, 84, 0.26);
                background:
                    linear-gradient(90deg, rgba(255, 180, 84, 0.14), rgba(99, 230, 190, 0.08)),
                    rgba(15, 27, 45, 0.92);
                box-shadow: var(--shadow-soft);
            }

            [data-testid="stAlert"] p,
            [data-testid="stAlert"] div,
            div[data-baseweb="notification"] p,
            div[data-baseweb="notification"] div {
                color: #f3f7fb;
            }

            .method-tip {
                display: inline-flex;
                align-items: center;
                gap: 0.48rem;
                margin: 0.15rem 0 0.45rem 0;
                padding: 0.34rem 0.55rem;
                border-radius: 999px;
                border: 1px solid rgba(159, 176, 199, 0.16);
                background: rgba(15, 27, 45, 0.44);
                font-family: "Space Grotesk", "Segoe UI", sans-serif;
                font-size: 0.88rem;
                font-weight: 700;
                color: var(--text-main);
            }

            .hint-icon {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 1rem;
                height: 1rem;
                border-radius: 999px;
                border: 1px solid rgba(159, 176, 199, 0.38);
                color: #b8c6da;
                font-size: 0.68rem;
                cursor: help;
                line-height: 1rem;
            }

            .hint-value {
                font-family: "Manrope", "Segoe UI", sans-serif;
                font-size: 0.72rem;
                font-weight: 700;
                color: #e6eef8;
                background: rgba(124, 184, 255, 0.14);
                border: 1px solid rgba(159, 176, 199, 0.22);
                border-radius: 999px;
                padding: 0.1rem 0.44rem;
            }

            @media (max-width: 900px) {
                .hero-card {
                    grid-template-columns: 1fr;
                    padding: 1.2rem 1.1rem;
                }

                .hero-badges {
                    justify-content: flex-start;
                }

                .summary-item-primary {
                    grid-column: span 1;
                }

                .section-intro {
                    flex-direction: column;
                    align-items: flex-start;
                }

                .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1.4rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
