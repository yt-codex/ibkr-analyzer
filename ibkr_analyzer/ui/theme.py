import streamlit as st


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Manrope:wght@400;500;600;700;800&display=swap');

            :root {
                --bg-1: #08111f;
                --bg-2: #0c1730;
                --bg-3: #122347;
                --panel-bg: rgba(11, 19, 36, 0.78);
                --panel-bg-strong: rgba(10, 18, 34, 0.92);
                --panel-border: rgba(146, 170, 211, 0.18);
                --panel-border-strong: rgba(160, 188, 234, 0.26);
                --text-main: #f3f7ff;
                --text-soft: #acbedf;
                --text-muted: #7d90b6;
                --accent: #7ae7c7;
                --accent-2: #78a8ff;
                --accent-3: #ffb86a;
                --danger: #ff6b88;
                --shadow-soft: 0 16px 38px rgba(2, 8, 19, 0.24);
                --shadow-strong: 0 26px 60px rgba(2, 8, 19, 0.32);
            }

            html, body {
                font-family: "Manrope", "Segoe UI", sans-serif;
                color: var(--text-main);
            }

            .stApp {
                background:
                    radial-gradient(circle at 14% 14%, rgba(122, 231, 199, 0.12), transparent 32%),
                    radial-gradient(circle at 86% 10%, rgba(120, 168, 255, 0.14), transparent 34%),
                    linear-gradient(160deg, var(--bg-1) 0%, var(--bg-2) 48%, var(--bg-3) 100%);
                color: var(--text-main);
            }

            .block-container {
                max-width: 1360px;
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }

            section[data-testid="stSidebar"] > div {
                background: linear-gradient(180deg, rgba(10, 18, 35, 0.98) 0%, rgba(13, 22, 41, 0.94) 100%);
                border-right: 1px solid rgba(138, 160, 199, 0.12);
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
                    linear-gradient(135deg, rgba(122, 231, 199, 0.12), rgba(120, 168, 255, 0.12) 48%, rgba(255, 184, 106, 0.08)),
                    linear-gradient(180deg, rgba(12, 21, 40, 0.98), rgba(10, 18, 35, 0.94));
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
                border: 1px solid rgba(160, 193, 236, 0.22);
                background: rgba(8, 15, 29, 0.3);
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #dbf7ee;
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
                color: #c3d4ef;
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
                border: 1px solid rgba(160, 193, 236, 0.18);
                background: rgba(8, 15, 29, 0.38);
                font-size: 0.79rem;
                font-weight: 700;
                color: #e7f2ff;
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
                border: 1px solid rgba(152, 180, 222, 0.16);
                background: rgba(10, 18, 35, 0.76);
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
                border: 1px solid rgba(152, 180, 222, 0.14);
                background: linear-gradient(90deg, rgba(122, 231, 199, 0.07), rgba(120, 168, 255, 0.05) 44%, transparent 100%);
            }

            .section-eyebrow {
                margin-bottom: 0.28rem;
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #a5dfd1;
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
                border: 1px solid rgba(152, 180, 222, 0.2);
                background: rgba(9, 16, 30, 0.5);
                font-size: 0.8rem;
                font-weight: 700;
                color: #e4efff;
            }

            div[data-testid="stMetric"] {
                background: rgba(10, 18, 35, 0.78);
                border: 1px solid rgba(152, 180, 222, 0.16);
                padding: 0.82rem 0.9rem;
                border-radius: 18px;
                min-height: 118px;
                box-shadow: var(--shadow-soft);
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
                color: #daf6ec;
            }

            div[data-testid="stTabs"] [data-baseweb="tab-list"] {
                gap: 0.38rem;
                flex-wrap: wrap;
                margin-bottom: 1rem;
                padding: 0.42rem;
                border-radius: 18px;
                background: rgba(9, 16, 30, 0.44);
                border: 1px solid rgba(152, 180, 222, 0.12);
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
                background: linear-gradient(135deg, rgba(122, 231, 199, 0.16), rgba(120, 168, 255, 0.16));
                color: var(--text-main);
                box-shadow: inset 0 0 0 1px rgba(165, 195, 232, 0.2);
            }

            div[data-testid="stPlotlyChart"],
            div[data-testid="stDataFrame"] {
                border-radius: 20px;
                overflow: hidden;
                background: rgba(9, 16, 30, 0.36);
                border: 1px solid rgba(152, 180, 222, 0.12);
                box-shadow: var(--shadow-soft);
            }

            [data-testid="stFileUploaderDropzone"] {
                border-radius: 18px;
                border: 1px dashed rgba(152, 180, 222, 0.28);
                background: rgba(9, 16, 30, 0.52);
            }

            [data-testid="stFileUploaderDropzone"] * {
                color: var(--text-soft);
            }

            .method-tip {
                display: inline-flex;
                align-items: center;
                gap: 0.48rem;
                margin: 0.15rem 0 0.45rem 0;
                padding: 0.34rem 0.55rem;
                border-radius: 999px;
                border: 1px solid rgba(152, 180, 222, 0.16);
                background: rgba(9, 16, 30, 0.44);
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
                border: 1px solid rgba(160, 185, 225, 0.38);
                color: #b8caea;
                font-size: 0.68rem;
                cursor: help;
                line-height: 1rem;
            }

            .hint-value {
                font-family: "Manrope", "Segoe UI", sans-serif;
                font-size: 0.72rem;
                font-weight: 700;
                color: #d9e7ff;
                background: rgba(95, 125, 178, 0.18);
                border: 1px solid rgba(151, 178, 224, 0.22);
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
