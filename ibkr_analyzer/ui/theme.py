import streamlit as st


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Manrope:wght@400;500;600;700;800&display=swap');

            :root {
                --bg-1: #07101f;
                --bg-2: #0c1630;
                --bg-3: #0f1f3d;
                --panel-bg: rgba(10, 18, 35, 0.74);
                --panel-strong: rgba(13, 22, 41, 0.92);
                --panel-border: rgba(136, 158, 199, 0.18);
                --panel-border-strong: rgba(152, 180, 222, 0.3);
                --text-main: #f4f8ff;
                --text-soft: #a4b5d4;
                --text-muted: #7687ab;
                --accent: #7ae7c7;
                --accent-2: #78a8ff;
                --accent-3: #ffb86a;
                --danger: #ff6b88;
                --shadow-soft: 0 18px 40px rgba(2, 8, 19, 0.32);
                --shadow-strong: 0 26px 70px rgba(2, 7, 18, 0.45);
            }

            html, body, [class*="css"] {
                font-family: "Manrope", "Segoe UI", sans-serif;
            }

            .stApp {
                position: relative;
                background:
                    radial-gradient(circle at 12% 12%, rgba(122, 231, 199, 0.16), transparent 32%),
                    radial-gradient(circle at 88% 10%, rgba(120, 168, 255, 0.17), transparent 34%),
                    radial-gradient(circle at 50% 100%, rgba(255, 184, 106, 0.12), transparent 28%),
                    linear-gradient(160deg, var(--bg-1) 0%, var(--bg-2) 48%, var(--bg-3) 100%);
                color: var(--text-main);
                overflow: hidden;
            }

            .stApp::before {
                content: "";
                position: fixed;
                inset: 0;
                background-image:
                    linear-gradient(rgba(140, 160, 199, 0.055) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(140, 160, 199, 0.055) 1px, transparent 1px);
                background-size: 34px 34px;
                mask-image: radial-gradient(circle at center, black 45%, transparent 88%);
                pointer-events: none;
                opacity: 0.22;
            }

            .stApp::after {
                content: "";
                position: fixed;
                width: 26rem;
                height: 26rem;
                right: -8rem;
                bottom: -11rem;
                border-radius: 999px;
                background: radial-gradient(circle, rgba(122, 231, 199, 0.22) 0%, rgba(122, 231, 199, 0) 70%);
                filter: blur(10px);
                animation: driftGlow 18s ease-in-out infinite;
                pointer-events: none;
            }

            @keyframes driftGlow {
                0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
                50% { transform: translate3d(-1.5rem, -1rem, 0) scale(1.06); }
            }

            @keyframes fadeSlide {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .block-container {
                max-width: 1380px;
                padding-top: 1.6rem;
                padding-bottom: 2rem;
            }

            section[data-testid="stSidebar"] > div {
                background:
                    linear-gradient(180deg, rgba(10, 18, 35, 0.96) 0%, rgba(12, 21, 42, 0.92) 100%);
                border-right: 1px solid rgba(138, 160, 199, 0.12);
            }

            section[data-testid="stSidebar"] .block-container {
                padding-top: 1.15rem;
            }

            h1, h2, h3 {
                font-family: "Space Grotesk", "Segoe UI", sans-serif;
                letter-spacing: -0.025em;
                color: var(--text-main);
            }

            p, li, label {
                color: var(--text-soft);
            }

            .hero-card {
                position: relative;
                overflow: hidden;
                display: flex;
                align-items: flex-end;
                justify-content: space-between;
                gap: 1.2rem;
                padding: 1.5rem 1.55rem;
                margin: 0.1rem 0 1.1rem 0;
                border: 1px solid var(--panel-border-strong);
                border-radius: 26px;
                background:
                    linear-gradient(130deg, rgba(122, 231, 199, 0.16), rgba(120, 168, 255, 0.14) 45%, rgba(255, 184, 106, 0.08)),
                    linear-gradient(180deg, rgba(13, 22, 41, 0.96), rgba(11, 19, 37, 0.9));
                box-shadow: var(--shadow-strong);
                animation: fadeSlide 420ms ease-out;
            }

            .hero-card::before {
                content: "";
                position: absolute;
                inset: auto -6rem -7rem auto;
                width: 18rem;
                height: 18rem;
                border-radius: 999px;
                background: radial-gradient(circle, rgba(120, 168, 255, 0.25) 0%, transparent 72%);
                pointer-events: none;
            }

            .hero-copy {
                position: relative;
                z-index: 1;
                max-width: 52rem;
            }

            .hero-eyebrow {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                margin-bottom: 0.55rem;
                padding: 0.28rem 0.62rem;
                border-radius: 999px;
                border: 1px solid rgba(160, 193, 236, 0.24);
                background: rgba(7, 13, 27, 0.34);
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #d9f4eb;
            }

            .hero-title {
                font-size: clamp(2rem, 4vw, 3.1rem);
                line-height: 0.95;
                font-weight: 700;
                margin-bottom: 0.45rem;
                max-width: 10ch;
            }

            .hero-sub {
                max-width: 48rem;
                font-size: 1rem;
                line-height: 1.6;
                margin-bottom: 0;
                color: #bfd0ec;
            }

            .hero-badges {
                position: relative;
                z-index: 1;
                display: flex;
                flex-wrap: wrap;
                justify-content: flex-end;
                gap: 0.55rem;
                max-width: 24rem;
            }

            .hero-badge {
                padding: 0.48rem 0.72rem;
                border-radius: 999px;
                border: 1px solid rgba(160, 193, 236, 0.18);
                background: rgba(8, 15, 29, 0.42);
                font-size: 0.79rem;
                font-weight: 700;
                color: #e7f2ff;
                backdrop-filter: blur(8px);
            }

            .panel {
                background:
                    linear-gradient(180deg, rgba(255, 255, 255, 0.035), rgba(255, 255, 255, 0)),
                    var(--panel-bg);
                border: 1px solid var(--panel-border);
                border-radius: 18px;
                padding: 0.95rem 1rem;
                margin-bottom: 0.95rem;
                backdrop-filter: blur(12px);
                box-shadow: var(--shadow-soft);
            }

            .summary-strip {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 0.75rem;
                margin-bottom: 1.15rem;
            }

            .summary-item {
                min-height: 92px;
                padding: 0.9rem 0.95rem;
                border-radius: 18px;
                border: 1px solid rgba(152, 180, 222, 0.18);
                background:
                    linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0)),
                    rgba(10, 18, 35, 0.7);
                box-shadow: var(--shadow-soft);
            }

            .summary-item-primary {
                grid-column: span 2;
            }

            .summary-label {
                display: block;
                margin-bottom: 0.45rem;
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
                margin-top: 0.38rem;
                font-size: 0.78rem;
                color: var(--text-soft);
            }

            .section-intro {
                display: flex;
                align-items: end;
                justify-content: space-between;
                gap: 1rem;
                margin: 0.15rem 0 1rem 0;
                padding: 0.95rem 1rem;
                border-radius: 18px;
                border: 1px solid rgba(152, 180, 222, 0.14);
                background: linear-gradient(90deg, rgba(122, 231, 199, 0.08), rgba(120, 168, 255, 0.04) 45%, transparent 100%);
            }

            .section-eyebrow {
                margin-bottom: 0.3rem;
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #9fd8c8;
            }

            .section-title {
                font-family: "Space Grotesk", "Segoe UI", sans-serif;
                font-size: 1.3rem;
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
                align-self: center;
                padding: 0.48rem 0.72rem;
                border-radius: 999px;
                border: 1px solid rgba(152, 180, 222, 0.2);
                background: rgba(9, 16, 30, 0.52);
                font-size: 0.8rem;
                font-weight: 700;
                color: #e4efff;
            }

            div[data-testid="stMetric"] {
                position: relative;
                overflow: hidden;
                background:
                    linear-gradient(180deg, rgba(255, 255, 255, 0.045), rgba(255, 255, 255, 0)),
                    rgba(10, 18, 35, 0.78);
                border: 1px solid rgba(152, 180, 222, 0.18);
                padding: 0.82rem 0.9rem;
                border-radius: 18px;
                min-height: 118px;
                box-shadow: var(--shadow-soft);
            }

            div[data-testid="stMetric"]::before {
                content: "";
                position: absolute;
                inset: 0 0 auto 0;
                height: 3px;
                background: linear-gradient(90deg, var(--accent), var(--accent-2), var(--accent-3));
                opacity: 0.88;
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
                color: #d8f6ea;
            }

            div[data-testid="stTabs"] [data-baseweb="tab-list"] {
                gap: 0.4rem;
                flex-wrap: wrap;
                margin-bottom: 1rem;
                padding: 0.42rem;
                border-radius: 18px;
                background: rgba(9, 16, 30, 0.46);
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
                letter-spacing: 0.01em;
            }

            div[data-testid="stTabs"] [aria-selected="true"] {
                background: linear-gradient(135deg, rgba(122, 231, 199, 0.18), rgba(120, 168, 255, 0.18));
                color: var(--text-main);
                box-shadow: inset 0 0 0 1px rgba(165, 195, 232, 0.22);
            }

            div[data-testid="stPlotlyChart"] {
                background: rgba(9, 16, 30, 0.38);
                border: 1px solid rgba(152, 180, 222, 0.12);
                border-radius: 22px;
                padding: 0.4rem 0.45rem 0.2rem 0.45rem;
                box-shadow: var(--shadow-soft);
            }

            div[data-testid="stDataFrame"] {
                border: 1px solid rgba(152, 180, 222, 0.12);
                border-radius: 18px;
                overflow: hidden;
                background: rgba(9, 16, 30, 0.42);
                box-shadow: var(--shadow-soft);
            }

            [data-testid="stFileUploaderDropzone"] {
                border-radius: 18px;
                border: 1px dashed rgba(152, 180, 222, 0.28);
                background:
                    linear-gradient(180deg, rgba(122, 231, 199, 0.06), rgba(120, 168, 255, 0.05)),
                    rgba(9, 16, 30, 0.56);
            }

            [data-testid="stFileUploaderDropzone"] * {
                color: var(--text-soft);
            }

            .stButton > button,
            button[kind="secondary"] {
                border-radius: 12px;
                border: 1px solid rgba(152, 180, 222, 0.18);
                background: linear-gradient(135deg, rgba(122, 231, 199, 0.14), rgba(120, 168, 255, 0.18));
                color: var(--text-main);
                font-weight: 700;
                box-shadow: var(--shadow-soft);
            }

            div[data-baseweb="notification"] {
                border-radius: 18px;
                border: 1px solid rgba(152, 180, 222, 0.15);
                background: rgba(9, 16, 30, 0.78);
            }

            .method-tip {
                display: inline-flex;
                align-items: center;
                gap: 0.48rem;
                margin: 0.15rem 0 0.45rem 0;
                padding: 0.34rem 0.55rem;
                border-radius: 999px;
                border: 1px solid rgba(152, 180, 222, 0.16);
                background: rgba(9, 16, 30, 0.46);
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
                    padding: 1.2rem 1.15rem;
                    border-radius: 22px;
                    flex-direction: column;
                    align-items: flex-start;
                }

                .hero-badges {
                    justify-content: flex-start;
                    max-width: none;
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
