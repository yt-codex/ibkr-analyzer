import streamlit as st


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=DM+Sans:wght@400;500;700&display=swap');

            :root {
                --card-bg: rgba(16, 23, 42, 0.78);
                --card-border: rgba(131, 151, 183, 0.22);
                --text-soft: #9fb1d1;
                --accent: #28d5b5;
            }

            html, body, [class*="css"] {
                font-family: "DM Sans", "Segoe UI", sans-serif;
            }

            .stApp {
                background:
                    radial-gradient(circle at 10% 5%, rgba(40, 213, 181, 0.18), transparent 35%),
                    radial-gradient(circle at 90% 10%, rgba(92, 163, 255, 0.2), transparent 40%),
                    linear-gradient(165deg, #0a0f1e 0%, #0d1328 50%, #101a34 100%);
                color: #e8efff;
            }

            .block-container {
                max-width: 1300px;
                padding-top: 2.2rem;
                padding-bottom: 1.25rem;
            }

            h1, h2, h3 {
                font-family: "Space Grotesk", "Segoe UI", sans-serif;
                letter-spacing: -0.02em;
            }

            .hero-card {
                background: linear-gradient(140deg, rgba(40, 213, 181, 0.15), rgba(92, 163, 255, 0.14));
                border: 1px solid rgba(148, 189, 255, 0.25);
                border-radius: 18px;
                padding: 1.1rem 1.25rem;
                margin-top: 0.35rem;
                margin-bottom: 1rem;
                box-shadow: 0 16px 32px rgba(4, 8, 20, 0.45);
            }

            .hero-title {
                font-size: 1.75rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }

            .hero-sub {
                color: var(--text-soft);
                font-size: 0.95rem;
                margin-bottom: 0;
            }

            .panel {
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                border-radius: 14px;
                padding: 0.8rem 0.95rem;
                margin-bottom: 0.9rem;
                backdrop-filter: blur(8px);
            }

            div[data-testid="stMetric"] {
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                padding: 0.55rem 0.75rem;
                border-radius: 12px;
            }

            div[data-testid="stMetricLabel"] {
                color: #9bb0d6;
            }

            div[data-testid="stMetricValue"] {
                color: #f5fbff;
                font-family: "Space Grotesk", "Segoe UI", sans-serif;
            }

            .method-tip {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                font-family: "Space Grotesk", "Segoe UI", sans-serif;
                font-size: 1rem;
                font-weight: 700;
                margin: 0.2rem 0 0.35rem 0;
                color: #eff5ff;
            }

            .hint-icon {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 1rem;
                height: 1rem;
                border-radius: 999px;
                border: 1px solid rgba(160, 185, 225, 0.5);
                color: #aec5ed;
                font-size: 0.7rem;
                cursor: help;
                line-height: 1rem;
            }

            .hint-value {
                font-family: "DM Sans", "Segoe UI", sans-serif;
                font-size: 0.75rem;
                font-weight: 600;
                color: #a8bce0;
                background: rgba(95, 125, 178, 0.2);
                border: 1px solid rgba(151, 178, 224, 0.28);
                border-radius: 999px;
                padding: 0.1rem 0.44rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
