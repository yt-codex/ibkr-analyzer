from __future__ import annotations

_THEME_PRESETS = {
    "slate_mint": {
        "plotly_template": "plotly_dark",
        "chart_colors": [
            "#63e6be",
            "#7cb8ff",
            "#ffb454",
            "#ff6b6b",
            "#4fc3a1",
            "#b8c6da",
        ],
        "chart": {
            "font_color": "#f3f7fb",
            "plot_bg": "rgba(15, 27, 45, 0.24)",
            "grid_color": "rgba(159, 176, 199, 0.10)",
            "marker_line": "rgba(15, 27, 45, 0.95)",
            "nav_fill": "rgba(99, 230, 190, 0.14)",
            "drawdown_fill": "rgba(255, 107, 107, 0.14)",
            "neutral": "#9fb0c7",
            "diverging_scale": ["#ff6b6b", "#9fb0c7", "#63e6be"],
            "positive_scale": ["#9fb0c7", "#63e6be"],
            "negative_scale": ["#ffc2c2", "#ff6b6b"],
            "cashflow_scale": ["#ff6b6b", "#9fb0c7", "#63e6be"],
            "capital_scale": ["#7cb8ff", "#63e6be"],
            "risk_scale": ["#ff6b6b", "#7cb8ff", "#63e6be"],
            "heatmap_scale": [
                [0.0, "#7a3131"],
                [0.45, "#22354d"],
                [0.5, "#2b3f58"],
                [0.55, "#396c75"],
                [1.0, "#63e6be"],
            ],
        },
    },
    "editorial": {
        "plotly_template": "plotly_white",
        "chart_colors": [
            "#e6a15a",
            "#2f7f88",
            "#c05a3d",
            "#4b6cb7",
            "#8e3b46",
            "#9b8f81",
            "#d8c2a8",
        ],
        "chart": {
            "font_color": "#1f2328",
            "plot_bg": "#f7f1e7",
            "grid_color": "rgba(65, 59, 52, 0.12)",
            "marker_line": "rgba(232, 222, 208, 0.95)",
            "nav_fill": "rgba(230, 161, 90, 0.22)",
            "drawdown_fill": "rgba(192, 90, 61, 0.18)",
            "neutral": "#9b8f81",
            "diverging_scale": ["#c05a3d", "#d4c7b8", "#2f7f88"],
            "positive_scale": ["#d4c7b8", "#2f7f88"],
            "negative_scale": ["#e7b9ae", "#c05a3d"],
            "cashflow_scale": ["#c05a3d", "#d4c7b8", "#2f7f88"],
            "capital_scale": ["#4b6cb7", "#e6a15a"],
            "risk_scale": ["#c05a3d", "#4b6cb7", "#2f7f88"],
            "heatmap_scale": [
                [0.0, "#a24e3a"],
                [0.45, "#d9cfc0"],
                [0.5, "#efe6d9"],
                [0.55, "#b8c7c3"],
                [1.0, "#e6a15a"],
            ],
        },
    },
}

_ACTIVE_THEME = "slate_mint"

CHART_COLORS = list(_THEME_PRESETS[_ACTIVE_THEME]["chart_colors"])


def set_active_theme(theme_name: str) -> str:
    global _ACTIVE_THEME

    if theme_name not in _THEME_PRESETS:
        theme_name = "slate_mint"

    _ACTIVE_THEME = theme_name
    CHART_COLORS[:] = _THEME_PRESETS[theme_name]["chart_colors"]
    return _ACTIVE_THEME


def get_active_theme() -> str:
    return _ACTIVE_THEME


def get_plotly_template() -> str:
    return str(_THEME_PRESETS[_ACTIVE_THEME]["plotly_template"])


def get_chart_theme() -> dict[str, object]:
    return _THEME_PRESETS[_ACTIVE_THEME]["chart"]
