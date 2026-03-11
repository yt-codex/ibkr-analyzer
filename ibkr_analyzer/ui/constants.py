from __future__ import annotations

PLOTLY_TEMPLATE = "plotly_dark"

_THEME_PRESETS = {
    "slate_mint": {
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
        "chart_colors": [
            "#e6a15a",
            "#8ab4f8",
            "#f2c14e",
            "#f87171",
            "#d9c6a5",
            "#c5d7f2",
        ],
        "chart": {
            "font_color": "#f5f1ea",
            "plot_bg": "rgba(35, 38, 44, 0.28)",
            "grid_color": "rgba(185, 174, 161, 0.12)",
            "marker_line": "rgba(26, 28, 32, 0.95)",
            "nav_fill": "rgba(230, 161, 90, 0.18)",
            "drawdown_fill": "rgba(248, 113, 113, 0.16)",
            "neutral": "#b9aea1",
            "diverging_scale": ["#f87171", "#b9aea1", "#e6a15a"],
            "positive_scale": ["#b9aea1", "#e6a15a"],
            "negative_scale": ["#ffd7cf", "#f87171"],
            "cashflow_scale": ["#f87171", "#b9aea1", "#e6a15a"],
            "capital_scale": ["#8ab4f8", "#e6a15a"],
            "risk_scale": ["#f87171", "#8ab4f8", "#e6a15a"],
            "heatmap_scale": [
                [0.0, "#7a3c35"],
                [0.45, "#413b40"],
                [0.5, "#52494e"],
                [0.55, "#7b624d"],
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


def get_chart_theme() -> dict[str, object]:
    return _THEME_PRESETS[_ACTIVE_THEME]["chart"]
