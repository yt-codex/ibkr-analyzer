from .chrome import render_section_intro
from .theme import inject_custom_css


def render_cashflow_income_tab(*args, **kwargs):
    from .tabs.cashflow import render_cashflow_income_tab as _render_cashflow_income_tab

    return _render_cashflow_income_tab(*args, **kwargs)


def render_concentration_tab(*args, **kwargs):
    from .tabs.concentration import render_concentration_tab as _render_concentration_tab

    return _render_concentration_tab(*args, **kwargs)


def render_holdings_tab(*args, **kwargs):
    from .tabs.holdings import render_holdings_tab as _render_holdings_tab

    return _render_holdings_tab(*args, **kwargs)


def render_overview_tab(*args, **kwargs):
    from .tabs.overview import render_overview_tab as _render_overview_tab

    return _render_overview_tab(*args, **kwargs)


def render_performance_tab(*args, **kwargs):
    from .tabs.performance import render_performance_tab as _render_performance_tab

    return _render_performance_tab(*args, **kwargs)


def render_raw_tables_tab(*args, **kwargs):
    from .tabs.raw_tables import render_raw_tables_tab as _render_raw_tables_tab

    return _render_raw_tables_tab(*args, **kwargs)


def render_risk_esg_tab(*args, **kwargs):
    from .tabs.risk_esg import render_risk_esg_tab as _render_risk_esg_tab

    return _render_risk_esg_tab(*args, **kwargs)
