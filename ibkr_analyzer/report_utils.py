from __future__ import annotations

import csv
import html
import io
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ParsedIBKRReport:
    tables: dict[str, list[pd.DataFrame]]
    metadata: dict[str, list[list[str]]]


NULL_MARKERS = {"", "-", "--", "N/A", "NA", "null", "None"}
TOTAL_ROW_PATTERN = re.compile(r"(?:grand\s+)?totals?|sub\s*total", re.IGNORECASE)


def make_unique_headers(headers: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique_headers: list[str] = []
    for index, header in enumerate(headers, start=1):
        cleaned = header.strip() if header else ""
        cleaned = cleaned or f"Column_{index}"
        counts[cleaned] = counts.get(cleaned, 0) + 1
        if counts[cleaned] > 1:
            cleaned = f"{cleaned}_{counts[cleaned]}"
        unique_headers.append(cleaned)
    return unique_headers


def parse_ibkr_report(file_bytes: bytes) -> ParsedIBKRReport:
    decoded = file_bytes.decode("utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(decoded))

    raw_tables: dict[str, list[dict[str, list]]] = defaultdict(list)
    metadata: dict[str, list[list[str]]] = defaultdict(list)
    active_index: dict[str, int] = {}

    for row in reader:
        if len(row) < 2:
            continue

        section = row[0].strip()
        row_type = row[1].strip()
        payload = [cell.strip() for cell in row[2:]]

        if row_type == "Header":
            raw_tables[section].append(
                {
                    "columns": make_unique_headers(payload),
                    "rows": [],
                }
            )
            active_index[section] = len(raw_tables[section]) - 1
            continue

        if row_type == "MetaInfo":
            metadata[section].append(payload)
            continue

        if row_type != "Data":
            continue

        if section not in active_index:
            raw_tables[section].append(
                {
                    "columns": make_unique_headers(
                        [f"Column_{idx}" for idx in range(1, len(payload) + 1)]
                    ),
                    "rows": [],
                }
            )
            active_index[section] = len(raw_tables[section]) - 1

        table = raw_tables[section][active_index[section]]
        columns = table["columns"]
        row_values = payload[:]

        if len(row_values) < len(columns):
            row_values.extend([""] * (len(columns) - len(row_values)))
        elif len(row_values) > len(columns):
            extra_columns_count = len(row_values) - len(columns)
            start = len(columns) + 1
            extra_columns = [
                f"Extra_{idx}" for idx in range(start, start + extra_columns_count)
            ]
            table["columns"].extend(extra_columns)
            for existing_row in table["rows"]:
                existing_row.extend([""] * extra_columns_count)

        table["rows"].append(row_values)

    tables: dict[str, list[pd.DataFrame]] = {}
    for section, section_tables in raw_tables.items():
        parsed_tables: list[pd.DataFrame] = []
        for section_table in section_tables:
            data_frame = pd.DataFrame(
                section_table["rows"], columns=section_table["columns"]
            )
            for column in data_frame.columns:
                data_frame[column] = data_frame[column].map(
                    lambda value: value.strip() if isinstance(value, str) else value
                )
            parsed_tables.append(data_frame)
        tables[section] = parsed_tables

    return ParsedIBKRReport(tables=tables, metadata=dict(metadata))


def get_table(
    report: ParsedIBKRReport,
    section: str,
    index: int = 0,
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    section_tables = report.tables.get(section, [])
    if not section_tables:
        return pd.DataFrame()

    if required_columns:
        required = set(required_columns)
        for table in section_tables:
            if required.issubset(table.columns):
                return table.copy()
        return pd.DataFrame()

    if index < 0 or index >= len(section_tables):
        return pd.DataFrame()
    return section_tables[index].copy()


def parse_number(value: object) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (float, int)):
        return float(value)

    text = str(value).strip()
    if text in NULL_MARKERS:
        return np.nan

    text = text.replace(",", "").replace("%", "")
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1]}"

    try:
        return float(text)
    except ValueError:
        return np.nan


def to_numeric(series: pd.Series) -> pd.Series:
    return series.map(parse_number)


def parse_report_date(value: object) -> pd.Timestamp:
    if value is None:
        return pd.NaT

    text = str(value).strip()
    if not text or text in NULL_MARKERS or text.lower() == "total":
        return pd.NaT

    normalized_text = re.sub(r"^Sept(?=-)", "Sep", text)
    for date_format in (
        "%Y%m%d",
        "%Y%m",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%b %Y",
        "%B %Y",
        "%b-%y",
        "%b-%Y",
        "%B-%y",
        "%B-%Y",
    ):
        parsed = pd.to_datetime(normalized_text, format=date_format, errors="coerce")
        if pd.notna(parsed):
            return parsed

    quarter_match = re.fullmatch(r"(\d{4})\s*Q([1-4])", text)
    if quarter_match:
        year = int(quarter_match.group(1))
        quarter = int(quarter_match.group(2))
        return pd.Timestamp(year=year, month=((quarter - 1) * 3) + 1, day=1)

    return pd.to_datetime(normalized_text, errors="coerce")


def parse_analysis_period_text(text: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    cleaned = re.sub(r"\(.*?\)", "", str(text)).strip()
    cleaned = cleaned.replace("–", " - ").replace("—", " - ")
    parts = re.split(r"\s+(?:to|-)\s+", cleaned, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return pd.NaT, pd.NaT

    start = pd.to_datetime(parts[0].strip(), errors="coerce")
    end = pd.to_datetime(parts[1].strip(), errors="coerce")
    return start, end


def period_years(start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    if pd.isna(start_date) or pd.isna(end_date):
        return np.nan
    days = (end_date - start_date).days
    if days <= 0:
        return np.nan
    return days / 365.25


def annualize_return(cumulative_return_pct: float, years: float) -> float:
    if pd.isna(cumulative_return_pct) or pd.isna(years) or years <= 0:
        return np.nan
    growth = 1 + (cumulative_return_pct / 100.0)
    if growth <= 0:
        return np.nan
    return ((growth ** (1 / years)) - 1) * 100


def extract_report_period(
    report: ParsedIBKRReport, profile: dict[str, str]
) -> tuple[pd.Timestamp, pd.Timestamp]:
    key_stats_meta = report.metadata.get("Key Statistics", [])
    for row in key_stats_meta:
        if len(row) >= 2 and "analysis period" in str(row[0]).lower():
            start, end = parse_analysis_period_text(row[1])
            if pd.notna(start) and pd.notna(end):
                return start, end

    profile_period = profile.get("AnalysisPeriod", "")
    if profile_period:
        start, end = parse_analysis_period_text(profile_period)
        if pd.notna(start) and pd.notna(end):
            return start, end

    allocation = get_table(report, "Allocation by Asset Class", required_columns=["Date"])
    if not allocation.empty:
        parsed_dates = allocation["Date"].map(parse_report_date).dropna().sort_values()
        if not parsed_dates.empty:
            return parsed_dates.iloc[0], parsed_dates.iloc[-1]

    return pd.NaT, pd.NaT


def return_method_label_and_tooltip(performance_measure: str) -> tuple[str, str]:
    normalized = str(performance_measure or "").strip().upper()
    if normalized == "TWR":
        return (
            "TWR",
            "IBKR PerformanceMeasure is TWR (Time-Weighted Return). "
            "Cumulative and annualized returns here are time-weighted and chain-linked.",
        )
    if normalized == "MWR":
        return (
            "MWR",
            "IBKR PerformanceMeasure is MWR (Money-Weighted Return). "
            "Cumulative and annualized returns here are cash-flow weighted.",
        )
    return (
        normalized or "Derived",
        "Performance method was not found in the report. "
        "Cumulative and annualized returns are derived from the report's return series.",
    )


def format_money(value: float, currency: str = "") -> str:
    if pd.isna(value):
        return "-"
    prefix = f"{currency} " if currency else ""
    return f"{prefix}{value:,.2f}"


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.2f}%"


def format_panel_value(value: object, default: str = "-") -> str:
    text = str(value).strip() if value is not None else ""
    return html.escape(text) if text else default


def value_or_zero(value: float) -> float:
    return 0.0 if pd.isna(value) else float(value)


def find_projected_remaining_income_column(columns: pd.Index | list[str]) -> str | None:
    candidates: list[tuple[int, str]] = []
    for column in columns:
        column_name = str(column).strip()
        if re.fullmatch(
            r"Estimated \d{4} Remaining Income", column_name, flags=re.IGNORECASE
        ):
            year_match = re.search(r"(\d{4})", column_name)
            year = int(year_match.group(1)) if year_match else -1
            candidates.append((year, column_name))
        elif re.fullmatch(
            r"Estimated Remaining Income", column_name, flags=re.IGNORECASE
        ):
            candidates.append((-1, column_name))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def remaining_income_metric_label(column_name: str | None) -> str:
    if not column_name:
        return "Remaining Income"
    year_match = re.search(r"(\d{4})", column_name)
    if not year_match:
        return "Remaining Income"
    return f"Remaining {year_match.group(1)} Income"


def build_report_summary_html(
    report_source: str,
    account_name: str,
    account_id: str,
    base_currency: str,
    performance_measure: str,
    analysis_period: str,
    period_length_display: str,
    parsed_sections: int,
) -> str:
    account_suffix = (
        f" ({format_panel_value(account_id, default='')})"
        if str(account_id).strip()
        else ""
    )
    return f"""
        <div class="panel">
            <b>{format_panel_value(report_source)}</b><br/>
            Account: <b>{format_panel_value(account_name, default="Unknown")}</b>{account_suffix}<br/>
            Base Currency: <b>{format_panel_value(base_currency)}</b><br/>
            Return Measure: <b>{format_panel_value(performance_measure)}</b><br/>
            Analysis Period: <b>{format_panel_value(analysis_period)}</b><br/>
            Period Length: <b>{format_panel_value(period_length_display)}</b><br/>
            Parsed Sections: <b>{parsed_sections}</b>
        </div>
        """


def sanitize_total_rows(
    data_frame: pd.DataFrame, column_name: str, drop_blank: bool = False
) -> pd.DataFrame:
    filtered = data_frame.copy()
    if column_name not in filtered.columns:
        return filtered

    normalized = filtered[column_name].astype(str).str.strip().str.rstrip(":")
    mask = normalized.str.fullmatch(TOTAL_ROW_PATTERN, na=False)
    filtered = filtered.loc[~mask]
    if drop_blank:
        filtered = filtered.loc[normalized != ""]
    return filtered


def build_benchmark_long(data_frame: pd.DataFrame) -> pd.DataFrame:
    required = {"Date", "BM1", "BM1Return", "BM2", "BM2Return", "BM3", "BM3Return"}
    if data_frame.empty or not required.issubset(set(data_frame.columns)):
        return pd.DataFrame()

    working = data_frame.copy()
    working["DateParsed"] = working["Date"].map(parse_report_date)

    series_pairs: list[tuple[str, str]] = []
    for benchmark_col, return_col in (
        ("BM1", "BM1Return"),
        ("BM2", "BM2Return"),
        ("BM3", "BM3Return"),
    ):
        benchmark_name_series = (
            working[benchmark_col].replace("", np.nan).dropna()
            if benchmark_col in working.columns
            else pd.Series(dtype=object)
        )
        benchmark_name = (
            str(benchmark_name_series.iloc[0])
            if not benchmark_name_series.empty
            else benchmark_col
        )
        if return_col in working.columns:
            series_pairs.append((benchmark_name, return_col))

    known_columns = {
        "Date",
        "DateParsed",
        "BM1",
        "BM1Return",
        "BM2",
        "BM2Return",
        "BM3",
        "BM3Return",
    }
    extra_columns = [column for column in working.columns if column not in known_columns]
    account_return_column = ""
    account_name = ""

    return_like_columns = [
        column for column in extra_columns if column.lower().endswith("return")
    ]
    if return_like_columns:
        account_return_column = return_like_columns[0]
        account_name_candidates = [
            column for column in extra_columns if column != account_return_column
        ]
        if account_name_candidates:
            account_series = (
                working[account_name_candidates[0]].replace("", np.nan).dropna()
            )
            if not account_series.empty:
                account_name = str(account_series.iloc[0])
        if not account_name:
            account_name = account_return_column.replace("Return", "").strip() or "Portfolio"
    elif extra_columns:
        account_return_column = extra_columns[-1]
        account_name = account_return_column.replace("Return", "").strip() or "Portfolio"

    if account_return_column and account_return_column in working.columns:
        series_pairs.append((account_name, account_return_column))

    normalized_frames: list[pd.DataFrame] = []
    for series_name, return_column in series_pairs:
        normalized = pd.DataFrame(
            {
                "Date": working["DateParsed"],
                "Series": series_name,
                "Return": to_numeric(working[return_column]),
            }
        )
        normalized = normalized.dropna(subset=["Date", "Return"])
        normalized_frames.append(normalized)

    if not normalized_frames:
        return pd.DataFrame()
    return pd.concat(normalized_frames, ignore_index=True).sort_values("Date")


def find_profile_info(report: ParsedIBKRReport) -> tuple[dict[str, str], pd.Series]:
    profile: dict[str, str] = {}
    intro_table = get_table(report, "Introduction")
    if not intro_table.empty:
        intro_row = intro_table.iloc[0]
        for column in intro_table.columns:
            profile[column] = str(intro_row[column])

    key_stats_table = get_table(report, "Key Statistics")
    key_stats_row = (
        key_stats_table.iloc[0] if not key_stats_table.empty else pd.Series(dtype=object)
    )
    return profile, key_stats_row
