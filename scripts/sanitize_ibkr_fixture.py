from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ACCOUNT_PATTERN = re.compile(r"^U\d+$")
ACCOUNT_RETURN_PATTERN = re.compile(r"^(U\d+)(Return)$")
SUMMARY_ROW_PATTERN = re.compile(
    r"(?:(?:grand\s+)?totals?|sub\s*total)(?::)?(?:\s|$)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanitize an IBKR Portfolio Analyst CSV for committed test fixtures."
    )
    parser.add_argument("source", type=Path, help="Path to the raw IBKR CSV export.")
    parser.add_argument("target", type=Path, help="Path to write the sanitized fixture.")
    parser.add_argument(
        "--max-data-rows-per-table",
        type=int,
        default=6,
        help="Maximum non-total data rows to preserve per table.",
    )
    return parser.parse_args()


def sanitize_header_cell(cell: str) -> str:
    text = cell.strip()
    if ACCOUNT_PATTERN.fullmatch(text):
        return "Account"
    match = ACCOUNT_RETURN_PATTERN.fullmatch(text)
    if match:
        return "AccountReturn"
    return cell


def is_summary_row_value(value: str) -> bool:
    return bool(SUMMARY_ROW_PATTERN.match(value.strip()))


def anonymize_value(
    header: str,
    value: str,
    symbol_map: dict[str, str],
    description_map: dict[str, str],
    account_map: dict[str, str],
) -> str:
    text = value.strip()
    if not text or text in {"-", "--"}:
        return value

    if header == "Name":
        return "ACCOUNT HOLDER"
    if header == "Account" and ACCOUNT_PATTERN.fullmatch(text):
        return "U0000000"
    if header == "Alias":
        return ""
    if header == "Symbol":
        if is_summary_row_value(text):
            return value
        return symbol_map.setdefault(text, f"SYM{len(symbol_map) + 1:03d}")
    if header == "Description":
        if is_summary_row_value(text):
            return value
        return description_map.setdefault(text, f"Description {len(description_map) + 1:03d}")
    if header == "Account" and text:
        return account_map.setdefault(text, "ACCOUNT")
    if header in {"BM1", "BM2", "BM3"}:
        return value
    return value


def sanitize_fixture(
    source: Path,
    target: Path,
    max_data_rows_per_table: int,
) -> None:
    symbol_map: dict[str, str] = {}
    description_map: dict[str, str] = {}
    account_map: dict[str, str] = {}
    headers_for_table: dict[tuple[str, int], list[str]] = {}
    section_table_index: dict[str, int] = {}
    data_counts: dict[tuple[str, int], int] = {}

    target.parent.mkdir(parents=True, exist_ok=True)

    with source.open("r", encoding="utf-8-sig", errors="replace", newline="") as input_file, target.open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        for row in reader:
            if len(row) < 2:
                continue

            section = row[0].strip()
            row_type = row[1].strip()
            table_index = section_table_index.get(section, -1)

            if row_type == "Header":
                section_table_index[section] = section_table_index.get(section, -1) + 1
                table_index = section_table_index[section]
                sanitized_header = [sanitize_header_cell(cell) for cell in row[2:]]
                headers_for_table[(section, table_index)] = sanitized_header
                data_counts[(section, table_index)] = 0
                writer.writerow([section, row_type, *sanitized_header])
                continue

            if row_type == "MetaInfo":
                writer.writerow(row)
                continue

            if row_type != "Data" or table_index < 0:
                continue

            headers = headers_for_table.get((section, table_index), [])
            payload = row[2:]
            include_row = data_counts[(section, table_index)] < max_data_rows_per_table or any(
                is_summary_row_value(str(cell)) for cell in payload
            )
            data_counts[(section, table_index)] += 1
            if not include_row:
                continue

            sanitized_payload = []
            for index, value in enumerate(payload):
                header = headers[index].strip() if index < len(headers) else ""
                sanitized_payload.append(
                    anonymize_value(
                        header=header,
                        value=value,
                        symbol_map=symbol_map,
                        description_map=description_map,
                        account_map=account_map,
                    )
                )
            writer.writerow([section, row_type, *sanitized_payload])


def main() -> None:
    args = parse_args()
    sanitize_fixture(
        source=args.source,
        target=args.target,
        max_data_rows_per_table=args.max_data_rows_per_table,
    )


if __name__ == "__main__":
    main()
