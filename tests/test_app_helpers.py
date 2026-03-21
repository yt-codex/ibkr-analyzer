import unittest

import pandas as pd
from ibkr_analyzer import report_utils


class AppHelperTests(unittest.TestCase):
    def test_parse_analysis_period_text_handles_iso_dates(self) -> None:
        start, end = report_utils.parse_analysis_period_text("2024-01-01 - 2024-12-31")

        self.assertEqual(start, pd.Timestamp("2024-01-01"))
        self.assertEqual(end, pd.Timestamp("2024-12-31"))

    def test_parse_report_date_handles_month_year_labels(self) -> None:
        self.assertEqual(
            report_utils.parse_report_date("Apr-22"),
            pd.Timestamp("2022-04-01"),
        )
        self.assertEqual(
            report_utils.parse_report_date("Sept-22"),
            pd.Timestamp("2022-09-01"),
        )

    def test_sanitize_total_rows_preserves_legitimate_names(self) -> None:
        data_frame = pd.DataFrame(
            [
                {"Description": "TotalEnergies SE", "Symbol": "TTE"},
                {"Description": "Total", "Symbol": ""},
                {"Description": "Apple Inc.", "Symbol": "AAPL"},
            ]
        )

        filtered = report_utils.sanitize_total_rows(data_frame, "Description")

        self.assertEqual(filtered["Symbol"].tolist(), ["TTE", "AAPL"])

    def test_find_projected_remaining_income_column_uses_latest_year(self) -> None:
        column_name = report_utils.find_projected_remaining_income_column(
            [
                "Estimated 2026 Remaining Income",
                "Estimated 2028 Remaining Income",
                "Estimated 2027 Remaining Income",
            ]
        )

        self.assertEqual(column_name, "Estimated 2028 Remaining Income")

    def test_remaining_income_metric_label_reflects_selected_year(self) -> None:
        self.assertEqual(
            report_utils.remaining_income_metric_label("Estimated 2027 Remaining Income"),
            "Remaining 2027 Income",
        )

    def test_build_report_summary_html_escapes_untrusted_values(self) -> None:
        panel_html = report_utils.build_report_summary_html(
            report_source='Uploaded report: report<script>alert("x")</script>.csv',
            account_name="<b>Unsafe</b>",
            account_id="ACC-123",
            base_currency="USD",
            performance_measure="TWR",
            analysis_period="2024-01-01 - 2024-12-31",
            period_length_display="1.00 years",
            parsed_sections=7,
        )

        self.assertNotIn("<script>", panel_html)
        self.assertIn("&lt;script&gt;", panel_html)
        self.assertIn("&lt;b&gt;Unsafe&lt;/b&gt;", panel_html)

    def test_value_or_zero_coalesces_nan(self) -> None:
        self.assertEqual(
            report_utils.value_or_zero(float("nan")) + report_utils.value_or_zero(12.5),
            12.5,
        )

    def test_build_benchmark_long_accepts_partial_benchmark_set(self) -> None:
        benchmark_table = pd.DataFrame(
            {
                "Date": ["Jan-26", "Feb-26"],
                "BM1": ["SPXTR", "SPXTR"],
                "BM1Return": [1.2, 2.3],
                "Account": ["U0000000", "U0000000"],
                "AccountReturn": [0.8, 1.4],
            }
        )

        normalized = report_utils.build_benchmark_long(benchmark_table)

        self.assertEqual(sorted(normalized["Series"].unique().tolist()), ["SPXTR", "U0000000"])
        self.assertEqual(len(normalized), 4)

    def test_partial_year_label_year_uses_report_end_instead_of_wall_clock(self) -> None:
        self.assertEqual(
            report_utils.partial_year_label_year(pd.Timestamp("2026-02-25")),
            2026,
        )
        self.assertIsNone(
            report_utils.partial_year_label_year(pd.Timestamp("2026-12-31"))
        )

    def test_normalize_cashflow_amount_uses_transaction_type_hint(self) -> None:
        self.assertEqual(
            report_utils.normalize_cashflow_amount("WITHDRAWAL", 150.0),
            -150.0,
        )
        self.assertEqual(
            report_utils.normalize_cashflow_amount("DEPOSIT", -250.0),
            250.0,
        )


if __name__ == "__main__":
    unittest.main()
