from pathlib import Path
import unittest

import pandas as pd

from ibkr_analyzer import report_utils


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class ReportVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.missing_risk_esg_report = report_utils.parse_ibkr_report(
            (FIXTURES_DIR / "ibkr_sample_missing_risk_esg.csv").read_bytes()
        )
        cls.mwr_report = report_utils.parse_ibkr_report(
            (FIXTURES_DIR / "ibkr_sample_mwr_2027.csv").read_bytes()
        )

    def test_missing_risk_and_esg_sections_are_optional(self) -> None:
        profile, _ = report_utils.find_profile_info(self.missing_risk_esg_report)
        start, end = report_utils.extract_report_period(self.missing_risk_esg_report, profile)

        self.assertNotIn("ESG", self.missing_risk_esg_report.tables)
        self.assertNotIn(
            "Risk Measures Benchmark Comparison",
            self.missing_risk_esg_report.tables,
        )
        self.assertEqual(start, pd.Timestamp("2022-04-26"))
        self.assertEqual(end, pd.Timestamp("2026-02-25"))

    def test_mwr_fixture_uses_dynamic_remaining_income_year(self) -> None:
        profile, _ = report_utils.find_profile_info(self.mwr_report)
        projected_income = report_utils.get_table(
            self.mwr_report,
            "Projected Income",
            required_columns=["Estimated Annual Income"],
        )
        remaining_column = report_utils.find_projected_remaining_income_column(
            projected_income.columns
        )
        total_row = projected_income.loc[projected_income["Symbol"] == "Total"].iloc[0]
        method_label, method_tip = report_utils.return_method_label_and_tooltip(
            profile["PerformanceMeasure"]
        )

        self.assertEqual(profile["PerformanceMeasure"], "MWR")
        self.assertEqual(method_label, "MWR")
        self.assertIn("Money-Weighted Return", method_tip)
        self.assertEqual(remaining_column, "Estimated 2027 Remaining Income")
        self.assertEqual(
            report_utils.remaining_income_metric_label(remaining_column),
            "Remaining 2027 Income",
        )
        self.assertAlmostEqual(
            report_utils.parse_number(total_row[remaining_column]),
            240.0,
        )

    def test_mwr_fixture_benchmark_rows_build_series(self) -> None:
        time_table = report_utils.get_table(
            self.mwr_report,
            "Time Period Benchmark Comparison",
            required_columns=[
                "Date",
                "BM1",
                "BM1Return",
                "BM2",
                "BM2Return",
                "BM3",
                "BM3Return",
            ],
        )

        normalized = report_utils.build_benchmark_long(time_table)

        self.assertEqual(normalized["Series"].nunique(), 4)
        self.assertEqual(sorted(normalized["Date"].dt.strftime("%Y-%m-%d").unique().tolist()), ["2026-01-01", "2026-02-01"])


if __name__ == "__main__":
    unittest.main()
