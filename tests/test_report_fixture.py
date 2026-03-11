from pathlib import Path
import unittest

import pandas as pd

from ibkr_analyzer import report_utils


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ibkr_sample_anonymized.csv"


class ReportFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = report_utils.parse_ibkr_report(FIXTURE_PATH.read_bytes())

    def test_fixture_sections_are_parsed(self) -> None:
        self.assertEqual(len(self.report.tables), 25)
        self.assertEqual(len(self.report.tables["Concentration"]), 7)
        self.assertEqual(len(self.report.tables["ESG"]), 2)
        self.assertEqual(len(self.report.tables["Risk Measures Benchmark Comparison"]), 2)

    def test_profile_and_period_are_extracted_from_fixture(self) -> None:
        profile, key_stats_row = report_utils.find_profile_info(self.report)
        start, end = report_utils.extract_report_period(self.report, profile)

        self.assertEqual(profile["BaseCurrency"], "SGD")
        self.assertEqual(profile["PerformanceMeasure"], "TWR")
        self.assertEqual(pd.Timestamp("2022-04-26"), start)
        self.assertEqual(pd.Timestamp("2026-02-25"), end)
        self.assertAlmostEqual(
            report_utils.parse_number(key_stats_row["EndingNAV"]),
            708812.1419,
            places=4,
        )

    def test_benchmark_tables_build_long_series(self) -> None:
        time_table = report_utils.get_table(
            self.report,
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

        self.assertFalse(normalized.empty)
        self.assertEqual(normalized["Series"].nunique(), 4)
        self.assertTrue(normalized["Date"].notna().all())
        self.assertTrue(normalized["Return"].notna().all())

    def test_projected_income_fixture_matches_dynamic_column_lookup(self) -> None:
        projected_income = report_utils.get_table(
            self.report,
            "Projected Income",
            required_columns=["Estimated Annual Income"],
        )

        remaining_column = report_utils.find_projected_remaining_income_column(
            projected_income.columns
        )
        total_row = projected_income.loc[projected_income["Symbol"] == "Total"].iloc[0]

        self.assertEqual(remaining_column, "Estimated 2026 Remaining Income")
        self.assertEqual(
            report_utils.remaining_income_metric_label(remaining_column),
            "Remaining 2026 Income",
        )
        self.assertAlmostEqual(
            report_utils.parse_number(total_row[remaining_column]),
            218.7645408,
            places=4,
        )

    def test_core_render_inputs_exist_in_fixture(self) -> None:
        open_positions = report_utils.get_table(
            self.report,
            "Open Position Summary",
            required_columns=["Date", "Symbol", "Description", "Value", "UnrealizedP&L"],
        )
        concentration = report_utils.get_table(
            self.report,
            "Concentration",
            required_columns=[
                "SubSection",
                "Symbol",
                "Description",
                "LongParsedWeight",
                "ShortParsedWeight",
                "NetParsedWeight",
            ],
        )
        symbol_perf = report_utils.get_table(
            self.report,
            "Performance by Symbol",
            required_columns=["Symbol", "Description", "Contribution", "Return", "AvgWeight"],
        )
        esg_holdings = report_utils.get_table(
            self.report,
            "ESG",
            required_columns=["SubSection", "Symbol", "Description", "Weight (%)", "ESG", "Combined"],
        )

        self.assertFalse(open_positions.empty)
        self.assertFalse(concentration.empty)
        self.assertFalse(symbol_perf.empty)
        self.assertFalse(esg_holdings.empty)


if __name__ == "__main__":
    unittest.main()
