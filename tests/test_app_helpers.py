import sys
import types
import unittest

import pandas as pd


def _install_ui_stubs() -> None:
    plotly_module = types.ModuleType("plotly")
    plotly_express_module = types.ModuleType("plotly.express")
    plotly_graph_objects_module = types.ModuleType("plotly.graph_objects")
    streamlit_module = types.ModuleType("streamlit")

    plotly_module.express = plotly_express_module
    plotly_module.graph_objects = plotly_graph_objects_module

    sys.modules.setdefault("plotly", plotly_module)
    sys.modules.setdefault("plotly.express", plotly_express_module)
    sys.modules.setdefault("plotly.graph_objects", plotly_graph_objects_module)
    sys.modules.setdefault("streamlit", streamlit_module)


_install_ui_stubs()

import app


class AppHelperTests(unittest.TestCase):
    def test_parse_analysis_period_text_handles_iso_dates(self) -> None:
        start, end = app.parse_analysis_period_text("2024-01-01 - 2024-12-31")

        self.assertEqual(start, pd.Timestamp("2024-01-01"))
        self.assertEqual(end, pd.Timestamp("2024-12-31"))

    def test_sanitize_total_rows_preserves_legitimate_names(self) -> None:
        data_frame = pd.DataFrame(
            [
                {"Description": "TotalEnergies SE", "Symbol": "TTE"},
                {"Description": "Total", "Symbol": ""},
                {"Description": "Apple Inc.", "Symbol": "AAPL"},
            ]
        )

        filtered = app.sanitize_total_rows(data_frame, "Description")

        self.assertEqual(filtered["Symbol"].tolist(), ["TTE", "AAPL"])

    def test_find_projected_remaining_income_column_uses_latest_year(self) -> None:
        column_name = app.find_projected_remaining_income_column(
            [
                "Estimated 2026 Remaining Income",
                "Estimated 2028 Remaining Income",
                "Estimated 2027 Remaining Income",
            ]
        )

        self.assertEqual(column_name, "Estimated 2028 Remaining Income")

    def test_remaining_income_metric_label_reflects_selected_year(self) -> None:
        self.assertEqual(
            app.remaining_income_metric_label("Estimated 2027 Remaining Income"),
            "Remaining 2027 Income",
        )

    def test_build_report_summary_html_escapes_untrusted_values(self) -> None:
        panel_html = app.build_report_summary_html(
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
        self.assertEqual(app.value_or_zero(float("nan")) + app.value_or_zero(12.5), 12.5)


if __name__ == "__main__":
    unittest.main()
