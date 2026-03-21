import unittest
from unittest.mock import patch
import sys
import types

import pandas as pd

from ibkr_analyzer.report_utils import ParsedIBKRReport


class _DummyFigure:
    def update_layout(self, *args, **kwargs) -> None:
        return None

    def update_xaxes(self, *args, **kwargs) -> None:
        return None

    def update_yaxes(self, *args, **kwargs) -> None:
        return None

    def update_traces(self, *args, **kwargs) -> None:
        return None


dummy_plotly = types.ModuleType("plotly")
dummy_express = types.ModuleType("plotly.express")
dummy_express.bar = lambda *args, **kwargs: _DummyFigure()
dummy_express.scatter = lambda *args, **kwargs: _DummyFigure()

sys.modules.setdefault("plotly", dummy_plotly)
sys.modules.setdefault("plotly.express", dummy_express)

from ibkr_analyzer.ui.tabs.risk_esg import render_risk_esg_tab


class RiskEsgTabTests(unittest.TestCase):
    @patch("ibkr_analyzer.ui.tabs.risk_esg.render_dataframe")
    @patch("ibkr_analyzer.ui.tabs.risk_esg.st.subheader")
    @patch("ibkr_analyzer.ui.tabs.risk_esg.st.plotly_chart")
    @patch("ibkr_analyzer.ui.tabs.risk_esg.st.info")
    def test_risk_tab_accepts_single_benchmark_reports(
        self,
        mock_info,
        mock_plotly_chart,
        _mock_subheader,
        _mock_render_dataframe,
    ) -> None:
        absolute_risk = pd.DataFrame(
            {
                "Risk Measure": ["Sharpe Ratio", "Max Drawdown"],
                "BM1": ["SPXTR", "SPXTR"],
                "BM1 Value": [1.15, -9.5],
                "Account": ["U0000000", "U0000000"],
                "Account Value": [0.98, -11.2],
            }
        )
        relative_risk = pd.DataFrame(
            {
                "Risk Measure Relative to Benchmark": ["Tracking Error"],
                "BM1": ["SPXTR"],
                "BM1 Value": [4.2],
            }
        )
        report = ParsedIBKRReport(
            tables={
                "Risk Measures Benchmark Comparison": [absolute_risk, relative_risk],
            },
            metadata={},
        )

        render_risk_esg_tab(report)

        self.assertTrue(mock_plotly_chart.called)
        info_messages = [call.args[0] for call in mock_info.call_args_list]
        self.assertNotIn("Absolute risk measures section not found.", info_messages)


if __name__ == "__main__":
    unittest.main()
