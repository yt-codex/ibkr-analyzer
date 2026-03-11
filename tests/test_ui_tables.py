import unittest
from unittest.mock import patch

import pandas as pd

from ibkr_analyzer.ui.constants import set_active_theme
from ibkr_analyzer.ui.tables import render_dataframe


class RenderDataframeTests(unittest.TestCase):
    def tearDown(self) -> None:
        set_active_theme("slate_mint")

    @patch("ibkr_analyzer.ui.tables.st.dataframe")
    def test_default_theme_omits_height_when_unspecified(self, mock_dataframe) -> None:
        set_active_theme("slate_mint")

        render_dataframe(pd.DataFrame({"Symbol": ["AAPL"]}))

        _, kwargs = mock_dataframe.call_args
        self.assertNotIn("height", kwargs)

    @patch("ibkr_analyzer.ui.tables.st.dataframe")
    def test_default_theme_passes_explicit_height(self, mock_dataframe) -> None:
        set_active_theme("slate_mint")

        render_dataframe(pd.DataFrame({"Symbol": ["AAPL"]}), height=480)

        _, kwargs = mock_dataframe.call_args
        self.assertEqual(kwargs["height"], 480)

    @patch("ibkr_analyzer.ui.tables.st.markdown")
    def test_editorial_theme_ignores_auto_height_in_inline_style(self, mock_markdown) -> None:
        set_active_theme("editorial")

        render_dataframe(pd.DataFrame({"Symbol": ["AAPL"]}), height="auto")

        html = mock_markdown.call_args.args[0]
        self.assertNotIn("autopx", html)


if __name__ == "__main__":
    unittest.main()
