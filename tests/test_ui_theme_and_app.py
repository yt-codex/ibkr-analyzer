import contextlib
import unittest
from unittest.mock import patch

import app

from ibkr_analyzer.ui.theme import inject_custom_css


class ThemeAndAppTests(unittest.TestCase):
    @patch("ibkr_analyzer.ui.theme.st.markdown")
    def test_editorial_theme_styles_streamlit_toolbar_icons(self, mock_markdown) -> None:
        inject_custom_css("editorial")

        css = mock_markdown.call_args.args[0]
        self.assertIn('[data-testid="stHeader"] [data-testid="stToolbar"] button', css)
        self.assertIn("fill: currentColor !important;", css)
        self.assertIn("stroke: currentColor !important;", css)

    def test_empty_state_requests_uploaded_report_only(self) -> None:
        with patch.object(app.st, "set_page_config"), patch.object(
            app.st, "session_state", {}
        ), patch.object(app.st, "sidebar", contextlib.nullcontext()), patch.object(
            app.st, "toggle", return_value=False
        ), patch.object(app.st, "markdown"), patch.object(
            app.st, "caption"
        ), patch.object(
            app.st, "file_uploader", return_value=None
        ), patch.object(
            app.st, "expander", return_value=contextlib.nullcontext()
        ), patch.object(
            app.st, "info"
        ) as mock_info:
            app.streamlit_app()

        mock_info.assert_called_once_with("Upload your IBKR CSV report to start analysis.")


if __name__ == "__main__":
    unittest.main()
