import unittest

from scripts import sanitize_ibkr_fixture


class SanitizeFixtureScriptTests(unittest.TestCase):
    def test_summary_row_detection_matches_totals_but_not_legitimate_names(self) -> None:
        self.assertTrue(sanitize_ibkr_fixture.is_summary_row_value("Total"))
        self.assertTrue(sanitize_ibkr_fixture.is_summary_row_value("Total Sector"))
        self.assertTrue(sanitize_ibkr_fixture.is_summary_row_value("Grand Total"))
        self.assertFalse(sanitize_ibkr_fixture.is_summary_row_value("TOTALENERGIES SE"))

    def test_anonymize_value_redacts_non_summary_descriptions(self) -> None:
        redacted = sanitize_ibkr_fixture.anonymize_value(
            header="Description",
            value="TOTALENERGIES SE",
            symbol_map={},
            description_map={},
            account_map={},
        )

        self.assertEqual(redacted, "Description 001")


if __name__ == "__main__":
    unittest.main()
