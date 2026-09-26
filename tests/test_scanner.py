import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

import master_production_runner as scanner


SCRIPT = Path(scanner.__file__)


class ScannerFlowTests(unittest.TestCase):
    def test_offline_run_uses_repository_data_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env.pop("ODDS_API_KEY", None)
            result = subprocess.run(
                [sys.executable, str(SCRIPT)], cwd=directory, env=env,
                capture_output=True, text=True, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("K-Props (120), NRFI (15), F5 (15), Full (15)", result.stdout)
        self.assertIn("Full Game projections:", result.stdout)
        self.assertIn("Minnesota Twins @ Detroit Tigers", result.stdout)
        self.assertIn("live odds and recommendations skipped", result.stdout)

    def test_live_requires_key_before_loading_data(self):
        env = os.environ.copy()
        env.pop("ODDS_API_KEY", None)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--live"], env=env,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--live requires ODDS_API_KEY", result.stderr)

    def test_missing_input_reports_file(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--data-dir", directory],
                capture_output=True, text=True, check=False,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("pitcher_statcast_rolling.csv", result.stderr)

    def test_live_only_recommends_matched_teams_and_reports_api_failures(self):
        matchup = "Minnesota Twins @ Detroit Tigers"
        event = {
            "away_team": "Minnesota Twins", "home_team": "Detroit Tigers",
            "bookmakers": [{"key": "fanduel", "markets": [{"key": "h2h", "outcomes": [
                {"name": "Detroit Tigers", "price": 2.0},
                {"name": "New York Yankees", "price": 2.0},
            ]}]}],
        }
        response = Mock()
        response.json.return_value = [event]
        with patch.object(scanner, "run_full_game_ensemble") as model, \
             patch.object(scanner.requests, "get", return_value=response) as get, \
             contextlib.redirect_stdout(io.StringIO()):
            model.return_value = scanner.pd.DataFrame([{
                "matchup": matchup, "home_ml_prob": 0.60, "away_ml_prob": 0.40,
            }])
            bets = scanner.scan_live_market(odds_api_key="test-key")
        self.assertEqual(len(bets), 1)
        self.assertEqual(bets[0]["selection"], "Detroit Tigers")
        self.assertAlmostEqual(bets[0]["edge"], 0.10)
        self.assertEqual(get.call_args.kwargs["params"]["apiKey"], "test-key")

        with patch.object(scanner, "run_full_game_ensemble", return_value=model.return_value), \
             patch.object(scanner.requests, "get", side_effect=requests.Timeout), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Could not retrieve live odds"):
                scanner.scan_live_market(odds_api_key="test-key")


if __name__ == "__main__":
    unittest.main()
