import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app
from src.features import save_processed_window


def make_cycles() -> list[list[float]]:
    rows = []
    for cycle in range(50):
        settings = [0.2 + cycle * 0.002, 0.1 + cycle * 0.001, 0.3 + cycle * 0.0015]
        sensors = [0.05 * index + cycle * 0.01 for index in range(1, 22)]
        normalized_cycle = [cycle / 49]
        rows.append(settings + sensors + normalized_cycle)
    return rows


class PredictiveMaintenanceApiTests(unittest.TestCase):
    def test_predict_returns_failure_probability_and_rul(self) -> None:
        with TestClient(app) as client:
            response = client.post("/predict", json={"asset_id": "engine-1", "cycles": make_cycles()})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("failure_probability", payload)
        self.assertGreaterEqual(payload["failure_probability"], 0.0)
        self.assertLessEqual(payload["failure_probability"], 1.0)
        self.assertGreater(payload["predicted_rul"], 0)
        self.assertTrue(payload["top_contributors"])

    def test_predict_rejects_invalid_window_shape(self) -> None:
        with TestClient(app) as client:
            response = client.post("/predict", json={"cycles": make_cycles()[:49]})

        self.assertEqual(response.status_code, 422)

    def test_reports_endpoint_serves_default_artifacts(self) -> None:
        with TestClient(app) as client:
            listing = client.get("/reports")
            html_report = client.get("/reports/drift_report.html")
            png_report = client.get("/reports/shap_summary.png")

        self.assertEqual(listing.status_code, 200)
        self.assertIn("drift_report.html", listing.json()["reports"])
        self.assertEqual(html_report.status_code, 200)
        self.assertEqual(html_report.headers["content-type"].split(";")[0], "text/html")
        self.assertEqual(png_report.status_code, 200)
        self.assertEqual(png_report.headers["content-type"], "image/png")

    def test_save_processed_window_writes_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            output = save_processed_window(make_cycles(), Path(tempdir) / "engine.parquet")
            self.assertTrue(output.exists())
            self.assertEqual(output.suffix, ".parquet")


if __name__ == "__main__":
    unittest.main()
