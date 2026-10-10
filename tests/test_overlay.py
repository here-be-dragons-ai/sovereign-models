"""The local overlay (data-global/) is merged only on request and never published.

    python3 -m unittest discover tests
"""

import csv
import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import scorecard  # noqa: E402


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


class TestOverlay(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.overlay = Path(self.tmp.name)
        public = scorecard._read(scorecard.ROOT / "data" / "scorecard.csv")
        self.public_ids = {r["id"] for r in public}
        self.row = {**{k: "" for k in public[0]}, "id": "overlay-only", "name": "Overlay"}

    def tearDown(self):
        os.environ.pop("SM_OVERLAY", None)
        importlib.reload(scorecard)
        self.tmp.cleanup()

    def reload_with_overlay(self):
        os.environ["SM_OVERLAY"] = str(self.overlay)
        return importlib.reload(scorecard)

    def test_without_overlay_only_public_rows(self):
        ids = {r["id"] for r in scorecard.read("scorecard.csv")}
        self.assertEqual(ids, self.public_ids)

    def test_overlay_rows_are_added(self):
        write(self.overlay / "scorecard.csv", [self.row])
        sc = self.reload_with_overlay()
        ids = {r["id"] for r in sc.read("scorecard.csv")}
        self.assertEqual(ids, self.public_ids | {"overlay-only"})

    def test_overlay_may_not_repeat_a_public_id(self):
        write(self.overlay / "scorecard.csv", [{**self.row, "id": sorted(self.public_ids)[0]}])
        sc = self.reload_with_overlay()
        with self.assertRaises(SystemExit):
            sc.read("scorecard.csv")

    def test_publish_refuses_overlay_models(self):
        dist = {"models": [{"id": i} for i in self.public_ids] + [{"id": "overlay-only"}]}
        with self.assertRaises(SystemExit):
            scorecard.publishable(dist, scorecard.ROOT / "data" / "scorecard.csv")

    def test_publish_refuses_when_overlay_is_set(self):
        dist = {"models": [{"id": i} for i in self.public_ids]}
        scorecard.publishable(dist, scorecard.ROOT / "data" / "scorecard.csv")
        os.environ["SM_OVERLAY"] = str(self.overlay)
        with self.assertRaises(SystemExit):
            scorecard.publishable(dist, scorecard.ROOT / "data" / "scorecard.csv")


if __name__ == "__main__":
    unittest.main()
