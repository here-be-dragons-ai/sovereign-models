"""Checks of the tier rule and the issuer validation (standard library only).

    python3 -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from scorecard import Data  # noqa: E402


def source(**kw):
    base = dict(scope="family", subject="alia", field="data", url="", issuer="provider",
                checked="yes", retrieved="2026-10-08", archive_url="", note="")
    return {**base, **kw}


class TestData(unittest.TestCase):
    def setUp(self):
        self.d = Data()

    def model(self, mid):
        return next(m for m in self.d.resolved() if m["id"] == mid)

    def test_current_data_is_valid(self):
        self.assertEqual(self.d.validate(), [])

    def test_community_url_cannot_be_a_provider_source(self):
        self.d.sources.append(source(url="https://huggingface.co/tonidurans/ALIA-40b-instruct-2605-mlx-q4"))
        self.assertTrue(any("URL says 'community'" in e for e in self.d.validate()))

    def test_press_cannot_be_official(self):
        self.d.sources.append(source(scope="provider", subject="aleph-alpha", field="control",
                                     url="https://siliconangle.com/x", issuer="official"))
        self.assertTrue(any("URL says 'press'" in e for e in self.d.validate()))

    def test_provider_namespace_must_be_provider(self):
        self.d.sources.append(source(url="https://huggingface.co/BSC-LT/ALIA-40b-instruct-2606", issuer="community"))
        self.assertTrue(any("URL says 'provider'" in e for e in self.d.validate()))

    def test_unchecked_sources_do_not_count(self):
        m = self.model("luciole-23b")
        self.assertEqual(self.d.tier(m), "A")
        for s in self.d.sources:
            if s["subject"] == "luciole-1.1" and s["field"] in ("data", "compute"):
                s["checked"] = "no"
        self.assertEqual(self.d.tier(m), "B")

    def test_community_sources_do_not_count(self):
        m = self.model("luciole-23b")
        for s in self.d.sources:
            if s["subject"] == "luciole-1.1" and s["field"] in ("data", "compute"):
                s["issuer"] = "community"
        self.assertEqual(self.d.tier(m), "B")

    def test_family_values_are_inherited_and_overridable(self):
        self.assertEqual(self.model("eurollm-9b")["compute"], "unknown")
        self.assertEqual(self.model("eurollm-22b")["compute"], "eu")
        self.assertEqual(self.model("eurollm-22b")["compute_scope"], "model")

    def test_publishers(self):
        kinds = {b["repo"]: self.d.publisher(b) for b in self.d.builds}
        self.assertEqual(kinds["speakleash/Bielik-11B-v3.0-Instruct-MLX-8bit"], "official")
        self.assertEqual(kinds["here-be-dragons-ai/Kolibri-1-MLX-3bit"], "hbd")
        self.assertEqual(kinds["mlx-community/EuroLLM-22B-Instruct-2512-mlx-8bit"], "curated")
        self.assertEqual(kinds["tonidurans/ALIA-40b-instruct-2605-mlx-q4"], "individual")


if __name__ == "__main__":
    unittest.main()
