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

    def test_teacher_origin_ignores_judges_and_ranks_unknown_above_european(self):
        origin = Data.teacher_origin
        listed = {"teacher_status": "listed"}
        t = lambda stage, o: {"stage": stage, "model_origin": o}
        self.assertEqual(origin(listed, [t("judge", "foreign"), t("posttraining", "own")]), "own")
        self.assertEqual(origin(listed, [t("posttraining", "unknown"), t("pretraining", "european")]), "unknown")
        self.assertEqual(origin(listed, [t("judge", "foreign")]), "judges only")
        self.assertEqual(origin({"teacher_status": "unknown"}, []), "unknown")

    def test_teacher_needs_a_source(self):
        self.d.teachers.append({"family_id": "alia", "stage": "posttraining", "model": "X-1",
                                "model_origin": "foreign", "purpose": ""})
        self.assertTrue(any("teacher:X-1" in e for e in self.d.validate()))

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
        self.assertEqual(self.model("eurollm-22b")["origin"], "scratch")
        self.assertEqual(self.model("eurollm-22b")["origin_scope"], "family")
        self.assertEqual(self.model("eurollm-9b")["origin"], "continued-own")
        self.assertEqual(self.model("eurollm-9b")["origin_scope"], "model")

    def test_category_a_needs_documented_data(self):
        # Velvet: trained on Leonardo, but training data undisclosed -> B (decided 2026-10-30).
        self.assertEqual(self.d.tier(self.model("velvet-14b")), "B")
        self.assertEqual(self.d.tier(self.model("alia-40b")), "A")

    def test_own_changeable_policy_allows_a_third_party_one_does_not(self):
        apertus = self.model("apertus-1.5-8b")
        self.assertEqual(apertus["terms_owner"], "own")
        self.assertEqual(self.d.tier(apertus), "A")
        third = {**apertus, "terms_owner": "third-party"}
        self.assertEqual(self.d.tier(third), "B")
        self.assertEqual(self.model("helium-1-2b")["terms_owner"], "third-party")

    def test_owner_in_europe_does_not_cap(self):
        m = {**self.model("alia-40b"), "control": "acquired-europe"}
        confirmed = self.d.confirmed  # as if a source backed the owner (e.g. Swisscom for Fastweb)
        self.d.confirmed = lambda mm, f: "acquired-europe" if f == "control" else confirmed(mm, f)
        self.assertEqual(self.d.tier(m), "A")
        self.d.confirmed = lambda mm, f: "acquired-non-europe" if f == "control" else confirmed(mm, f)
        self.assertEqual(self.d.tier(m), "B")
        self.d.confirmed = confirmed
        self.assertIn("owned outside the EU", Data.flags({**m, "control": "acquired-europe"}))

    def test_press_release_wire_is_not_press(self):
        self.d.sources.append(source(scope="provider", subject="mbzuai", field="control",
                                     url="https://www.prnewswire.com/news-releases/x.html", issuer="press"))
        self.assertTrue(any("press-release wire" in e for e in self.d.validate()))

    def test_publishers(self):
        kinds = {b["repo"]: self.d.publisher(b) for b in self.d.builds}
        self.assertEqual(kinds["speakleash/Bielik-11B-v3.0-Instruct-MLX-8bit"], "official")
        self.assertEqual(kinds["here-be-dragons-ai/Kolibri-1-MLX-3bit"], "hbd")
        self.assertEqual(kinds["mlx-community/EuroLLM-22B-Instruct-2512-mlx-8bit"], "curated")
        self.assertEqual(kinds["tonidurans/ALIA-40b-instruct-2605-mlx-q4"], "individual")


if __name__ == "__main__":
    unittest.main()
