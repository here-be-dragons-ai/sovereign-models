"""Checks of the rubric rules, the quote matching and the evidence validation.

    python3 -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import rubric  # noqa: E402
from scorecard import Data  # noqa: E402


def ans(question, answer, url="https://example.org/card", quote="q"):
    return {"scope": "family", "subject": "alia", "question": question, "answer": answer,
            "url": url, "quote": quote, "retrieved": "2026-10-29", "by": "test"}


class TestDerive(unittest.TestCase):
    def derive(self, field, *pairs):
        return rubric.derive(field, [ans(q, a) for q, a in pairs])

    def test_origin(self):
        self.assertEqual(self.derive("origin", ("init_random", "yes")), "scratch")
        base = [("base_model", "Llama 3.1 8B"), ("base_publisher", "foreign")]
        self.assertEqual(self.derive("origin", *base, ("post_training_only", "no")), "continued-foreign")
        self.assertEqual(self.derive("origin", *base, ("post_training_only", "yes")), "finetune-foreign")
        # Whether a foreign or European base was only fine-tuned must be stated, not assumed.
        self.assertIsNone(self.derive("origin", ("base_model", "Mistral 7B"), ("base_publisher", "european"),
                                      ("post_training_only", "not stated")))
        self.assertEqual(self.derive("origin", ("base_model", "Apertus 1"), ("base_publisher", "own")),
                         "continued-own")
        self.assertEqual(self.derive("origin", ("init_random", "not stated")), "unknown")
        self.assertIsNone(self.derive("origin"))

    def test_data(self):
        self.assertEqual(self.derive("data", ("data_downloadable", "nearly all")), "open")
        self.assertEqual(self.derive("data", ("datasets_named", "yes")), "documented")
        self.assertEqual(self.derive("data", ("datasets_named", "no"), ("data_shares", "no")), "undisclosed")
        self.assertIsNone(self.derive("data", ("datasets_named", "no")))

    def test_compute_takes_the_least_european_site(self):
        self.assertEqual(self.derive("compute", ("pretraining_site", "Leonardo")), "eu")
        self.assertEqual(self.derive("compute", ("pretraining_site", "Alps")), "europe")
        self.assertEqual(self.derive("compute", ("pretraining_site", "LUMI"), ("pretraining_site", "US")),
                         "non-europe")
        self.assertEqual(self.derive("compute", ("pretraining_site", "Leonardo"),
                                     ("pretraining_site", "not stated")), "unknown")
        self.assertIsNone(rubric.region_of("Some Cloud"))

    def test_licence(self):
        clean = [("licence_modified", "no"), ("inherited_terms", "no"), ("noncommercial", "no")]
        self.assertEqual(self.derive("license_class", ("licence_text", "apache-2.0"), *clean,
                                     ("use_policy_binding", "no")), "osi")
        self.assertEqual(self.derive("license_class", ("licence_text", "apache-2.0"), *clean,
                                     ("use_policy_binding", "yes")), "osi-aup")
        self.assertEqual(self.derive("license_class", ("licence_text", "mit"), ("licence_modified", "yes")),
                         "custom")
        self.assertEqual(self.derive("license_class", ("licence_text", "mit"), ("inherited_terms", "yes")),
                         "custom")
        self.assertEqual(self.derive("license_class", ("noncommercial", "yes")), "nc")
        # An open licence text alone does not decide: the extra conditions must be answered.
        self.assertIsNone(self.derive("license_class", ("licence_text", "apache-2.0")))


class TestQuotes(unittest.TestCase):
    def test_normalised_and_loose_matches(self):
        doc = "We trained it from\nscratch on 13 trillion tokens — on Leonardo’s A100s.\n| **Developer:** | Almawave |"
        self.assertEqual(rubric.find_quote("trained it from scratch on 13 trillion tokens", doc), "found")
        self.assertEqual(rubric.find_quote("tokens - on Leonardo's A100s", doc), "found")
        self.assertEqual(rubric.find_quote("Developer: Almawave", doc), "found (loose)")
        self.assertEqual(rubric.find_quote("We trained it ... Leonardo", doc), "found")
        self.assertEqual(rubric.find_quote("trained on LUMI", doc), "not found")

    def test_parts_must_appear_in_order(self):
        self.assertEqual(rubric.find_quote("Leonardo ... trained it", "We trained it on Leonardo."), "not found")

    def test_absence_needs_no_quote_but_a_claim_does(self):
        self.assertFalse(rubric.needs_quote("datasets_named", "no"))
        self.assertTrue(rubric.needs_quote("datasets_named", "yes"))
        self.assertFalse(rubric.needs_quote("pretraining_site", "not stated"))
        self.assertTrue(rubric.needs_quote("licence_text", "apache-2.0"))

    def test_unverified_quote_does_not_count(self):
        row = ans("init_random", "yes", quote="trained from scratch")
        self.assertEqual(rubric.usable([row], {}), [])
        key = rubric.quote_key(row["url"], row["quote"])
        self.assertEqual(rubric.usable([row], {key: "found"}), [row])
        self.assertEqual(rubric.usable([row], {key: "not found"}), [])


class TestEvidenceValidation(unittest.TestCase):
    def setUp(self):
        self.d = Data()

    def test_bad_answer_is_an_error(self):
        self.d.evidence.append(ans("datasets_named", "maybe"))
        self.assertTrue(any("datasets_named='maybe'" in e for e in self.d.validate()))

    def test_claim_without_quote_is_an_error(self):
        self.d.evidence.append(ans("init_random", "yes", quote=""))
        self.assertTrue(any("needs a verbatim quote" in e for e in self.d.validate()))

    def test_rubric_must_agree_with_the_curated_value(self):
        # alia is curated as trained from scratch; a verified answer naming a foreign base contradicts it.
        rows = [ans("base_model", "Llama 3.1 8B", quote="based on Llama 3.1 8B"), ans("base_publisher", "foreign"),
                ans("post_training_only", "no", quote="continued pre-training")]
        self.d.evidence += rows
        for r in rows:
            self.d.quote_status[rubric.quote_key(r["url"], r["quote"])] = "found"
        self.assertTrue(any("rubric gives 'continued-foreign'" in e for e in self.d.validate()))


if __name__ == "__main__":
    unittest.main()
