"""The weekly review: what the diff reports, and results kept after a failed fetch.

    python3 -m unittest discover tests
"""

import sys
import unittest
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_terms  # noqa: E402
import verify_quotes  # noqa: E402
import weekly_diff  # noqa: E402

A, B = "a" * 40, "b" * 40


def fact(repo, revision=A, **kw):
    return {"hf_repo": repo, "fetched": "2026-11-23", "license": "apache-2.0", "revision": revision,
            "params_b": "7.0", **kw}


def template(repo, status, mid="alpha-1", sha="s1", findings="", kind="gguf"):
    return {"id": mid, "repo": repo, "kind": kind, "file": "f", "status": status, "reason": "",
            "findings": findings, "sha256": sha, "checked": "2026-11-23"}


def by(key, *rows):
    return {r[key]: r for r in rows}


class TestReport(unittest.TestCase):
    def test_nothing_changed(self):
        facts = by("hf_repo", fact("acme/Alpha"))
        self.assertEqual(weekly_diff.report({"hf_facts.csv": facts}, {"hf_facts.csv": facts}), ([], [], False))

    def test_moved_commit_is_noted_a_changed_fact_needs_a_look(self):
        old = {"hf_facts.csv": by("hf_repo", fact("acme/Alpha"), fact("acme/Beta"))}
        new = {"hf_facts.csv": by("hf_repo", fact("acme/Alpha", revision=B),
                                  fact("acme/Beta", revision=B, license="other", fetched="2026-11-30"))}
        look, record, changed = weekly_diff.report(old, new)
        self.assertTrue(changed)
        self.assertEqual(look, ["### Hub facts (`hf_facts.csv`)", "",
                                f"- `acme/Beta`: license: `apache-2.0` → `other`; revision: `{A}` → `{B}`", ""])
        self.assertIn("### Repos at a new commit, tracked facts unchanged (1)", record)
        self.assertIn(f"- `acme/Alpha`: `{A[:12]}` → `{B[:12]}`", record)

    def test_builds_are_grouped_per_model_when_the_original_changes_its_template(self):
        old = {"template_checks.csv": by("repo", template("acme/Alpha", "reference", kind="original"),
                                         *[template(f"x/Alpha-{i}-GGUF", "identical") for i in range(5)],
                                         template("y/Alpha-GGUF", "differs"))}
        new = {"template_checks.csv": by("repo", template("acme/Alpha", "reference", kind="original", sha="s2"),
                                         *[template(f"x/Alpha-{i}-GGUF", "differs") for i in range(5)],
                                         template("y/Alpha-GGUF", "differs"),
                                         template("z/Alpha-GGUF", "needs review", findings="raise_exception"))}
        look, _, changed = weekly_diff.report(old, new)
        self.assertTrue(changed)
        self.assertEqual(look[2:-1], [
            "- `alpha-1`: the original's chat template changed (`s1` → `s2`)",
            "- `alpha-1`: 5 builds `identical` → `differs` (the original's template changed): "
            "`x/Alpha-0-GGUF`, `x/Alpha-1-GGUF`, `x/Alpha-2-GGUF` and 2 more",
            "- `alpha-1`: 1 build new: `z/Alpha-GGUF`"])

    def test_new_finding_in_a_template_is_listed_by_build(self):
        old = {"template_checks.csv": by("repo", template("x/Alpha-GGUF", "differs"))}
        new = {"template_checks.csv": by("repo", template("x/Alpha-GGUF", "differs", sha="s2", findings="import os"))}
        look, _, _ = weekly_diff.report(old, new)
        self.assertEqual(look[2:-1], ["- `alpha-1`: 1 build with a changed template, same status: `x/Alpha-GGUF`",
                                      "- `x/Alpha-GGUF` (alpha-1): findings: `` → `import os`"])

    def test_new_release_of_a_runtime_alone_is_no_change(self):
        row = {"id": "alpha-1", "llama_cpp": "supported", "architecture": "X", "detail": "", "checked": "2026-11-16",
               "llama_cpp_rev": "v1.0.0"}
        new = {"llama_cpp_support.csv": {"alpha-1": {**row, "checked": "2026-11-23", "llama_cpp_rev": "v1.1.0"}}}
        self.assertEqual(weekly_diff.report({"llama_cpp_support.csv": {"alpha-1": row}}, new), ([], [], False))

    def test_kept_result_is_noted_and_counts_once_it_is_stale(self):
        def terms(day):
            return {"terms_hashes.csv": by("url", {"url": "https://a.example/licence", "sha256": "1", "checked": day},
                                           {"url": "https://b.example/licence", "sha256": "2", "checked": "2026-11-23"})}
        look, record, changed = weekly_diff.report(terms("2026-11-16"), terms("2026-11-16"))
        self.assertEqual((look, changed), ([], False))
        self.assertEqual(record[:3], ["### Licence and policy documents: not fetched in this run, last result kept (1)",
                                      "", "- `https://a.example/licence`: last checked 2026-11-16"])
        self.assertTrue(weekly_diff.report(terms("2026-11-02"), terms("2026-11-02"))[2])  # three weeks behind


class TestKeptResults(unittest.TestCase):
    def test_quote_survives_a_failed_fetch_but_not_a_changed_document(self):
        keep, found = verify_quotes.keep_previous, {"status": "found", "checked": "2026-11-16"}
        self.assertTrue(keep(found, "not found", "snapshot; live README.md failed (HTTPError)"))
        self.assertTrue(keep({"status": "found (loose)"}, "no text", "live failed (URLError)"))
        self.assertFalse(keep(found, "not found", "snapshot; live README.md"))  # read in full: the quote is gone
        self.assertFalse(keep({"status": "not found"}, "not found", "snapshot failed (URLError); live"))
        self.assertFalse(keep(None, "no text", "live failed (URLError)"))

    def test_licence_hash_survives_a_blocked_fetch_but_not_a_missing_document(self):
        keep, old = check_terms.keep_previous, {"sha256": "1", "checked": "2026-11-16"}
        error = lambda code: type("HTTPError", (urllib.error.URLError,), {"code": code})("")  # noqa: E731
        self.assertTrue(keep(old, error(403)))
        self.assertTrue(keep(old, TimeoutError()))
        self.assertFalse(keep(old, error(404)))
        self.assertFalse(keep({"sha256": "", "error": "HTTP Error 403: Forbidden"}, error(403)))  # never read
        self.assertFalse(keep(None, error(403)))


if __name__ == "__main__":
    unittest.main()
