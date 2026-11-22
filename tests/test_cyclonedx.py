"""The CycloneDX export: mapping of a scorecard row and the limits tools put on a BOM.

    python3 -m unittest discover tests
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import cyclonedx_bom  # noqa: E402

try:  # optional: the official schema, as shipped with cyclonedx-python-lib
    import jsonschema  # noqa: F401 - the validator needs it
    from cyclonedx.schema import SchemaVersion
    from cyclonedx.validation.json import JsonStrictValidator
except ImportError:
    JsonStrictValidator = None

SHA = "770af27b30611fc513d41cae28cb45e3ad9a79b3"


def source(**kw):
    base = dict(scope="family", field="origin", url="https://example.org/card", issuer="provider", checked="yes",
                retrieved="2026-10-29", archive_url="https://web.archive.org/web/2026/https://example.org/card",
                note="trained from scratch")
    return {**base, **kw}


def model(**kw):
    base = dict(
        id="alpha-1", name="Alpha 1", hf_repo="acme-ai/Alpha-1", hf_revision=SHA, hf_fetched="2026-11-22",
        family="Alpha", provider="ACME AI", provider_country="FR", region="eu", control="independent",
        majority_owner="", tier="A", origin="scratch", base_model="", license_class="osi", license="apache-2.0",
        data="open", compute="eu", ai_act_summary="yes", cop_signatory="yes", terms_can_change="no", recipe="code",
        teacher_origin="none", training_hardware="nvidia", languages="en;fr", eu_languages=2, modalities="text",
        hbd_involvement="", flags="", last_reviewed="2026-10-29", sources=[source()], teachers=[])
    return {**base, **kw}


def teacher(name, stage="posttraining", origin="foreign", purpose="synthetic SFT data"):
    return {"family_id": "x", "stage": stage, "model": name, "model_origin": origin, "purpose": purpose}


def strings(node):
    """Every string value in a JSON tree."""
    if isinstance(node, str):
        yield node
    for child in node.values() if isinstance(node, dict) else node if isinstance(node, list) else ():
        yield from strings(child)


def props(component, name):
    return [p["value"] for p in component["properties"] if p["name"] == f"sovereign-models:{name}"]


class TestBom(unittest.TestCase):
    def bom(self, *models, **kw):
        return cyclonedx_bom.bom(list(models), "2026-11-22", **kw)

    def component(self, bom, ref):
        return next(c for c in bom["components"] if c["bom-ref"] == ref)

    def test_model_is_pinned_to_its_commit(self):
        c = self.component(self.bom(model()), "model:alpha-1")
        self.assertEqual((c["type"], c["group"], c["name"], c["version"]),
                         ("machine-learning-model", "acme-ai", "Alpha-1", SHA))
        self.assertEqual(c["purl"], f"pkg:huggingface/acme-ai/Alpha-1@{SHA}")
        self.assertEqual(c["externalReferences"][0],
                         {"type": "distribution", "url": f"https://huggingface.co/acme-ai/Alpha-1/tree/{SHA}",
                          "comment": "Original release on the Hugging Face Hub, at the commit the Hub facts "
                                     "were read from on 2026-11-22"})
        self.assertEqual(c["supplier"], {"name": "ACME AI", "address": {"country": "FR"}})

    def test_short_or_missing_revision_gives_no_version(self):
        # A 12-character prefix (rows fetched before the pin) names no commit a tool could resolve.
        for rev in ("770af27b3061", ""):
            c = self.component(self.bom(model(hf_revision=rev)), "model:alpha-1")
            self.assertNotIn("version", c)
            self.assertEqual(c["purl"], "pkg:huggingface/acme-ai/Alpha-1")
            self.assertEqual(c["externalReferences"][0]["url"], "https://huggingface.co/acme-ai/Alpha-1")

    def test_tier_and_control_are_properties(self):
        c = self.component(self.bom(model(control="acquired-non-europe", majority_owner="Big Corp (US)", tier="B",
                                          flags="acquired non-Europe; binding use policy")), "model:alpha-1")
        self.assertEqual(props(c, "tier"), ["B"])
        self.assertEqual(props(c, "control"), ["acquired-non-europe"])
        self.assertEqual(props(c, "majority_owner"), ["Big Corp (US)"])
        self.assertEqual(props(c, "flag"), ["acquired non-Europe", "binding use policy"])
        self.assertEqual(props(c, "hbd_involvement"), [])  # empty values are left out
        self.assertIn("Tier B: ", c["description"])

    def test_base_model_is_the_pedigree(self):
        c = self.component(self.bom(model()), "model:alpha-1")
        self.assertEqual(c["pedigree"], {"notes": "Origin of the weights: scratch"})
        c = self.component(self.bom(model(origin="finetune-foreign", base_model="Llama 3.1 8B (Meta, US)")),
                           "model:alpha-1")
        self.assertEqual(c["pedigree"], {
            "ancestors": [{"type": "machine-learning-model", "name": "Llama 3.1 8B (Meta, US)"}],
            "notes": "Origin of the weights: finetune-foreign"})
        self.assertEqual(props(c, "base_model"), ["Llama 3.1 8B (Meta, US)"])  # for tools that drop the pedigree

    def test_teachers_are_components_behind_the_dependency_graph(self):
        a = model(teachers=[teacher("Qwen3-32B"), teacher("Qwen3-32B", stage="judge", purpose="")])
        b = model(id="beta-1", hf_repo="acme-ai/Beta-1", teachers=[teacher("Qwen3-32B", origin="european")])
        bom = self.bom(a, b, model(id="gamma-1", hf_repo="acme-ai/Gamma-1"))
        # One component per teacher name, however many models used it; what differs per use stays on the model.
        t = self.component(bom, "teacher:qwen3-32b")
        self.assertEqual((t["name"], t["scope"]), ("Qwen3-32B", "excluded"))
        self.assertEqual(props(t, "role"), ["teacher"])
        self.assertEqual(len(bom["components"]), 4)
        deps = {d["ref"]: d["dependsOn"] for d in bom["dependencies"]}
        self.assertEqual(deps["sovereign-models-scorecard"], ["model:alpha-1", "model:beta-1", "model:gamma-1"])
        self.assertEqual(deps["model:alpha-1"], ["teacher:qwen3-32b"])
        self.assertEqual(deps["model:gamma-1"], [])
        self.assertEqual(deps["teacher:qwen3-32b"], [])
        self.assertEqual(props(self.component(bom, "model:alpha-1"), "teacher"),
                         ["Qwen3-32B (posttraining, foreign): synthetic SFT data", "Qwen3-32B (judge, foreign)"])
        self.assertEqual(props(self.component(bom, "model:beta-1"), "teacher"),
                         ["Qwen3-32B (posttraining, european): synthetic SFT data"])

    def test_teacher_names_that_slug_alike_keep_their_own_reference(self):
        refs = cyclonedx_bom.teacher_refs([model(teachers=[teacher("GLM 5.2"), teacher("GLM-5.2"), teacher("???")])])
        self.assertEqual(refs, {"???": "teacher:unnamed", "GLM 5.2": "teacher:glm-5-2", "GLM-5.2": "teacher:glm-5-2-2"})

    def test_sources_are_evidence_references(self):
        rows = [source(), source(),  # an exact duplicate is listed once
                source(field="control", scope="provider", issuer="press", checked="no", archive_url="", note="")]
        refs = self.component(self.bom(model(sources=rows)), "model:alpha-1")["externalReferences"][2:]
        self.assertEqual(refs, [
            {"type": "evidence", "url": "https://example.org/card",
             "comment": "origin: trained from scratch [issuer: provider; checked: yes; scope: family; retrieved: "
                        "2026-10-29; snapshot: https://web.archive.org/web/2026/https://example.org/card]"},
            {"type": "evidence", "url": "https://example.org/card",
             "comment": "control [issuer: press; checked: no; scope: provider; retrieved: 2026-10-29]"}])

    def test_spdx_identifier_only_where_it_is_the_whole_licence(self):
        licence = lambda **kw: cyclonedx_bom.licences(model(**kw))  # noqa: E731
        self.assertEqual(licence(), [{"license": {"id": "Apache-2.0"}}])
        self.assertEqual(licence(license="cc-by-nc-4.0", license_class="nc"), [{"license": {"id": "CC-BY-NC-4.0"}}])
        # Apache-2.0 with a binding use policy or custom terms must not pass a licence policy as Apache-2.0.
        self.assertEqual(licence(license_class="osi-aup"),
                         [{"license": {"name": "apache-2.0 (plus a binding use policy)"}}])
        self.assertEqual(licence(license="mit", license_class="custom"),
                         [{"license": {"name": "mit (custom or additional terms)"}}])
        self.assertEqual(licence(license="openmdw-1.1"), [{"license": {"name": "openmdw-1.1"}}])
        self.assertIsNone(licence(license=""))

    def test_text_stays_within_what_dependency_track_stores(self):
        long = "x" * 3000
        m = model(base_model=long, teachers=[teacher(long, purpose="line one\nline two\ttabbed\u00ad\u200b")],
                  sources=[source(note="first line\r\nsecond line")])
        bom = self.bom(m)
        for c in bom["components"]:
            self.assertLessEqual(len(c["name"]), 255)
            self.assertLessEqual(len(c.get("description", "")), 1024)
            for p in c["properties"]:
                self.assertLessEqual(len(p["value"]), 1024)
        for text in strings(bom):
            self.assertFalse(set(text) & set("\n\r\t\u00ad\u200b"), text[:60])
        self.assertLessEqual(len(self.component(bom, "model:alpha-1")["pedigree"]["ancestors"][0]["name"]), 255)

    def test_references_are_unique_and_resolve(self):
        bom = self.bom(model(teachers=[teacher("A"), teacher("B")]), model(id="beta-1", hf_repo="acme-ai/Beta-1"))
        refs = [c["bom-ref"] for c in bom["components"]] + [bom["metadata"]["component"]["bom-ref"]]
        self.assertEqual(len(refs), len(set(refs)))
        graph = [d["ref"] for d in bom["dependencies"]]
        self.assertEqual(sorted(graph), sorted(refs))  # every component has its node, none twice
        self.assertTrue(all(r in refs for d in bom["dependencies"] for r in d["dependsOn"]))

    def test_same_data_same_file_and_a_new_serial_number_with_new_content(self):
        one, two = self.bom(model()), self.bom(model())
        self.assertEqual(one, two)
        self.assertRegex(one["serialNumber"], r"^urn:uuid:[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$")
        self.assertEqual((one["bomFormat"], one["specVersion"], one["metadata"]["timestamp"]),
                         ("CycloneDX", "1.6", "2026-11-22T00:00:00Z"))
        self.assertNotEqual(one["serialNumber"], self.bom(model(tier="B"))["serialNumber"])

    def test_hub_metadata_fills_the_model_card(self):
        facts = {"acme-ai/Alpha-1": {"pipeline_tag": "text-generation", "model_type": "(config: HTTP 401)"}}
        c = self.component(self.bom(model(), facts=facts), "model:alpha-1")
        self.assertEqual(c["modelCard"], {"modelParameters": {"task": "text-generation"}})
        self.assertNotIn("modelCard", self.component(self.bom(model()), "model:alpha-1"))


@unittest.skipUnless(JsonStrictValidator, "needs cyclonedx-python-lib[json-validation]")
class TestSchema(unittest.TestCase):
    """The official CycloneDX schema; Dependency-Track rejects an upload that fails it.

        uv run --with 'cyclonedx-python-lib[json-validation]' python -m unittest tests.test_cyclonedx
    """

    def check(self, text):
        self.assertIsNone(JsonStrictValidator(SchemaVersion.V1_6).validate_str(text))

    def test_every_mapped_part_is_valid(self):
        m = model(origin="finetune-foreign", base_model="Llama 3.1 8B", license_class="osi-aup",
                  flags="binding use policy", teachers=[teacher("Qwen3-32B")])
        facts = {"acme-ai/Alpha-1": {"pipeline_tag": "text-generation", "model_type": "llama"}}
        self.check(json.dumps(cyclonedx_bom.bom([m, model(id="beta-1", hf_repo="acme-ai/Beta-1", hf_revision="")],
                                                "2026-11-22", facts)))

    def test_built_file_is_valid(self):
        self.check((ROOT / "dist" / "scorecard.cdx.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
