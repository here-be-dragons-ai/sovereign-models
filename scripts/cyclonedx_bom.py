"""The scorecard as a CycloneDX bill of materials (dist/scorecard.cdx.json).

A projection of the built scorecard for supply-chain tools such as Dependency-Track;
the CSV and JSON stay the source of truth. build.py writes it with every build.

    model          component of type machine-learning-model: group and name are the Hub
                   repo, version the pinned commit, purl pkg:huggingface/<repo>@<commit>
    provider       supplier and manufacturer of the component, with the seat as country
    base model     pedigree.ancestors (the weights the model was derived from);
                   pedigree.notes states the origin
    teachers       components of their own with scope "excluded" (used in training, not
                   shipped), reached from the model through the dependency graph; stage,
                   origin and purpose are properties of the model that used them
    sources        externalReferences of type "evidence": field and note, then issuer,
                   check, scope, date and Wayback snapshot in the comment
    tier, control  properties named sovereign-models:<column of dist/scorecard.csv>

Spec version 1.6, the newest Dependency-Track accepts. Text is kept within what it
stores: one line, no control characters, property values and descriptions of at most
1,024 characters, names of at most 255.
"""

import hashlib
import json
import re
import uuid
from urllib.parse import quote

from scorecard import TIERS

SPEC = "1.6"
NS = "sovereign-models"
ROOT_REF = "sovereign-models-scorecard"
REPO_URL = "https://github.com/here-be-dragons-ai/sovereign-models"
DATASET_URL = "https://huggingface.co/datasets/here-be-dragons-ai/sovereign-models-scorecard"
SPACE_URL = "https://huggingface.co/spaces/here-be-dragons-ai/sovereign-models"
PUBLISHER = {"name": "Here Be Dragons – Verein für verantwortungsvolle Mensch-Maschine-Zukünfte",
             "address": {"country": "AT"}, "url": ["https://here-be-dragons.ai"]}
DATA_LICENCE = [{"license": {"id": "CC-BY-4.0"}}]
VALUE_MAX, NAME_MAX = 1024, 255
COMMIT = re.compile(r"[0-9a-f]{40}")
# Hub licence tags that are SPDX identifiers. An identifier is given only where the licence
# class says it is the whole story (osi, nc); with a binding use policy or custom terms the
# licence goes out by name, so that no licence policy passes it as plain Apache-2.0 or MIT.
SPDX = {"apache-2.0": "Apache-2.0", "mit": "MIT", "cc-by-4.0": "CC-BY-4.0", "cc-by-sa-4.0": "CC-BY-SA-4.0",
        "cc-by-nc-4.0": "CC-BY-NC-4.0", "cc-by-nc-sa-4.0": "CC-BY-NC-SA-4.0"}
LICENCE_NOTE = {"osi-aup": "plus a binding use policy", "custom": "custom or additional terms",
                "unclear": "licence unclear"}
# Columns of dist/scorecard.csv that go out as properties, in this order; empty ones are left out.
PROPERTIES = [
    "tier", "id", "name", "family", "provider", "provider_country", "region", "control", "majority_owner",
    "origin", "base_model", "license_class", "license", "data", "compute", "ai_act_summary", "cop_signatory",
    "terms_can_change", "recipe", "teacher_origin", "training_hardware", "languages", "eu_languages",
    "modalities", "hbd_involvement", "hf_fetched", "last_reviewed",
]
TEACHER_NOTE = ("Teacher or judge model named in the training documentation of a listed model; not part of "
                f"the shipped weights. Stage, origin and purpose are on the model that used it ({NS}:teacher).")


def clean(value, limit=None):
    """One line of printable text, cut to `limit` characters."""
    text = "".join(ch if ch.isprintable() else " " for ch in str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return text if not limit or len(text) <= limit else text[:limit - 1].rstrip() + "…"


def prop(name, value):
    return {"name": f"{NS}:{name}", "value": clean(value, VALUE_MAX)}


def tier_text(tier):
    return f"{'Tier ' + tier if tier in ('A', 'B', 'C') else tier.capitalize()}: {TIERS.get(tier, '')}"


def licences(m):
    declared = m.get("license", "")
    if not declared:
        return None
    spdx = SPDX.get(declared.lower())
    if spdx and m["license_class"] in ("osi", "nc"):
        return [{"license": {"id": spdx}}]
    note = LICENCE_NOTE.get(m["license_class"])
    return [{"license": {"name": clean(f"{declared} ({note})" if note else declared, NAME_MAX)}}]


def evidence(source):
    """A row of sources.csv as an external reference; what the row records goes into the comment."""
    facts = [f"issuer: {source['issuer']}", f"checked: {source['checked']}", f"scope: {source['scope']}",
             f"retrieved: {source['retrieved']}"]
    if source["archive_url"]:
        facts.append(f"snapshot: {source['archive_url']}")
    head = source["field"] + (f": {source['note']}" if source["note"] else "")
    return {"type": "evidence", "url": source["url"], "comment": clean(f"{head} [{'; '.join(facts)}]")}


def teacher_refs(models):
    """bom-ref per teacher name. A name is one component however many families used it."""
    refs = {}
    for name in sorted({t["model"] for m in models for t in m["teachers"]}):
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:80] or "unnamed"
        ref, n = f"teacher:{slug}", 2
        while ref in refs.values():
            ref, n = f"teacher:{slug}-{n}", n + 1
        refs[name] = ref
    return refs


def model_component(m, facts):
    org, _, repo = m["hf_repo"].partition("/")
    sha = m.get("hf_revision", "")
    pinned = bool(COMMIT.fullmatch(sha))
    hub = f"https://huggingface.co/{m['hf_repo']}"
    provider = {"name": clean(m["provider"], NAME_MAX), "address": {"country": m["provider_country"]}}
    c = {"type": "machine-learning-model", "bom-ref": f"model:{m['id']}", "supplier": provider,
         "manufacturer": provider, "group": org, "name": repo}
    if pinned:
        c["version"] = sha
    c["description"] = clean(f"{m['name']} by {m['provider']} ({m['provider_country']}). {tier_text(m['tier'])}",
                             VALUE_MAX)
    licence = licences(m)
    if licence:
        c["licenses"] = licence
    c["purl"] = f"pkg:huggingface/{quote(org, safe='')}/{quote(repo, safe='')}" + (f"@{sha}" if pinned else "")
    c["pedigree"] = {"notes": f"Origin of the weights: {m['origin']}"}
    if m["base_model"]:
        c["pedigree"] = {"ancestors": [{"type": "machine-learning-model", "name": clean(m["base_model"], NAME_MAX)}],
                         **c["pedigree"]}
    at = f"/tree/{sha}" if pinned else ""
    refs = [{"type": "distribution", "url": hub + at,
             "comment": "Original release on the Hugging Face Hub"
                        + (f", at the commit the Hub facts were read from on {m['hf_fetched']}" if pinned else "")},
            {"type": "model-card", "url": f"{hub}/blob/{sha if pinned else 'main'}/README.md"}]
    for ref in map(evidence, m["sources"]):
        if ref not in refs:
            refs.append(ref)
    c["externalReferences"] = refs
    f = facts.get(m["hf_repo"], {})
    card = {"task": f.get("pipeline_tag", ""), "modelArchitecture": f.get("model_type", "")}
    card = {k: v for k, v in card.items() if v and not v.startswith("(")}  # "(config: HTTP 401)": not read
    if card:
        c["modelCard"] = {"modelParameters": card}
    props = [prop("role", "model")] + [prop(k, m[k]) for k in PROPERTIES if m.get(k) not in ("", None)]
    props += [prop("flag", flag) for flag in m.get("flags", "").split("; ") if flag]
    props += [prop("teacher", f"{t['model']} ({t['stage']}, {t['model_origin']})"
                              + (f": {t['purpose']}" if t["purpose"] else "")) for t in m["teachers"]]
    c["properties"] = props
    return c


def bom(models, as_of, facts=None):
    """The BOM for the models of a build (the rows of dist/scorecard.json).

    `as_of` (YYYY-MM-DD) is the date of the newest fact in the build; `facts` the rows of
    hf_facts.csv by repo. The same input gives the same file: the timestamp is `as_of` and
    the serial number a hash of the content, so it changes exactly when the content does.
    """
    facts = facts or {}
    teachers = teacher_refs(models)
    components = [model_component(m, facts) for m in models]
    components += [{"type": "machine-learning-model", "bom-ref": ref, "name": clean(name, NAME_MAX),
                    "scope": "excluded", "description": TEACHER_NOTE, "properties": [prop("role", "teacher")]}
                   for name, ref in teachers.items()]
    dependencies = [{"ref": ROOT_REF, "dependsOn": [f"model:{m['id']}" for m in models]}]
    dependencies += [{"ref": f"model:{m['id']}", "dependsOn": sorted({teachers[t["model"]] for t in m["teachers"]})}
                     for m in models]
    dependencies += [{"ref": ref, "dependsOn": []} for ref in teachers.values()]
    doc = {
        "$schema": f"http://cyclonedx.org/schema/bom-{SPEC}.schema.json",
        "bomFormat": "CycloneDX",
        "specVersion": SPEC,
        "version": 1,
        "metadata": {
            "timestamp": f"{as_of}T00:00:00Z",
            "component": {
                "type": "data", "bom-ref": ROOT_REF, "supplier": PUBLISHER, "name": ROOT_REF, "version": as_of,
                "description": "Open-weight language models rated on sovereignty from a European point of view: "
                               "provider and control, origin of the weights, licence, training data and compute. "
                               "Tiers are computed by rules; every fact has a source.",
                "licenses": DATA_LICENCE,
                "externalReferences": [
                    {"type": "vcs", "url": REPO_URL},
                    {"type": "distribution", "url": DATASET_URL},
                    {"type": "website", "url": SPACE_URL},
                    {"type": "documentation", "url": f"{REPO_URL}/blob/main/METHODOLOGY.md",
                     "comment": "Criteria, tier rules and limits; data/schema.md describes this file"},
                ],
            },
            "manufacturer": PUBLISHER,
            "licenses": DATA_LICENCE,
            "properties": [prop(f"tier_rule:{tier}", text) for tier, text in TIERS.items()],
        },
        "components": components,
        "dependencies": dependencies,
    }
    digest = hashlib.sha256(json.dumps(doc, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    serial = {"serialNumber": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, f'{REPO_URL}#{digest}')}"}
    head = ("$schema", "bomFormat", "specVersion")
    return {**{k: doc[k] for k in head}, **serial, **{k: v for k, v in doc.items() if k not in head}}
