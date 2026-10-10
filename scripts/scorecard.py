"""Shared logic: load the data files, resolve models, check sources, compute tiers.

Data model:
    providers.csv  one row per organisation (seat, control, owners, official namespaces)
    families.csv   one row per model family (origin, licence, data, compute, ...)
    scorecard.csv  one row per model; empty family fields inherit from the family
    sources.csv    one row per fact, at the scope where the fact is stated
                   (provider, family or model)

A fact counts for the tier only if a source backs it whose issuer is the
provider itself, an official body, an academic publication or the press, and
which we have checked (`checked = yes`). Community sources (individuals,
third-party accounts) are shown but never count.
"""

import csv
import os
import re
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
# SM_DATA points every script at another data directory, e.g. a local overlay
# of models that are not published (data-global/, gitignored): the fact
# scripts then read and write the overlay's own files. SM_OVERLAY instead adds
# an overlay's rows to data/ when reading (build.py --overlay).
DATA = Path(os.environ.get("SM_DATA", ROOT / "data")).resolve()
OVERLAY = Path(os.environ["SM_OVERLAY"]).resolve() if os.environ.get("SM_OVERLAY") else None
# Files whose rows are keyed by id; an overlay may not repeat an id of data/.
KEYED = {"providers.csv": "id", "families.csv": "id", "scorecard.csv": "id"}

PROVIDER_ALLOWED = {
    "region": {"eu", "europe", "non-europe"},
    "control": {"independent", "acquired-non-eu", "merger-pending-non-eu"},
    "cop_signatory": {"yes", "no", "unknown"},
}
FAMILY_ALLOWED = {
    "origin": {
        "scratch", "continued-own", "continued-european",
        "finetune-european", "continued-foreign", "finetune-foreign",
    },
    "license_class": {"osi", "osi-aup", "custom", "nc", "unclear"},
    "data": {"open", "documented", "undisclosed", "unknown"},
    "compute": {"eu", "europe", "non-europe", "unknown"},
    "ai_act_summary": {"yes", "no", "unknown"},
}
# Informative family criteria (#5): shown and sourced, not part of the tier.
INFO_ALLOWED = {
    "teacher_status": {"listed", "none", "unknown"},
    "recipe": {"weights", "code", "code+data", "full", "unknown"},
    "terms_can_change": {"no", "yes", "unknown"},
    "languages_basis": {"supported", "trained", "metadata", "unknown"},
    "training_hardware": {"nvidia", "amd", "google-tpu", "other", "unknown"},
}
# Chat-template check per build (#7, 3a); "reference" is the original's row.
TEMPLATE_STATUS = {"reference", "identical", "equivalent", "differs", "suspicious", "not checked"}
TEACHER_ALLOWED = {
    "stage": {"pretraining", "midtraining", "posttraining", "judge"},
    "model_origin": {"own", "european", "foreign", "unknown"},
}
EU_LANGUAGES = {"bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu",
                "ga", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv"}
FAMILY_FIELDS = list(FAMILY_ALLOWED) + ["base_model"]
ISSUERS = {"provider", "official", "academic", "press", "community", "hbd"}
COUNTING_ISSUERS = {"provider", "official", "academic", "press"}
# Sourced facts and the scope they belong to. `control` needs a source only
# when it is not "independent".
SOURCED = {
    "provider": ["control", "cop_signatory"],
    "family": ["provider_id", "origin", "data", "compute"],
}
# Hosts whose issuer is fixed.
OFFICIAL_HOSTS = {"digital-strategy.ec.europa.eu", "eur-lex.europa.eu", "www.sec.gov"}
PRESS_HOSTS = {"siliconangle.com", "www.heise.de", "www.reuters.com", "techcrunch.com"}
# Organisations on the Hub that curate community conversions.
CURATED_ORGS = {"mlx-community", "lmstudio-community", "ggml-org", "unsloth"}
HBD_ORG = "here-be-dragons-ai"

TIERS = {
    "A": "European provider, independent, own weights, open licence, open data or European compute",
    "B": "European provider and open licence, but data or compute undisclosed, or control outside Europe",
    "C": "European provider, but a non-European base model or a custom licence",
    "pending": "Licence unclear; tier withheld until checked",
    "excluded": "Non-commercial licence",
    "comparison": "Provider outside Europe: shown for comparison with the same facts, without a tier",
}
PUBLISHER_ORDER = ["official", "hbd", "curated", "organisation", "individual", "unknown"]


def _read(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read(name):
    rows = _read(DATA / name)
    if OVERLAY is None:
        return rows
    extra = _read(OVERLAY / name)
    key = KEYED.get(name)
    if key:
        clash = {r[key] for r in rows} & {r[key] for r in extra}
        if clash:
            raise SystemExit(f"{OVERLAY / name}: ids also in {DATA / name}: {sorted(clash)}")
    return rows + extra


def publishable(dist_json, public_scorecard):
    """Refuse anything but data/ itself: no overlay, and only ids that data/ lists.

    A local overlay (data-global/, see build.py --overlay) must never reach the
    Hub; this catches a dist/ built with it or an environment pointing at it.
    """
    for var in ("SM_DATA", "SM_OVERLAY"):
        if os.environ.get(var):
            raise SystemExit(f"{var} is set: publish only from data/ (unset it and rebuild)")
    public = {r["id"] for r in _read(Path(public_scorecard))}
    extra = sorted({m["id"] for m in dist_json["models"]} - public)
    if extra:
        raise SystemExit(f"dist/ holds models that data/ does not list: {extra}; rebuild without --overlay")


def split(value):
    return [v for v in (value or "").split(";") if v]


class Data:
    def __init__(self):
        self.providers = {p["id"]: p for p in read("providers.csv")}
        self.families = {f["id"]: f for f in read("families.csv")}
        self.models = read("scorecard.csv")
        self.sources = read("sources.csv")
        self.namespaces = {n["namespace"]: n for n in read("hf_namespaces.csv")}
        self.builds = read("mlx_builds.csv")
        self.gguf_builds = read("gguf_builds.csv")
        self.teachers = read("teachers.csv")
        self.template_checks = read("template_checks.csv")

    # -- resolution ------------------------------------------------------------
    def resolve(self, model):
        """A model with its family and provider facts merged in."""
        family = self.families.get(model["family_id"], {})
        provider = self.providers.get(family.get("provider_id", ""), {})
        out = dict(model)
        for field in FAMILY_FIELDS:
            out[field] = model.get(field) or family.get(field, "")
            out[f"{field}_scope"] = "model" if model.get(field) else "family"
        out["family"] = family.get("name", "")
        out["provider_id"] = provider.get("id", "")
        out["provider"] = provider.get("name", "")
        out["provider_country"] = provider.get("country", "")
        for field in ("region", "control", "cop_signatory", "majority_owner"):
            out[field] = provider.get(field, "")
        out["provider_notes"] = provider.get("notes", "")
        for field in INFO_ALLOWED:
            out[field] = family.get(field, "") or "unknown"
        out["languages"] = family.get("languages", "")
        out["eu_languages"] = self.eu_languages(family)
        out["teachers"] = [t for t in self.teachers if t["family_id"] == model["family_id"]]
        out["teacher_origin"] = self.teacher_origin(family, out["teachers"])
        return out

    @staticmethod
    def eu_languages(family):
        codes = set(split(family.get("languages")))
        return len(codes & EU_LANGUAGES) if codes else ""

    @staticmethod
    def teacher_origin(family, teachers):
        """Most foreign origin over data-generating stages; judges do not count."""
        status = family.get("teacher_status") or "unknown"
        if status != "listed":
            return status
        origins = {t["model_origin"] for t in teachers if t["stage"] != "judge"}
        for origin in ("foreign", "unknown", "european", "own"):
            if origin in origins:
                return origin
        return "judges only"

    def resolved(self):
        return [self.resolve(m) for m in self.models]

    def sources_for(self, model, field):
        """Sources for a field of a resolved model, most specific scope first."""
        family_id, provider_id = model["family_id"], model["provider_id"]
        scopes = [("model", model["id"]), ("family", family_id), ("provider", provider_id)]
        if model.get(f"{field}_scope") == "model":
            scopes = scopes[:1]  # an override must be backed at model level
        return [s for scope in scopes for s in self.sources
                if (s["scope"], s["subject"]) == scope and s["field"] == field]

    def all_sources(self, model):
        keys = {("model", model["id"]), ("family", model["family_id"]), ("provider", model["provider_id"])}
        return [s for s in self.sources if (s["scope"], s["subject"]) in keys]

    def confirmed(self, model, field):
        """The value if a counting, checked source backs it; else 'unknown'."""
        value = model[field]
        if value == "unknown" or (field == "control" and value == "independent"):
            return value
        ok = any(s["issuer"] in COUNTING_ISSUERS and s["checked"] == "yes"
                 for s in self.sources_for(model, field))
        return value if ok else "unknown"

    # -- tiers -----------------------------------------------------------------
    def tier(self, m):
        if m["region"] == "non-europe":
            return "comparison"
        if m["license_class"] == "nc":
            return "excluded"
        if m["license_class"] == "unclear":
            return "pending"
        origin = self.confirmed(m, "origin")
        control = self.confirmed(m, "control")
        data = self.confirmed(m, "data")
        compute = self.confirmed(m, "compute")
        open_licence = m["license_class"] in {"osi", "osi-aup"}
        own = origin in {"scratch", "continued-own"}
        european = own or origin in {"continued-european", "finetune-european"}
        if open_licence and own and control == "independent" and (data == "open" or compute in {"eu", "europe"}):
            return "A"
        if open_licence and european:
            return "B"
        return "C"

    @staticmethod
    def flags(m):
        out = []
        if m["region"] == "europe":
            out.append("outside the EU")
        if m["control"] != "independent":
            out.append(m["control"].replace("-", " ").replace("non eu", "non-EU"))
        if m["license_class"] == "osi-aup":
            out.append("use policy on redistribution")
        if m["license_class"] == "custom":
            out.append("custom licence")
        if m["origin"].endswith("foreign"):
            out.append("non-European base model")
        if m["hbd_involvement"]:
            out.append("our own work involved")
        return out

    # -- issuers ---------------------------------------------------------------
    def namespace_kind(self, namespace):
        entry = self.namespaces.get(namespace)
        return entry["kind"] if entry else "unknown"

    def expected_issuer(self, url, provider_id):
        """Issuer implied by the URL, or None if the URL does not decide it."""
        parsed = urlparse(url)
        host, parts = parsed.netloc, [p for p in parsed.path.split("/") if p]
        provider = self.providers.get(provider_id, {})
        if host in OFFICIAL_HOSTS:
            return "official"
        if host in PRESS_HOSTS:
            return "press"
        if host == "huggingface.co" and parts:
            ns = parts[1] if parts[0] in ("datasets", "spaces", "api") and len(parts) > 1 else parts[0]
            if parts[0] == "api" and len(parts) > 2:
                ns = parts[2]
            if ns in split(provider.get("hf_orgs")):
                return "provider"
            if ns == HBD_ORG:
                return "hbd"
            return "community"
        if host == "github.com" and parts:
            return "provider" if parts[0] in split(provider.get("github_orgs")) else "community"
        if any(host == d or host.endswith("." + d) for d in split(provider.get("domains"))):
            return "provider"
        return None

    def source_provider(self, source):
        scope, subject = source["scope"], source["subject"]
        if scope == "provider":
            return subject
        if scope == "family":
            return self.families.get(subject, {}).get("provider_id", "")
        model = next((m for m in self.models if m["id"] == subject), None)
        return self.families.get(model["family_id"], {}).get("provider_id", "") if model else ""

    def publisher(self, build):
        """Who published an MLX build: official, hbd, curated, organisation, individual."""
        ns = build["repo"].split("/")[0]
        model = next((m for m in self.models if m["id"] == build["id"]), None)
        provider = self.providers.get(self.families.get(model["family_id"], {}).get("provider_id", ""), {}) if model else {}
        if ns in split(provider.get("hf_orgs")):
            return "official"
        if ns == HBD_ORG:
            return "hbd"
        if ns in CURATED_ORGS:
            return "curated"
        kind = self.namespace_kind(ns)
        return {"org": "organisation", "user": "individual"}.get(kind, "unknown")

    # -- validation ------------------------------------------------------------
    def validate(self):
        errors = []

        def check_values(kind, row, allowed):
            for col, values in allowed.items():
                if row.get(col) and row[col] not in values:
                    errors.append(f"{kind} {row['id']}: {col}={row[col]!r} not in {sorted(values)}")

        for p in self.providers.values():
            for col in ("name", "country", "region", "control", "cop_signatory"):
                if not p.get(col):
                    errors.append(f"provider {p['id']}: {col} is empty")
            check_values("provider", p, PROVIDER_ALLOWED)
            if not re.fullmatch(r"[A-Z]{2}", p.get("country", "")):
                errors.append(f"provider {p['id']}: country must be ISO alpha-2")
            if p.get("control") != "independent" and not p.get("majority_owner"):
                errors.append(f"provider {p['id']}: control={p['control']} needs majority_owner")
        teacher_families = {t["family_id"] for t in self.teachers}
        for t in self.teachers:
            if t["family_id"] not in self.families:
                errors.append(f"teacher {t['model']}: unknown family_id {t['family_id']!r}")
            for col, values in TEACHER_ALLOWED.items():
                if t.get(col) not in values:
                    errors.append(f"teacher {t['family_id']}/{t['model']}: {col}={t.get(col)!r} not in {sorted(values)}")
            if not any(s["scope"] == "family" and s["subject"] == t["family_id"] and s["field"] == f"teacher:{t['model']}"
                       for s in self.sources):
                errors.append(f"teacher {t['family_id']}/{t['model']}: no source (field 'teacher:{t['model']}')")
        for f in self.families.values():
            check_values("family", f, {k: v for k, v in INFO_ALLOWED.items() if f.get(k)})
            status = f.get("teacher_status") or "unknown"
            if (status == "listed") != (f["id"] in teacher_families):
                errors.append(f"family {f['id']}: teacher_status={status!r} does not match teachers.csv")
            codes = split(f.get("languages"))
            if any(not re.fullmatch(r"[a-z]{2,3}", c) for c in codes):
                errors.append(f"family {f['id']}: languages must be ISO 639 codes separated by ';'")
            needs = {"recipe": f.get("recipe"), "training_hardware": f.get("training_hardware"),
                     "languages": f.get("languages"), "teacher_status": "none" if status == "none" else ""}
            # Plain open licences without a use policy cannot change by construction.
            if f.get("terms_can_change") and not (f["terms_can_change"] == "no" and f["license_class"] == "osi"):
                needs["terms_can_change"] = f["terms_can_change"]
            for field, value in needs.items():
                if value and value != "unknown" and not any(
                        s["scope"] == "family" and s["subject"] == f["id"] and s["field"] == field for s in self.sources):
                    errors.append(f"family {f['id']}: {field}={value!r} has no source")
            if f.get("provider_id") not in self.providers:
                errors.append(f"family {f['id']}: unknown provider_id {f.get('provider_id')!r}")
            for col in FAMILY_ALLOWED:
                if not f.get(col):
                    errors.append(f"family {f['id']}: {col} is empty")
            check_values("family", f, FAMILY_ALLOWED)
        ids = [m["id"] for m in self.models]
        if len(ids) != len(set(ids)):
            errors.append("scorecard.csv: duplicate model ids")
        for m in self.models:
            if not re.fullmatch(r"[a-z0-9.\-]+", m["id"]):
                errors.append(f"model {m['id']}: id must match [a-z0-9.-]+")
            if m.get("family_id") not in self.families:
                errors.append(f"model {m['id']}: unknown family_id {m.get('family_id')!r}")
            if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", m.get("hf_repo", "")):
                errors.append(f"model {m['id']}: hf_repo must be org/name")
            check_values("model", m, FAMILY_ALLOWED)

        subjects = {"provider": set(self.providers), "family": set(self.families), "model": set(ids)}
        for s in self.sources:
            tag = f"source {s['scope']}/{s['subject']}/{s['field']}"
            if s["subject"] not in subjects.get(s["scope"], set()):
                errors.append(f"{tag}: unknown subject")
                continue
            if s["issuer"] not in ISSUERS:
                errors.append(f"{tag}: issuer {s['issuer']!r} not in {sorted(ISSUERS)}")
            if s["checked"] not in ("yes", "no"):
                errors.append(f"{tag}: checked must be yes or no")
            if not s["url"].startswith("https://"):
                errors.append(f"{tag}: url must be https")
            if s["archive_url"] and not s["archive_url"].startswith("https://web.archive.org/"):
                errors.append(f"{tag}: archive_url must be a web.archive.org snapshot")
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s["retrieved"]):
                errors.append(f"{tag}: retrieved must be YYYY-MM-DD")
            expected = self.expected_issuer(s["url"], self.source_provider(s))
            if expected and s["issuer"] != expected:
                errors.append(f"{tag}: issuer {s['issuer']!r}, but the URL says {expected!r}")

        for m in self.resolved():
            for field in SOURCED["family"]:
                if m.get(field) not in ("", "unknown") and not self.sources_for(m, field):
                    errors.append(f"model {m['id']}: {field}={m[field]!r} has no source")
            if m["control"] != "independent" and not self.sources_for(m, "control"):
                errors.append(f"model {m['id']}: control={m['control']!r} has no source")
            if m["cop_signatory"] != "unknown" and not self.sources_for(m, "cop_signatory"):
                errors.append(f"model {m['id']}: cop_signatory has no source")
        for b in self.builds:
            if b["id"] not in ids:
                errors.append(f"mlx_builds.csv: unknown model id {b['id']!r}")
        for c in read("changes.csv"):
            if c["field"] == "tier" and not c["reason"].strip():
                errors.append(f"changes.csv {c['date']} {c['model']}: tier change {c['old']} -> {c['new']} needs a reason")
        for t in self.template_checks:
            if t["status"] not in TEMPLATE_STATUS:
                errors.append(f"template_checks.csv {t['repo']}: status={t['status']!r} not in {sorted(TEMPLATE_STATUS)}")
        return errors
