"""Shared logic: load the data files, check them, and compute tiers.

The tier rules live here and nowhere else; METHODOLOGY.md describes them in
words. A value counts for the tier only if a `primary` or `press` source in
data/sources.csv backs it; otherwise it is treated as unknown.
"""

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

ALLOWED = {
    "region": {"eu", "europe", "non-europe"},
    "control": {"independent", "acquired-non-eu", "merger-pending-non-eu"},
    "origin": {
        "scratch", "continued-own", "continued-european",
        "finetune-european", "continued-foreign", "finetune-foreign",
    },
    "license_class": {"osi", "osi-aup", "custom", "nc", "unclear"},
    "data": {"open", "documented", "undisclosed", "unknown"},
    "compute": {"eu", "europe", "non-europe", "unknown"},
    "ai_act_summary": {"yes", "no", "unknown"},
    "cop_signatory": {"yes", "no", "unknown"},
}
REQUIRED = ["id", "name", "hf_repo", "provider", "provider_country", *ALLOWED]
# Fields that need a source unless their value is unknown.
SOURCED = ["provider", "origin", "data", "compute", "cop_signatory"]
SOURCE_STATUS = {"primary", "press", "secondary"}
COUNTS = {"primary", "press"}

TIERS = {
    "A": "European provider, independent, own weights, open licence, open data or European compute",
    "B": "European provider and open licence, but data or compute undisclosed, or control outside Europe",
    "C": "European provider, but a non-European base model or a custom licence",
    "pending": "Licence unclear; tier withheld until checked",
    "excluded": "Non-commercial licence",
    "out of scope": "Provider outside Europe",
}


def read(name):
    path = DATA / name
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load():
    models = read("scorecard.csv")
    sources = read("sources.csv")
    by_field = {}
    for s in sources:
        by_field.setdefault((s["id"], s["field"]), []).append(s)
    return models, sources, by_field


def confirmed(model, field, by_field):
    """The value of a field if a primary or press source backs it, else 'unknown'."""
    value = model[field]
    if value == "unknown":
        return value
    if field not in SOURCED and field != "control":
        return value
    if field == "control" and value == "independent":
        return value  # independence is the default; acquisitions need a source
    statuses = {s["status"] for s in by_field.get((model["id"], field), [])}
    return value if statuses & COUNTS else "unknown"


def tier(model, by_field):
    region = model["region"]
    lic = model["license_class"]
    if region == "non-europe":
        return "out of scope"
    if lic == "nc":
        return "excluded"
    if lic == "unclear":
        return "pending"
    origin = confirmed(model, "origin", by_field)
    control = confirmed(model, "control", by_field)
    data = confirmed(model, "data", by_field)
    compute = confirmed(model, "compute", by_field)
    open_licence = lic in {"osi", "osi-aup"}
    own_weights = origin in {"scratch", "continued-own"}
    european_lineage = own_weights or origin in {"continued-european", "finetune-european"}
    if (
        open_licence
        and own_weights
        and control == "independent"
        and (data == "open" or compute in {"eu", "europe"})
    ):
        return "A"
    if open_licence and european_lineage:
        return "B"
    return "C"


def flags(model):
    out = []
    if model["region"] == "europe":
        out.append("outside the EU")
    if model["control"] != "independent":
        out.append(model["control"].replace("-", " "))
    if model["license_class"] == "osi-aup":
        out.append("use policy on redistribution")
    if model["license_class"] == "custom":
        out.append("custom licence")
    if model["origin"].endswith("foreign"):
        out.append("non-European base model")
    if model["hbd_involvement"]:
        out.append("our own work involved")
    return out


def validate(models, sources):
    errors = []
    ids = [m["id"] for m in models]
    if len(ids) != len(set(ids)):
        errors.append("duplicate ids in scorecard.csv")
    known = set(ids)
    for m in models:
        mid = m.get("id", "?")
        for col in REQUIRED:
            if not m.get(col):
                errors.append(f"{mid}: {col} is empty")
        for col, allowed in ALLOWED.items():
            if m.get(col) and m[col] not in allowed:
                errors.append(f"{mid}: {col}={m[col]!r} not in {sorted(allowed)}")
        if not re.fullmatch(r"[a-z0-9.\-]+", mid):
            errors.append(f"{mid}: id must match [a-z0-9.-]+")
        if not re.fullmatch(r"[A-Z]{2}", m.get("provider_country", "")):
            errors.append(f"{mid}: provider_country must be ISO alpha-2")
        if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", m.get("hf_repo", "")):
            errors.append(f"{mid}: hf_repo must be org/name")
    have = {(s["id"], s["field"]) for s in sources}
    for m in models:
        for field in SOURCED:
            if m.get(field) not in ("", "unknown") and (m["id"], field) not in have:
                errors.append(f"{m['id']}: {field}={m[field]!r} has no source")
        if m.get("control") not in ("independent", "") and (m["id"], "control") not in have:
            errors.append(f"{m['id']}: control={m['control']!r} has no source")
    for s in sources:
        if s["id"] not in known:
            errors.append(f"sources.csv: unknown id {s['id']!r}")
        if s["status"] not in SOURCE_STATUS:
            errors.append(f"sources.csv: {s['id']}/{s['field']}: bad status {s['status']!r}")
        if not s["url"].startswith("https://"):
            errors.append(f"sources.csv: {s['id']}/{s['field']}: url must be https")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s["retrieved"]):
            errors.append(f"sources.csv: {s['id']}/{s['field']}: retrieved must be YYYY-MM-DD")
    return errors
