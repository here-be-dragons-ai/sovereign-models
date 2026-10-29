"""The rating rubric as code: atomic questions and the rules that turn answers into values.

METHODOLOGY.md describes how origin, training data, compute, licence class and
changeable terms are rated. Here the same rules are written down so that a value
follows from answers to small factual questions, each answered with a URL and a
verbatim quote (data/evidence.csv). Whoever reads the source, a person or a
language model, only answers the questions; the value is computed.

An answer that can move a model up (a base model, "trained from scratch", named
datasets, a site in Europe, a standard open licence) counts only if its quote was
found in the cited document (data/quote_checks.csv, scripts/verify_quotes.py).
Answers that state an absence ("not stated", "no") cannot be quoted; they record
the document that was read and can only lower a value, except for the licence
questions, where the absence of extra conditions is checked by the human spot
check (scripts/review_sample.py).

Standard library only: validate.py and build.py import this module.
"""

import hashlib
import re
import unicodedata

# -- questions -----------------------------------------------------------------
# question id: (field, text, allowed answers or None for free text, answers that need a quote)
QUESTIONS = {
    # origin
    "init_random": ("origin",
                    "Did this model's own pre-training start from randomly initialised weights? Answer 'no' if it "
                    "starts from an earlier model's weights, even the provider's own earlier model.",
                    {"yes", "no", "not stated"}, {"yes", "no"}),
    "base_model": ("origin",
                   "Whose weights does the model start from? Name the base model, or 'none'.",
                   None, "any but none"),
    "base_publisher": ("origin",
                       "Who published that base model: the same provider (own), another European provider, "
                       "or a provider outside Europe (foreign)?",
                       {"own", "european", "foreign"}, set()),
    "base_scratch": ("origin",
                     "For a base model of the same provider: does the provider document its own pre-training of "
                     "that base (or of its lineage root), e.g. corpus, token count, own tokenizer, 'from scratch' or "
                     "'no model dependencies', without naming another provider's weights it starts from? "
                     "Answer 'no' if the lineage reaches another provider's weights.",
                     {"yes", "no", "not stated"}, {"yes", "no"}),
    "post_training_only": ("origin",
                           "Was the base only post-trained (SFT, preference tuning, RL), without continued pre-training?",
                           {"yes", "no", "not stated"}, {"yes", "no"}),
    # training data
    "data_downloadable": ("data",
                          "How much of the training data has been published as it was used (the corpus itself, by "
                          "the provider or others)? Public source datasets that were filtered, sampled or mixed "
                          "without releasing the result count as 'part'. For a continued or fine-tuned model, the "
                          "data of the provider's own training stage, not the base model's.",
                          {"all", "nearly all", "part", "none", "not stated"}, {"all", "nearly all", "part"}),
    "datasets_named": ("data",
                       "Are the main datasets or sources of the provider's own training named, so that the mix "
                       "could roughly be rebuilt? Naming one dataset among unnamed others does not count.",
                       {"yes", "no"}, {"yes"}),
    "data_shares": ("data",
                    "Are shares or token counts per source or domain of the provider's own training given?",
                    {"yes", "no"}, {"yes"}),
    # compute
    "pretraining_site": ("compute",
                         "Where did pre-training run? A cluster name (e.g. Leonardo), an ISO country code, "
                         "'europe' if only Europe is stated, a cloud named without a location (e.g. DGX Cloud), "
                         "or 'not stated'. One answer per site.",
                         None, "any but not stated"),
    # licence
    "licence_text": ("license_class",
                     "Which licence text? An SPDX id (apache-2.0, mit, cc-by-4.0, ...) if it is a standard "
                     "licence, else 'custom'.",
                     None, "any"),
    "licence_modified": ("license_class",
                         "Is the licence text itself modified or supplemented by extra conditions "
                         "(model-as-a-service, user thresholds, attribution or naming duties)? "
                         "A separate use policy is asked about in use_policy_binding, a base model's "
                         "licence in inherited_terms.",
                         {"yes", "no"}, {"yes"}),
    "noncommercial": ("license_class",
                      "Does the licence forbid commercial use?",
                      {"yes", "no"}, {"yes"}),
    "use_policy_binding": ("license_class",
                           "Is the model bound to a use policy, including a third party's: incorporated in the "
                           "licence, or made a condition of access or use (e.g. accepted at a download gate)? "
                           "A policy that is only linked or recommended on the card does not count.",
                           {"yes", "no"}, {"yes"}),
    "inherited_terms": ("license_class",
                        "Does a base model's licence impose its own terms on this model (e.g. Llama, Gemma)?",
                        {"yes", "no"}, {"yes"}),
    # terms that can change
    "policy_updatable": ("terms_can_change",
                         "Do the licence or a use policy that is binding (see use_policy_binding) refer to a version "
                         "'as updated', 'in force' or 'the most recent copy', so the terms can change after release?",
                         {"yes", "no"}, {"yes"}),
}
FIELDS = ["origin", "data", "compute", "license_class", "terms_can_change"]

# Licences the methodology counts as open (osi): OSI-approved licences plus the
# Creative Commons attribution licences and permissive model licences in MIT style.
OPEN_LICENCES = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc", "mpl-2.0",
                 "cc-by-4.0", "cc-by-sa-4.0", "openmdw-1.0", "openmdw-1.1"}

EU_EEA = {"AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE",
          "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
          "IS", "LI", "NO"}
EUROPE_OTHER = {"CH", "GB", "UA", "RS", "ME", "MK", "AL", "BA", "MD", "TR"}
# Training clusters named in sources, and their country.
CLUSTERS = {
    "leonardo": "IT", "cineca": "IT", "marenostrum": "ES", "marenostrum 5": "ES", "bsc": "ES",
    "lumi": "FI", "jupiter": "DE", "juwels": "DE", "juwels booster": "DE", "jureca": "DE", "hawk": "DE",
    "alps": "CH", "cscs": "CH", "helios": "PL", "athena": "PL", "deucalion": "PT",
    "karolina": "CZ", "vega": "SI", "meluxina": "LU", "discoverer": "BG", "jean zay": "FR",
    "adastra": "FR", "dawn": "GB", "isambard-ai": "GB", "mimer": "SE", "olivia": "NO",
    "bem2": "PL", "ucloud": "DK", "abci": "JP", "abci 3.0": "JP", "tsubame": "JP", "tsubame 4.0": "JP",
    "prometheus": "US",
}
# Compute named without a location: the region stays unknown.
UNLOCATED = {"dgx cloud", "nvidia dgx cloud", "dgx cloud lepton", "lepton", "coreweave", "aws", "azure",
             "google cloud", "gcp", "oci"}


def _site_key(site):
    s = re.sub(r"[\s_\-]+", " ", site.strip().lower())
    return re.sub(r"(?<=[a-z])(?=\d)", " ", s)  # "MareNostrum5", "TSUBAME4.0" -> "marenostrum 5", "tsubame 4.0"


def needs_quote(question, answer):
    rule = QUESTIONS[question][3]
    if rule == "any":
        return True
    if rule == "any but none":
        return answer.lower() != "none"
    if rule == "any but not stated":
        return answer.lower() != "not stated"
    return answer in rule


def check_answer(question, answer):
    """Error text for an invalid answer, or None."""
    if question not in QUESTIONS:
        return f"unknown question {question!r}"
    allowed = QUESTIONS[question][2]
    if allowed is not None and answer not in allowed:
        return f"{question}={answer!r} not in {sorted(allowed)}"
    if not answer.strip():
        return f"{question}: empty answer"
    if question == "pretraining_site" and region_of(answer) is None:
        return f"pretraining_site={answer!r}: unknown cluster or country (add it to CLUSTERS)"
    return None


def region_of(site):
    """eu, europe, non-europe or unknown for a site answer; None if not understood."""
    s = _site_key(site)
    if s == "not stated" or s in {_site_key(u) for u in UNLOCATED}:
        return "unknown"
    if s == "europe":
        return "europe"
    country = {_site_key(k): v for k, v in CLUSTERS.items()}.get(s) \
        or (s.upper() if re.fullmatch(r"[a-z]{2}", s) else None)
    if not country:
        return None
    if country in EU_EEA:
        return "eu"
    if country in EUROPE_OTHER:
        return "europe"
    return "non-europe"


# -- quotes ----------------------------------------------------------------------
_TRANS = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u00ab": '"', "\u00bb": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u00a0": " ", "\u202f": " ", "\u00ad": None, "\u200b": None,
})


def normalise(text):
    """Comparable text: Unicode-compatible forms, plain quotes and dashes, words joined
    across line-break hyphenation, single spaces, case-folded."""
    t = unicodedata.normalize("NFKC", text).translate(_TRANS)
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)
    return re.sub(r"\s+", " ", t).strip().casefold()


def loose(text):
    """Letters and digits only, for markup that splits words (tables, emphasis)."""
    return re.sub(r"[\W_]+", "", normalise(text))


def quote_parts(quote):
    """A quote may skip text with '...' or '…'; every part must appear, in order."""
    parts = re.split(r"\s*(?:\.\.\.|…|\[\.\.\.\])\s*", quote.strip().strip("'\""))
    return [p for p in parts if len(p.strip()) >= 3]


def find_quote(quote, document):
    """'found', 'found (loose)' or 'not found'."""
    parts = quote_parts(quote)
    if not parts:
        return "not found"
    for norm, label in ((normalise, "found"), (loose, "found (loose)")):
        doc, pos = norm(document), 0
        for p in parts:
            i = doc.find(norm(p), pos)
            if i < 0:
                break
            pos = i + len(norm(p))
        else:
            return label
    return "not found"


def quote_key(url, quote):
    return hashlib.sha256(f"{url}\n{quote}".encode()).hexdigest()[:16]


# -- decisions -------------------------------------------------------------------
def _answers(rows, question):
    return [r["answer"] for r in rows if r["question"] == question]


def derive(field, rows):
    """The value the rubric gives for a field from usable answer rows, or None if the
    answers do not decide it yet. Rows must already be filtered to usable ones
    (valid, and quote found where one is needed)."""
    a = lambda q: _answers(rows, q)
    if field == "origin":
        # A quoted base model outweighs "from scratch": that usually describes the base's training.
        bases = [b for b in a("base_model") if b.lower() != "none"]
        publishers = a("base_publisher")
        # "No base model" counts as from scratch only where a document says so (quote found),
        # e.g. "Model dependencies: NA" in an AI Act training summary.
        stated_none = any(r["answer"].lower() == "none" and r.get("_quoted") for r in rows
                          if r["question"] == "base_model")
        if not bases and ("yes" in a("init_random") or stated_none):
            return "scratch"
        if bases and publishers:
            pub = "foreign" if "foreign" in publishers else "european" if "european" in publishers else "own"
            if pub == "own":
                # An own base counts only if its own lineage is documented (OELLM: base origin unreported).
                lineage = a("base_scratch")
                if "yes" in lineage:
                    return "continued-own"
                return "unknown" if lineage else None
            post = [v for v in a("post_training_only") if v != "not stated"]
            if not post:
                return None  # continued or fine-tuned: the answers do not say
            return ("finetune-" if "yes" in post else "continued-") + pub
        if a("init_random") or a("base_model"):
            return "unknown"
        return None
    if field == "data":
        dl = a("data_downloadable")
        if any(v in ("all", "nearly all") for v in dl):
            return "open"
        if "yes" in a("datasets_named") or "yes" in a("data_shares"):
            return "documented"
        if a("datasets_named") and a("data_shares"):
            return "undisclosed"
        return None
    if field == "compute":
        sites = a("pretraining_site")
        if not sites:
            return None
        regions = {region_of(s) for s in sites}
        for r in ("unknown", "non-europe", "europe"):
            if r in regions:
                return r
        return "eu"
    if field == "license_class":
        if "yes" in a("noncommercial"):
            return "nc"
        texts = [t.lower() for t in a("licence_text")]
        if not texts:
            return None
        # A base model's licence binds only a model built on its weights; a distilled model
        # bound to a third party's terms by its card falls under use_policy_binding.
        has_base = any(b.lower() != "none" for b in a("base_model"))
        if any(t not in OPEN_LICENCES for t in texts) or "yes" in a("licence_modified") \
                or ("yes" in a("inherited_terms") and has_base):
            return "custom"
        needed = ("licence_modified", "inherited_terms", "use_policy_binding", "noncommercial")
        if any(not a(q) for q in needed):
            return None
        return "osi-aup" if "yes" in a("use_policy_binding") else "osi"
    if field == "terms_can_change":
        upd = a("policy_updatable")
        if "yes" in upd:
            return "yes"
        if upd:
            return "no"
        return None
    raise ValueError(field)


def usable(rows, quote_status):
    """Answer rows that count: valid, and with a found quote where one is needed."""
    out = []
    for r in rows:
        if check_answer(r["question"], r["answer"]):
            continue
        quoted = bool(r["quote"].strip()) and \
            quote_status.get(quote_key(r["url"], r["quote"]), "").startswith("found")
        if needs_quote(r["question"], r["answer"]) and not quoted:
            continue
        out.append({**r, "_quoted": quoted})
    return out
