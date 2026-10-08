"""Sovereign models scorecard: a filterable table with a detail view.

Reads scorecard.json, which scripts/publish_hf.py bundles with the Space.
The source of truth is the GitHub repository here-be-dragons-ai/sovereign-models.
"""

import json
from pathlib import Path

import gradio as gr
import pandas as pd

DATA = json.loads((Path(__file__).parent / "scorecard.json").read_text(encoding="utf-8"))
TIERS = DATA["tiers"]
MODELS = {m["id"]: m for m in DATA["models"]}

TABLE = {
    "tier": "Tier",
    "name": "Model",
    "provider_country": "Country",
    "provider": "Provider",
    "origin": "Origin",
    "license_class": "Licence",
    "data": "Data",
    "compute": "Compute",
    "params_b": "Params (B)",
    "size_4bit_gb": "4-bit GB",
    "size_8bit_gb": "8-bit GB",
    "mlx_vlm": "mlx-vlm",
    "flags": "Flags",
    "id": "id",
}
FRAME = pd.DataFrame([{v: m.get(k, "") for k, v in TABLE.items()} for m in DATA["models"]])


def _gb(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def filter_table(tiers, countries, licences, mlx_only, ram_gb):
    df = FRAME
    if tiers:
        df = df[df["Tier"].isin(tiers)]
    if countries:
        df = df[df["Country"].isin(countries)]
    if licences:
        df = df[df["Licence"].isin(licences)]
    if mlx_only:
        df = df[df["mlx-vlm"] == "loads"]
    if ram_gb:
        # Leave a quarter of the RAM for the OS, the KV cache and activations.
        budget = ram_gb * 0.75
        fits = df["4-bit GB"].map(_gb).map(lambda gb: gb is not None and gb <= budget)
        df = df[fits]
    return df


def _sources(model, field):
    if field == "license_class" and not any(s["field"] == field for s in model["sources"]):
        return f"[Hub metadata](https://huggingface.co/api/models/{model['hf_repo']})"
    out = []
    for s in model["sources"]:
        if s["field"] == field:
            note = f" — {s['note']}" if s["note"] else ""
            out.append(f"[{s['status']}]({s['url']}){note}")
    return "; ".join(out) or "—"


def detail(evt: gr.SelectData, table):
    model_id = table.iloc[evt.index[0]]["id"]
    m = MODELS[model_id]
    rows = [
        ("Provider", m["provider"], "provider"),
        ("Country / region", f"{m['provider_country']} / {m['region']}", None),
        ("Control", m["control"], "control"),
        ("Origin", m["origin"] + (f" ({m['base_model']})" if m["base_model"] else ""), "origin"),
        ("Licence", f"{m['license_class']} ({m['license']})", "license_class"),
        ("Training data", m["data"], "data"),
        ("Training compute", m["compute"], "compute"),
        ("AI Act training summary", m["ai_act_summary"], "ai_act_summary"),
        ("GPAI Code of Practice", m["cop_signatory"], "cop_signatory"),
    ]
    lines = [
        f"## {m['name']}  ·  Tier {m['tier']}",
        f"*{TIERS[m['tier']]}*",
        "",
        f"[{m['hf_repo']}](https://huggingface.co/{m['hf_repo']})  ·  {m['params_b']}B parameters  ·  "
        f"context {m['context'] or '?'}  ·  `{m['model_type']}`  ·  {m['modalities']}",
        "",
        "| Criterion | Value | Sources |",
        "|---|---|---|",
    ]
    for label, value, field in rows:
        lines.append(f"| {label} | {value} | {_sources(m, field) if field else '—'} |")
    lines += [
        "",
        "### On a Mac",
        f"- mlx-vlm: **{m['mlx_vlm']}** {('— ' + m['mlx_vlm_detail']) if m['mlx_vlm_detail'] else ''} "
        f"(checked {m['mlx_vlm_checked'] or '—'})",
        f"- Size: {m['size_4bit_gb']} GB at 4 bit, {m['size_8bit_gb']} GB at 8 bit (weights only)",
    ]
    if m["mlx_builds"]:
        lines.append("- MLX builds: " + ", ".join(
            f"[{b['repo']}](https://huggingface.co/{b['repo']}) ({b['bits']} bit, {b['publisher_kind']})"
            for b in m["mlx_builds"]))
    if m["flags"]:
        lines += ["", f"**Flags:** {m['flags']}"]
    if m["hbd_involvement"]:
        lines += ["", f"**Our involvement:** {m['hbd_involvement']}"]
    if m["notes"]:
        lines += ["", f"**Notes:** {m['notes']}"]
    return "\n".join(lines)


INTRO = """# Sovereign open-weight models
Open-weight language models rated on **sovereignty from a European point of view** —
who controls them, where the weights come from, what the licence allows, how transparent
data and compute are — and whether they run on Apple Silicon with MLX.
The tier says nothing about model quality. Click a row for sources.
"""

TIER_TEXT = "\n".join(f"- **{t}**: {d}" for t, d in TIERS.items())

with gr.Blocks(title="Sovereign open-weight models") as demo:
    gr.Markdown(INTRO)
    with gr.Accordion("Tiers", open=False):
        gr.Markdown(TIER_TEXT + "\n\nSee METHODOLOGY.md in the repository for the full rules.")
    with gr.Row():
        tiers = gr.CheckboxGroup(list(TIERS), value=["A", "B"], label="Tier")
        countries = gr.Dropdown(sorted(FRAME["Country"].unique()), multiselect=True, label="Country")
        licences = gr.Dropdown(sorted(FRAME["Licence"].unique()), multiselect=True, label="Licence class")
    with gr.Row():
        mlx_only = gr.Checkbox(label="Loads in mlx-vlm")
        ram = gr.Dropdown([None, 16, 24, 32, 48, 64, 96, 128], value=None,
                          label="Fits a Mac with … GB (4 bit)")
    table = gr.Dataframe(filter_table(["A", "B"], [], [], False, None), interactive=False, wrap=True)
    info = gr.Markdown("*Select a row to see the details and sources.*")

    inputs = [tiers, countries, licences, mlx_only, ram]
    for control in inputs:
        control.change(filter_table, inputs, table)
    table.select(detail, table, info)

if __name__ == "__main__":
    demo.launch()
