"""Branded, self-contained HTML report for an EpiNet run.

``epinet-workbench run`` writes the science outputs; this module then renders a
single ``index.html`` into the same bundle — the polished, portable artifact you
share or print to PDF. It is fully offline: figures are base64-embedded, CSS is
inline, and the only outbound links are relative paths to the bundle's own CSVs.

Design contract: the theme block in ``analysis.yaml`` lets a user change the
brand name, title, logo, accent colours, and plot palette — but **not** remove
the scientific caveats, the claims check, or the provenance. Those sections are
always rendered. Colour can change; the warnings cannot.
"""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path

# CSVs surfaced as downloads when present, in display order.
_CSV_DOWNLOADS = [
    ("Node features", "node_features.csv"),
    ("Contestability (per node)", "node_contestability.csv"),
    ("Feature importance", "model_feature_importance.csv"),
    ("Baseline comparison", "baseline_comparison.csv"),
]

# Figures shown first, in a sensible reading order; any remaining plots follow.
_PLOT_ORDER = [
    "calibration.png", "permutation_null.png", "confusion_matrix.png",
    "feature_importance.png", "contestability.png", "metric_stability.png",
    "learning_curve.png", "feature_clusters.png", "network_overview.png",
    "degree_distribution.png",
]


# --------------------------------------------------------------------------- #
# Minimal markdown -> HTML (covers the model card + claims subset)
# --------------------------------------------------------------------------- #

def _inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<em>\1</em>", text)
    return text


def md_to_html(md: str) -> str:
    """Render the markdown subset used by EpiNet cards: headers, tables, lists."""
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("### "):
            out.append(f"<h3>{_inline(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{_inline(line[2:])}</h1>")
        elif line.lstrip().startswith("- "):
            out.append("<ul>")
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                out.append(f"<li>{_inline(lines[i].lstrip()[2:])}</li>")
                i += 1
            out.append("</ul>")
            continue
        elif line.strip().startswith("|"):
            table, sep = [], False
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if set("".join(cells)) <= {"-", ":", " "} and cells:
                    sep = True  # the |---|---| separator row
                else:
                    table.append(cells)
                i += 1
            out.append("<table>")
            for r, row in enumerate(table):
                tag = "th" if (r == 0 and sep) else "td"
                out.append("<tr>" + "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in row) + "</tr>")
            out.append("</table>")
            continue
        else:
            out.append(f"<p>{_inline(line)}</p>")
        i += 1
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #

def _img_tag(path: Path, alt: str) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'


def _theme_css(theme: dict) -> str:
    return ":root{" + "".join(f"--vh-{k}:{v};" for k, v in theme.items()) + "}"


_STYLE = """
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  color:var(--vh-text);background:var(--vh-canvas);line-height:1.55}
header{background:var(--vh-ink);color:#fff;padding:28px 40px;display:flex;
  align-items:center;gap:18px}
header img{max-height:48px}
header .titles h1{margin:0;font-size:22px}
header .titles p{margin:2px 0 0;opacity:.8;font-size:13px}
main{max-width:960px;margin:0 auto;padding:32px 40px 64px}
section{background:var(--vh-surface);border:1px solid var(--vh-grid);border-radius:10px;
  padding:20px 26px;margin:0 0 22px}
h1,h2,h3{color:var(--vh-ink)}
h2{border-bottom:2px solid var(--vh-primary);padding-bottom:6px;margin-top:0}
a{color:var(--vh-primary)}
code{background:var(--vh-grid);padding:1px 5px;border-radius:4px;font-size:90%}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}
th,td{border:1px solid var(--vh-grid);padding:7px 10px;text-align:left;vertical-align:top}
th{background:var(--vh-canvas)}
img{max-width:100%;height:auto;border:1px solid var(--vh-grid);border-radius:6px;margin:6px 0}
.figure{margin:18px 0}
.figure figcaption{font-size:13px;color:var(--vh-axis);margin-top:4px}
.headline{font-size:17px;font-weight:600;color:var(--vh-ink);
  border-left:5px solid var(--vh-accent);padding:10px 0 10px 16px;margin:0 0 8px}
.caveat{background:#fff7f0;border:1px solid #f0c9a8;border-left:5px solid #d97706;
  border-radius:8px;padding:14px 18px;color:#5b3a16}
.gate-ok{color:var(--vh-accent);font-weight:600}
.gate-warn{color:#b45309;font-weight:600}
.gate-bad{color:#b00020;font-weight:600}
.downloads a{display:inline-block;margin:4px 12px 4px 0}
footer{max-width:960px;margin:0 auto;padding:0 40px 48px;color:var(--vh-axis);font-size:12px}
@media print{
  header{background:#fff;color:#000;border-bottom:2px solid var(--vh-ink)}
  header .titles p{color:#333;opacity:1}
  section{break-inside:avoid;border:1px solid #ccc}
  .downloads{display:none}
  body{background:#fff}
}
"""


def _status_class(status: str) -> str:
    s = (status or "").lower()
    # "not resolvable" is an inconclusive gate, NOT a pass — it must never render
    # green, or the HTML report would over-claim exactly where the data is silent.
    if any(k in s for k in ("not detected", "at floor", "leakage", "incomplete", "not resolvable")):
        return "gate-warn"
    if any(k in s for k in ("not run", "not compared")):
        return "gate-warn"
    return "gate-ok"


def _claims_section(claims: dict) -> str:
    rows = [
        ("Permutation null", claims["permutation"]),
        ("Split sensitivity", claims["split_comparison"]),
        ("Baseline floor", claims["baselines"]),
        ("External validation", claims["external_validation"]),
    ]
    body = ['<table><tr><th>Gate</th><th>Status</th><th>Reading</th></tr>']
    for name, gate in rows:
        cls = _status_class(gate.get("status", ""))
        body.append(
            f'<tr><td>{name}</td>'
            f'<td class="{cls}">{html.escape(gate.get("status", "—"))}</td>'
            f'<td>{html.escape(gate.get("statement", ""))}</td></tr>'
        )
    body.append("</table>")
    return (
        '<section><h2>Scientific claims check</h2>'
        f'<p class="headline">{html.escape(claims["headline"])}</p>'
        + "".join(body) + "</section>"
    )


def build_html_report(
    output_dir: str | Path,
    *,
    config,
    claims: dict,
    metrics: dict | None = None,
    palette: dict | None = None,
) -> Path:
    """Render ``index.html`` into ``output_dir`` and return its path."""
    output_dir = Path(output_dir)
    reporting = config.analysis.reporting
    theme = dict((palette or {}).get("theme", {})) or {
        "ink": "#2D2440", "primary": "#5E4F99", "accent": "#8273C0",
        "canvas": "#F5F4F7", "surface": "#FFFFFF", "grid": "#E2E0E6",
        "muted": "#C2BFCB", "axis": "#5C5866", "text": "#2A2730",
    }
    # Per-run overrides: the user may recolour, but the structure is fixed.
    if reporting.primary_color:
        theme["primary"] = reporting.primary_color
    if reporting.accent_color:
        theme["accent"] = reporting.accent_color

    parts: list[str] = []

    # Header with optional logo.
    logo = ""
    if reporting.logo_path and Path(reporting.logo_path).exists():
        data = base64.b64encode(Path(reporting.logo_path).read_bytes()).decode("ascii")
        suffix = Path(reporting.logo_path).suffix.lstrip(".") or "png"
        logo = f'<img alt="logo" src="data:image/{suffix};base64,{data}">'
    parts.append(
        f'<header>{logo}<div class="titles">'
        f'<h1>{html.escape(reporting.report_title)}</h1>'
        f'<p>{html.escape(reporting.brand_name)} · project: '
        f'{html.escape(config.project.name)}</p></div></header><main>'
    )

    # Summary + the standing caveat (always present, top of report).
    parts.append(
        '<section><h2>Summary</h2>'
        f'<p class="headline">{html.escape(claims["headline"])}</p>'
        f'<div class="caveat"><strong>Scope:</strong> {html.escape(claims["clinical_caveat"])}'
        '</div></section>'
    )

    # Claims check.
    parts.append(_claims_section(claims))

    # Model card (rendered from the markdown the run wrote). Trim at Provenance so
    # the HTML's dedicated Claims + Provenance sections below stay canonical and
    # the card does not repeat them; the standalone model_card.md keeps both.
    card_path = output_dir / "model_card.md"
    if card_path.exists():
        card_md = card_path.read_text().split("\n## Provenance", 1)[0]
        card_md = card_md.split("\n## Scientific claims check", 1)[0]
        parts.append('<section>' + md_to_html(card_md) + '</section>')

    # Figures.
    plots_dir = output_dir / "plots"
    if plots_dir.is_dir():
        present = {p.name: p for p in plots_dir.glob("*.png")}
        ordered = [present[n] for n in _PLOT_ORDER if n in present]
        ordered += [p for n, p in sorted(present.items()) if p not in ordered]
        if ordered:
            figs = ['<section><h2>Figures</h2>']
            for p in ordered:
                caption = p.stem.replace("_", " ").title()
                figs.append(f'<figure class="figure">{_img_tag(p, caption)}'
                            f'<figcaption>{html.escape(caption)}</figcaption></figure>')
            figs.append('</section>')
            parts.append("".join(figs))

    # Provenance (always present when available).
    prov_path = output_dir / "provenance.json"
    if prov_path.exists():
        prov = json.loads(prov_path.read_text())
        git = prov.get("git", {})
        rows = [
            ("EpiNet version", prov.get("epinet_version")),
            ("Git commit", git.get("commit")),
            ("Working tree", "dirty" if git.get("dirty") else "clean" if git.get("available") else "—"),
            ("Python", prov.get("python_version")),
            ("Random seed", prov.get("random_seed")),
            ("Generated (UTC)", prov.get("created_utc")),
        ]
        trows = "".join(
            f"<tr><td>{html.escape(str(k))}</td><td><code>{html.escape(str(v))}</code></td></tr>"
            for k, v in rows
        )
        parts.append(f'<section><h2>Provenance</h2><table>{trows}</table></section>')

    # CSV downloads (relative links; work inside the unzipped bundle).
    downloads = [(label, fn) for label, fn in _CSV_DOWNLOADS if (output_dir / fn).exists()]
    if downloads:
        links = "".join(f'<a href="{fn}" download>{html.escape(label)} ↓</a>' for label, fn in downloads)
        parts.append(f'<section class="downloads"><h2>Data downloads</h2>{links}'
                     '<p>Links resolve against the files in this result bundle.</p></section>')

    parts.append('</main>')
    parts.append(
        '<footer>Generated by EpiNet — research and education demonstrator, not '
        'clinical decision support. The analysis is reproducible from '
        '<code>analysis.yaml</code> with <code>epinet-workbench run</code>.</footer>'
    )

    doc = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(reporting.report_title)}</title>"
        f"<style>{_theme_css(theme)}{_STYLE}</style></head><body>"
        + "".join(parts) + "</body></html>"
    )
    out_path = output_dir / "index.html"
    out_path.write_text(doc)
    return out_path
