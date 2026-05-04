"""Convert report_v2.md to a styled PDF via markdown → HTML → WeasyPrint."""

import sys
import re
import base64
from pathlib import Path

import markdown
from weasyprint import HTML, CSS


REPORT_PATH = Path("results/study_v3_20260424_2036/report_v2.md")
OUTPUT_PDF  = Path("results/study_v3_20260424_2036/report_v2.pdf")
BASE_DIR    = REPORT_PATH.parent   # images are relative to this directory


CSS_STYLE = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=Source+Code+Pro:wght@400&display=swap');

@page {
    size: A4;
    margin: 2.5cm 2.2cm 2.5cm 2.2cm;
    @bottom-center {
        content: counter(page);
        font-size: 9pt;
        color: #555;
    }
}

* { box-sizing: border-box; }

body {
    font-family: "Source Serif 4", "Georgia", "Times New Roman", serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #1a1a1a;
    text-align: justify;
    hyphens: auto;
}

/* ── Headings ─────────────────────────────────────────────── */
h1 {
    font-size: 17pt;
    font-weight: 600;
    line-height: 1.25;
    margin: 0 0 6pt 0;
    text-align: center;
    page-break-before: avoid;
    color: #111;
}
h2 {
    font-size: 12pt;
    font-weight: 600;
    margin: 18pt 0 6pt 0;
    padding-bottom: 3pt;
    border-bottom: 1pt solid #ccc;
    page-break-after: avoid;
    color: #111;
}
h3 {
    font-size: 10.5pt;
    font-weight: 600;
    font-style: italic;
    margin: 12pt 0 4pt 0;
    page-break-after: avoid;
    color: #1a1a1a;
}
h4 {
    font-size: 10.5pt;
    font-weight: 600;
    margin: 10pt 0 3pt 0;
    page-break-after: avoid;
}

/* ── Title block ──────────────────────────────────────────── */
.title-block {
    text-align: center;
    margin-bottom: 18pt;
}
.title-block p {
    margin: 2pt 0;
    font-size: 9pt;
    color: #444;
    text-align: center;
}

/* ── Abstract ────────────────────────────────────────────── */
blockquote, .abstract {
    margin: 12pt 20pt;
    padding: 8pt 12pt;
    background: #f7f7f7;
    border-left: 3pt solid #888;
    font-size: 9.5pt;
    line-height: 1.5;
    text-align: justify;
}

/* ── Paragraphs ──────────────────────────────────────────── */
p { margin: 0 0 7pt 0; }
li p { margin: 0; }

/* ── Tables ──────────────────────────────────────────────── */
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 9pt;
    margin: 10pt 0 14pt 0;
    page-break-inside: avoid;
}
thead tr {
    background: #2c3e50;
    color: #fff;
}
thead th {
    padding: 5pt 8pt;
    font-weight: 600;
    text-align: left;
    border: 1pt solid #2c3e50;
}
tbody tr:nth-child(even) { background: #f0f3f6; }
tbody tr:nth-child(odd)  { background: #fff; }
tbody td {
    padding: 4pt 8pt;
    border: 1pt solid #d0d5db;
    vertical-align: top;
}

/* ── Code ────────────────────────────────────────────────── */
code {
    font-family: "Source Code Pro", "Courier New", monospace;
    font-size: 8.5pt;
    background: #f3f4f5;
    padding: 1pt 3pt;
    border-radius: 2pt;
}
pre {
    background: #f3f4f5;
    border: 1pt solid #dde;
    border-radius: 3pt;
    padding: 8pt 10pt;
    font-size: 7.5pt;
    line-height: 1.45;
    overflow-x: auto;
    page-break-inside: avoid;
}
pre code {
    background: none;
    padding: 0;
}

/* ── Figures ─────────────────────────────────────────────── */
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 10pt auto;
    page-break-inside: avoid;
}
p > img:only-child {
    margin: 12pt auto;
}

/* figure captions: bold text immediately after an image block */
p strong:first-child {
    display: block;
    font-size: 9pt;
    text-align: center;
    color: #333;
    margin-top: -6pt;
    margin-bottom: 10pt;
}

/* ── Lists ───────────────────────────────────────────────── */
ul, ol { margin: 4pt 0 8pt 18pt; padding: 0; }
li { margin-bottom: 3pt; }

/* ── Horizontal rule ─────────────────────────────────────── */
hr {
    border: none;
    border-top: 1pt solid #ccc;
    margin: 16pt 0;
}

/* ── Emphasis / strong ───────────────────────────────────── */
strong { font-weight: 600; }
em { font-style: italic; }

/* ── Supplementary material break ───────────────────────── */
.supplement-break { page-break-before: always; }
"""


def embed_images(md_text: str, base_dir: Path) -> str:
    """Replace relative image paths with base64-encoded data URIs."""
    def replace_img(m: re.Match) -> str:
        alt  = m.group(1)
        path = m.group(2)
        if path.startswith(("http://", "https://", "data:")):
            return m.group(0)
        img_path = base_dir / path
        if not img_path.exists():
            print(f"  WARNING: image not found: {img_path}", file=sys.stderr)
            return m.group(0)
        ext  = img_path.suffix.lstrip(".").lower()
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "svg": "svg+xml"}.get(ext, "png")
        b64  = base64.b64encode(img_path.read_bytes()).decode()
        return f"![{alt}](data:image/{mime};base64,{b64})"

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_img, md_text)


def md_to_html(md_text: str) -> str:
    extensions = [
        "tables",
        "fenced_code",
        "footnotes",
        "attr_list",
        "def_list",
        "abbr",
        "md_in_html",
        "toc",
        "sane_lists",
    ]
    body = markdown.markdown(md_text, extensions=extensions)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width">
  <title>ThinkBench Report v3</title>
</head>
<body>
{body}
</body>
</html>"""


def main() -> None:
    print(f"Reading {REPORT_PATH} …")
    md_text = REPORT_PATH.read_text(encoding="utf-8")

    print("Embedding images …")
    md_text = embed_images(md_text, BASE_DIR)

    print("Converting Markdown → HTML …")
    html = md_to_html(md_text)

    print("Rendering HTML → PDF (WeasyPrint) …")
    css = CSS(string=CSS_STYLE)
    doc = HTML(string=html, base_url=str(BASE_DIR))
    doc.write_pdf(str(OUTPUT_PDF), stylesheets=[css])

    size_mb = OUTPUT_PDF.stat().st_size / 1_048_576
    print(f"PDF written: {OUTPUT_PDF}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
