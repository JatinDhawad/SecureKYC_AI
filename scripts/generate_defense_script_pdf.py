#!/usr/bin/env python3
"""
Generate a professional PDF document from presentation_defense_master_script.md
Uses python-markdown and Playwright headless Chromium for high-definition PDF rendering.
"""

import re
from pathlib import Path
import markdown
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MD_PATH = PROJECT_ROOT / "docs" / "presentation_defense_master_script.md"
PDF_PATH = PROJECT_ROOT / "docs" / "LipGuard_AI_Review_2_Defense_Guide.pdf"
HTML_TEMP_PATH = PROJECT_ROOT / "docs" / "_temp_defense_guide.html"

def build_pdf():
    print(f"Reading Markdown from: {MD_PATH}")
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Pre-process speaker cues for custom styling
    # Convert [JATIN SPEAKS]: into styled badge
    md_text = re.sub(
        r"\*\*\[JATIN SPEAKS\]:\*\*",
        r'<span class="badge badge-jatin">JATIN SPEAKS</span>',
        md_text
    )
    md_text = re.sub(
        r"\*\*\[PARAM SPEAKS\]:\*\*",
        r'<span class="badge badge-param">PARAM SPEAKS</span>',
        md_text
    )

    # Convert markdown to HTML with tables & code blocks
    html_content = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"]
    )

    # Wrap in modern styling
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LipGuard AI — Capstone Project Review-II Defense Script</title>
<style>
  @page {{
    size: A4;
    margin: 14mm 14mm 14mm 14mm;
    @bottom-right {{
      content: counter(page);
    }}
  }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1E293B;
    background-color: #FFFFFF;
    line-height: 1.5;
    font-size: 10pt;
    margin: 0;
    padding: 0;
  }}

  h1 {{
    color: #0F172A;
    font-size: 20pt;
    font-weight: 800;
    border-bottom: 3px solid #2563EB;
    padding-bottom: 6px;
    margin-top: 0;
    margin-bottom: 10px;
    letter-spacing: -0.5px;
  }}

  h2 {{
    color: #1E3A8A;
    font-size: 14pt;
    font-weight: 700;
    margin-top: 18px;
    margin-bottom: 10px;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 4px;
  }}

  h3 {{
    color: #0F172A;
    font-size: 11.5pt;
    font-weight: 700;
    margin-top: 14px;
    margin-bottom: 6px;
  }}

  h4 {{
    color: #334155;
    font-size: 10.5pt;
    font-weight: 600;
    margin-top: 10px;
    margin-bottom: 4px;
  }}

  p {{
    margin-top: 0;
    margin-bottom: 8px;
  }}

  /* Slide Box Demarcation */
  pre code {{
    display: block;
    background: #0F172A;
    color: #38BDF8;
    padding: 10px 14px;
    border-radius: 6px;
    font-family: "SF Mono", Monaco, Consolas, monospace;
    font-size: 9pt;
    line-height: 1.4;
    page-break-inside: avoid;
    margin: 20px 0 14px 0;
  }}

  /* Speaker Badges */
  .badge {{
    display: inline-block;
    font-weight: 700;
    font-size: 8.5pt;
    padding: 3px 8px;
    border-radius: 4px;
    margin-right: 6px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }}
  .badge-jatin {{
    background-color: #DBEAFE;
    color: #1E40AF;
    border: 1px solid #93C5FD;
  }}
  .badge-param {{
    background-color: #D1FAE5;
    color: #065F46;
    border: 1px solid #6EE7B7;
  }}

  /* Blockquotes / Scripts */
  blockquote {{
    background-color: #F8FAFC;
    border-left: 4px solid #2563EB;
    margin: 12px 0 16px 0;
    padding: 10px 16px;
    color: #1E293B;
    font-size: 10pt;
    border-radius: 0 6px 6px 0;
    page-break-inside: avoid;
  }}
  blockquote p {{
    margin-bottom: 8px;
  }}
  blockquote p:last-child {{
    margin-bottom: 0;
  }}

  /* Tables */
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0 18px 0;
    font-size: 9pt;
    page-break-inside: avoid;
  }}
  th, td {{
    padding: 8px 10px;
    border: 1px solid #CBD5E1;
    text-align: left;
    vertical-align: top;
  }}
  th {{
    background-color: #0F172A;
    color: #FFFFFF;
    font-weight: 600;
  }}
  tr:nth-child(even) {{
    background-color: #F8FAFC;
  }}

  /* Lists */
  ul, ol {{
    margin-top: 0;
    margin-bottom: 12px;
    padding-left: 20px;
  }}
  li {{
    margin-bottom: 4px;
  }}

  /* Code inline */
  code {{
    background-color: #F1F5F9;
    color: #0F172A;
    padding: 2px 5px;
    border-radius: 3px;
    font-family: "SF Mono", Monaco, Consolas, monospace;
    font-size: 9pt;
  }}

  /* Divider */
  hr {{
    border: 0;
    height: 1px;
    background-color: #E2E8F0;
    margin: 20px 0;
  }}

  /* Page Break Optimization */
  .page-break {{
    page-break-before: always;
  }}
</style>
</head>
<body>
{html_content}
</body>
</html>
"""

    with open(HTML_TEMP_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Temporary HTML written to: {HTML_TEMP_PATH}")

    print("Rendering PDF via Playwright Chromium...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"file://{HTML_TEMP_PATH.resolve()}", wait_until="networkidle")
        page.pdf(
            path=str(PDF_PATH),
            format="A4",
            print_background=True,
            margin={"top": "16mm", "bottom": "16mm", "left": "15mm", "right": "15mm"},
            display_header_footer=True,
            header_template='<div style="font-size:8pt; color:#64748B; width:100%; text-align:right; padding-right:15mm;">LipGuard AI — Review-II Defense Script</div>',
            footer_template='<div style="font-size:8pt; color:#64748B; width:100%; text-align:center;">Page <span class="pageNumber"></span> of <span class="totalPages"></span></div>'
        )
        browser.close()

    if HTML_TEMP_PATH.exists():
        HTML_TEMP_PATH.unlink()

    print(f"PDF generated successfully at: {PDF_PATH}")
    print(f"PDF Size: {PDF_PATH.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    build_pdf()
