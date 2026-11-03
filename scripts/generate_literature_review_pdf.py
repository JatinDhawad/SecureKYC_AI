#!/usr/bin/env python3
"""
Generate a professional, publication-grade academic PDF from paper/literature_review_article.md
Uses python-markdown and Playwright headless Chromium for high-definition rendering.
Complies with IEEE / Springer Review Article layout standards.
"""

import re
import shutil
from pathlib import Path
import markdown
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MD_PATH = PROJECT_ROOT / "paper" / "literature_review_article.md"
PDF_PAPER_PATH = PROJECT_ROOT / "paper" / "LipGuard_AI_Literature_Review_Article.pdf"
PDF_DOCS_PATH = PROJECT_ROOT / "docs" / "LipGuard_AI_Literature_Review_Article.pdf"
HTML_TEMP_PATH = PROJECT_ROOT / "scratch" / "_temp_lit_review.html"

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from generate_ieee_paper_pdf import latex_to_unicode

def build_pdf():
    print(f"Reading Markdown from: {MD_PATH}")
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Enhance math formatting for HTML rendering using latex_to_unicode
    # Convert $$ formula $$ into styled math block
    def format_math_block(match):
        formula = match.group(1).strip()
        return f'<div class="math-block">{latex_to_unicode(formula)}</div>'
    
    md_text = re.sub(r"\$\$(.*?)\$\$", format_math_block, md_text, flags=re.DOTALL)

    # Convert inline $math$ into styled inline math
    def format_inline_math(match):
        formula = match.group(1).strip()
        return f'<span class="inline-math">{latex_to_unicode(formula)}</span>'

    md_text = re.sub(r"\$([^$\n]+?)\$", format_inline_math, md_text)

    # Convert markdown to HTML with tables & code blocks
    html_content = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"]
    )

    # Custom post-processing on HTML for academic presentation
    # 1. Format Abstract box
    html_content = re.sub(
        r"<h3>Abstract</h3>\s*<p>(.*?)</p>",
        r'<div class="abstract-box"><div class="abstract-title">ABSTRACT</div><p class="abstract-text">\1</p></div>',
        html_content,
        flags=re.DOTALL
    )

    # 2. Format Keywords
    html_content = re.sub(
        r"<p><strong>Keywords</strong>:(.*?)</p>",
        r'<div class="keywords-box"><strong>Keywords—</strong>\1</div>',
        html_content
    )

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<base href="file://{PROJECT_ROOT / 'paper'}/">
<title>LipGuard AI — Systematic Literature Review Article</title>
<style>
  @page {{
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
    @top-left {{
      content: "SVKM's NMIMS MPSTME / STME — Department of Computer Engineering";
      font-size: 7.5pt;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      color: #64748B;
    }}
    @top-right {{
      content: "LipGuard AI: Real-Time Deepfake Injection Defense Review";
      font-size: 7.5pt;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      color: #64748B;
    }}
    @bottom-center {{
      content: "Capstone Review-II Deliverable | Page " counter(page) " of " counter(pages);
      font-size: 8pt;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      color: #64748B;
    }}
  }}

  body {{
    font-family: "Georgia", "Times New Roman", Cambria, serif;
    color: #1E293B;
    background-color: #FFFFFF;
    line-height: 1.55;
    font-size: 9.8pt;
    margin: 0;
    padding: 0;
    text-align: justify;
  }}

  /* Figures and Images */
  figure, p:has(> img) {{
    margin: 16px 0 10px 0;
    text-align: center;
    page-break-inside: avoid;
  }}

  img {{
    max-width: 95%;
    max-height: 220mm;
    height: auto;
    border-radius: 4px;
    border: 1px solid #CBD5E1;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    display: inline-block;
    margin: 0 auto;
  }}

  em {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 8.2pt;
    color: #475569;
    line-height: 1.35;
  }}

  /* Header & Title Section */
  h1 {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #0F172A;
    font-size: 19pt;
    font-weight: 800;
    text-align: center;
    line-height: 1.25;
    margin-top: 0;
    margin-bottom: 12px;
    letter-spacing: -0.3px;
  }}

  .author-block {{
    text-align: center;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 10pt;
    color: #334155;
    margin-bottom: 16px;
    line-height: 1.4;
  }}

  /* Section Headings */
  h2 {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1E3A8A;
    font-size: 13pt;
    font-weight: 700;
    margin-top: 22px;
    margin-bottom: 10px;
    border-bottom: 1.5px solid #CBD5E1;
    padding-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    page-break-after: avoid;
  }}

  h3 {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #0F172A;
    font-size: 10.8pt;
    font-weight: 700;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }}

  h4 {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #475569;
    font-size: 10pt;
    font-weight: 600;
    margin-top: 10px;
    margin-bottom: 4px;
    page-break-after: avoid;
  }}

  p {{
    margin-top: 0;
    margin-bottom: 10px;
  }}

  /* Abstract Box */
  .abstract-box {{
    background: #F8FAFC;
    border-left: 4px solid #2563EB;
    border-right: 1px solid #E2E8F0;
    border-top: 1px solid #E2E8F0;
    border-bottom: 1px solid #E2E8F0;
    padding: 12px 16px;
    margin: 16px 0 10px 0;
    border-radius: 4px;
    page-break-inside: avoid;
  }}

  .abstract-title {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 9.5pt;
    font-weight: 800;
    color: #1E3A8A;
    margin-bottom: 6px;
    letter-spacing: 0.8px;
  }}

  .abstract-text {{
    font-size: 9.2pt;
    line-height: 1.5;
    color: #1E293B;
    margin: 0;
    font-style: normal;
  }}

  .keywords-box {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 8.8pt;
    color: #334155;
    margin-bottom: 18px;
    padding: 4px 8px;
    background: #F1F5F9;
    border-radius: 4px;
    display: inline-block;
  }}

  /* Math Blocks */
  .math-block {{
    font-family: "Cambria Math", "Times New Roman", serif;
    font-style: italic;
    background: #F8FAFC;
    border-left: 3px solid #64748B;
    padding: 8px 14px;
    margin: 10px 0;
    font-size: 10.5pt;
    text-align: center;
    border-radius: 0 4px 4px 0;
    page-break-inside: avoid;
  }}

  .inline-math {{
    font-family: "Cambria Math", "Times New Roman", serif;
    font-style: italic;
  }}

  /* Monospaced Code & Diagram Blocks */
  pre {{
    background: #0F172A;
    color: #E2E8F0;
    padding: 10px 14px;
    border-radius: 6px;
    font-family: "SF Mono", Menlo, Monaco, Consolas, monospace;
    font-size: 7.8pt;
    line-height: 1.35;
    page-break-inside: avoid;
    margin: 12px 0;
    overflow-x: auto;
    border: 1px solid #1E293B;
  }}

  pre code {{
    background: transparent;
    color: inherit;
    padding: 0;
    font-size: inherit;
  }}

  /* Tables */
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 7.5pt;
    line-height: 1.35;
    page-break-inside: auto;
  }}

  tr {{
    page-break-inside: avoid;
    page-break-after: auto;
  }}

  th {{
    background-color: #1E3A8A;
    color: #FFFFFF;
    font-weight: 700;
    text-align: left;
    padding: 6px 6px;
    border: 1px solid #1E3A8A;
    letter-spacing: 0.2px;
  }}

  td {{
    padding: 5px 6px;
    border: 1px solid #CBD5E1;
    vertical-align: top;
  }}

  tr:nth-child(even) {{
    background-color: #F8FAFC;
  }}

  /* Lists */
  ul, ol {{
    margin-top: 0;
    margin-bottom: 10px;
    padding-left: 20px;
  }}

  li {{
    margin-bottom: 4px;
  }}

  /* Inline Code */
  code {{
    background-color: #F1F5F9;
    color: #0F172A;
    padding: 1px 4px;
    border-radius: 3px;
    font-family: "SF Mono", Menlo, Monaco, monospace;
    font-size: 8.5pt;
  }}

  /* Divider */
  hr {{
    border: 0;
    height: 1px;
    background-color: #CBD5E1;
    margin: 18px 0;
  }}

  /* References Section */
  ol li {{
    margin-bottom: 6px;
    line-height: 1.4;
  }}
</style>
</head>
<body>
{html_content}
</body>
</html>
"""

    HTML_TEMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HTML_TEMP_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Temporary HTML written to: {HTML_TEMP_PATH}")

    print("Rendering PDF via Playwright Chromium...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"file://{HTML_TEMP_PATH.resolve()}", wait_until="networkidle")
        
        # Render high-res PDF
        page.pdf(
            path=str(PDF_PAPER_PATH),
            format="A4",
            print_background=True,
            margin={"top": "20mm", "bottom": "20mm", "left": "16mm", "right": "16mm"},
            display_header_footer=False,
            header_template='<div style="font-size:7.5pt; color:#64748B; width:100%; text-align:right; padding-right:16mm; font-family:-apple-system,sans-serif;">LipGuard AI: Real-Time Deepfake Injection Defense — Literature Review</div>',
            footer_template='<div style="font-size:8pt; color:#64748B; width:100%; text-align:center; font-family:-apple-system,sans-serif;">SVKM’s NMIMS MPSTME / STME | Capstone Review-II | Page <span class="pageNumber"></span> of <span class="totalPages"></span></div>'
        )
        browser.close()

    # Also copy to docs/ folder for convenience
    PDF_DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PDF_PAPER_PATH, PDF_DOCS_PATH)

    if HTML_TEMP_PATH.exists():
        HTML_TEMP_PATH.unlink()

    print(f"PDF generated successfully at: {PDF_PAPER_PATH}")
    print(f"PDF copied to: {PDF_DOCS_PATH}")
    print(f"PDF Size: {PDF_PAPER_PATH.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    build_pdf()
