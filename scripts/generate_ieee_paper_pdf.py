#!/usr/bin/env python3
"""
Generate a publication-grade, two-column IEEE Conference format PDF from paper/ieee_research_paper.md.
Uses python-markdown and Playwright headless Chromium.
Features:
- Exact 6-page standard IEEE conference layout (Letter, 2-column, Times New Roman)
- Full-width title, authors, abstract, and index terms header
- Robust token-placeholder math preservation and balanced-brace fraction parser
- Side-by-side ROC (Fig 3) and EER (Fig 4) comparative evaluation panel
- Small-caps centered section titles (I. INTRODUCTION, etc.)
- IEEE style table formatting (no vertical borders, horizontal rules)
- Clean reference list with IEEE hanging indent
- Dual output: paper/SecureKYC_AI_IEEE_Research_Paper.pdf and docs/
"""

import re
import shutil
from pathlib import Path
import markdown
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MD_PATH = PROJECT_ROOT / "paper" / "ieee_research_paper.md"
PDF_CAMERA_READY_PATH = PROJECT_ROOT / "paper" / "SecureKYC_AI_IEEE_Research_Paper.pdf"
PDF_ANONYMIZED_PATH = PROJECT_ROOT / "paper" / "SecureKYC_AI_IEEE_Research_Paper_Anonymized.pdf"
DOCS_CAMERA_READY_PATH = PROJECT_ROOT / "docs" / "SecureKYC_AI_IEEE_Research_Paper.pdf"
DOCS_ANONYMIZED_PATH = PROJECT_ROOT / "docs" / "SecureKYC_AI_IEEE_Research_Paper_Anonymized.pdf"
HTML_TEMP_PATH = PROJECT_ROOT / "scratch" / "_temp_ieee_paper.html"

def replace_frac(text):
    """Replace LaTeX \\frac{num}{den} with semantic HTML fraction markup."""
    pattern = r"\frac{"
    while pattern in text:
        idx = text.find(pattern)
        open_count = 1
        num_end = -1
        for i in range(idx + len(pattern), len(text)):
            if text[i] == "{":
                open_count += 1
            elif text[i] == "}":
                open_count -= 1
                if open_count == 0:
                    num_end = i
                    break
        if num_end == -1 or num_end + 1 >= len(text) or text[num_end + 1] != "{":
            break
        den_start = num_end + 1
        open_count = 1
        den_end = -1
        for i in range(den_start + 1, len(text)):
            if text[i] == "{":
                open_count += 1
            elif text[i] == "}":
                open_count -= 1
                if open_count == 0:
                    den_end = i
                    break
        if den_end == -1:
            break
        num = text[idx + len(pattern):num_end]
        den = text[den_start + 1:den_end]
        text = text[:idx] + f"""<span class="frac"><span class="num">{num}</span><span class="den">{den}</span></span>""" + text[den_end + 1:]
    return text

def latex_to_unicode(text):
    """Cleanly replace common LaTeX math commands with unicode/HTML equivalents."""
    text = replace_frac(text)
    # Strip delimiters first so \left( doesn't collide with \le
    delims = [
        (r"\\left\(", "("),
        (r"\\right\)", ")"),
        (r"\\left\[", "["),
        (r"\\right\]", "]"),
        (r"\\left\|", "|"),
        (r"\\right\|", "|"),
        (r"\\left\.", ""),
        (r"\\right\.", ""),
        (r"\\\{", "{"),
        (r"\\\}", "}"),
        (r"\\\|", "||"),
    ]
    for pattern, repl in delims:
        text = re.sub(pattern, repl, text)

    replacements = [
        (r"\\gg(?![a-zA-Z])", "&gg;"),
        (r"\\ll(?![a-zA-Z])", "&ll;"),
        (r"\\ge(?![a-zA-Z])", "&ge;"),
        (r"\\le(?![a-zA-Z])", "&le;"),
        (r"\\pm(?![a-zA-Z])", "&plusmn;"),
        (r"\\approx(?![a-zA-Z])", "&asymp;"),
        (r"\\times(?![a-zA-Z])", "&times;"),
        (r"\\cdot(?![a-zA-Z])", "&sdot;"),
        (r"\\neq(?![a-zA-Z])", "&ne;"),
        (r"\\tau(?![a-zA-Z])", "&tau;"),
        (r"\\rho(?![a-zA-Z])", "&rho;"),
        (r"\\sigma(?![a-zA-Z])", "&sigma;"),
        (r"\\Delta(?![a-zA-Z])", "&Delta;"),
        (r"\\alpha(?![a-zA-Z])", "&alpha;"),
        (r"\\lambda(?![a-zA-Z])", "&lambda;"),
        (r"\\in(?![a-zA-Z])", "&isin;"),
        (r"\\cap(?![a-zA-Z])", "&cap;"),
        (r"\\emptyset(?![a-zA-Z])", "&empty;"),
        (r"\\sim(?![a-zA-Z])", "~"),
        (r"\\dots(?![a-zA-Z])", "&hellip;"),
        (r"\\cdots(?![a-zA-Z])", "&hellip;"),
        (r"\\quad(?![a-zA-Z])", "&emsp;"),
        (r"\\qquad(?![a-zA-Z])", "&emsp;&emsp;"),
        (r"\\[;,]", " "),
        (r"\\max(?![a-zA-Z])", "max"),
        (r"\\min(?![a-zA-Z])", "min"),
        (r"\\mathbb\{R\}", "<b>R</b>"),
        (r"\\mathbb\{I\}", "<b>I</b>"),
        (r"\\mathbf\{([^{}]+)\}", r"<b>\1</b>"),
        (r"\\mathit\{([^{}]+)\}", r"<i>\1</i>"),
        (r"\\mathrm\{([^{}]+)\}", r"\1"),
        (r"\\mathcal\{([^{}]+)\}", r"<i>\1</i>"),
        (r"\\text\{([^{}]+)\}", r"\1"),
        (r"\\sum_\{([^{}]+)\}\^\{([^{}]+)\}", r"&Sigma;<sub>\1</sub><sup>\2</sup>"),
        (r"\\sum(?![a-zA-Z])", "&Sigma;"),
        (r"\\forall(?![a-zA-Z])", "&forall;"),
        (r"\\rightarrow(?![a-zA-Z])", "&rarr;"),
        (r"\\to(?![a-zA-Z])", "&rarr;"),
        (r"\\%", "%"),
        (r"_\{([^{}]+)\}", r"<sub>\1</sub>"),
        (r"\^\{([^{}]+)\}", r"<sup>\1</sup>"),
        (r"(?<![A-Za-z0-9])([PSWDprTf])_(oral|ambient|desync|lip|vis|fail|fused|bilabial|max|mid|sync|audio)(?![A-Za-z0-9_])", r"\1<sub>\2</sub>"),
        (r"_([0-9a-zA-Z])", r"<sub>\1</sub>"),
        (r"\^([0-9a-zA-Z])", r"<sup>\1</sup>"),
        (r"\\tag\{([0-9]+)\}", r'<span style="float: right; font-weight: normal;">(\1)</span>'),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
    return text

def clean_inline_markup(text):
    """Clean markdown bold/italic and percent signs in plain HTML text."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = text.replace(r"\%", "%")
    return text

def build_single_pdf(anonymized=False, output_path=None, docs_output_path=None):
    print(f"Reading IEEE Paper Markdown from: {MD_PATH}")
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Extract title
    title_match = re.search(r"^#\s+(.+)$", md_text, re.MULTILINE)
    paper_title = title_match.group(1) if title_match else "Active Challenge-Response Biometric Verification for Video KYC"

    # Extract Abstract
    abstract_match = re.search(r"###\s+Abstract\s*\n+(.*?)(?=\n+\*\*Index Terms\*\*|\n+##\s+I\.)", md_text, re.DOTALL)
    abstract_text = abstract_match.group(1).strip() if abstract_match else ""

    # Extract Index Terms
    index_terms_match = re.search(r"\*\*Index Terms\*\*[—\-–\s]*(.*?)(?=\n+---|\n+##\s+I\.)", md_text, re.DOTALL)
    index_terms_text = index_terms_match.group(1).strip() if index_terms_match else ""

    # Remove title, authors, abstract from body markdown
    body_md = md_text
    body_md = re.sub(r"^#\s+.+?\n+(?=\*\*Jatin Dhawad\*\*)", "", body_md, flags=re.DOTALL)
    body_md = re.sub(r"\*\*Jatin Dhawad\*\*.*?\n+---\s*\n+", "", body_md, flags=re.DOTALL)
    body_md = re.sub(r"###\s+Abstract\s*\n+.*?(?=\n+##\s+I\.)", "", body_md, flags=re.DOTALL)

    if anonymized:
        body_md = re.sub(r"##\s+(?:ACKNOWLEDGMENT|ACKNOWLEDGEMENTS)\s*\n+.*?(?=\n+##\s+REFERENCES|\n+---)", "", body_md, flags=re.DOTALL)

    # 1. Protect Math Blocks with Alphanumeric Placeholders
    math_blocks = []
    def save_math_block(match):
        idx = len(math_blocks)
        math_blocks.append(match.group(1).strip())
        return f"QQQMATHBLOCK{idx}ZZZ"

    body_md = re.sub(r"\$\$(.*?)\$\$", save_math_block, body_md, flags=re.DOTALL)

    # 2. Protect Inline Math with Alphanumeric Placeholders
    inline_maths = []
    def save_inline_math(match):
        idx = len(inline_maths)
        inline_maths.append(match.group(1).strip())
        return f"QQQMATHINLINE{idx}ZZZ"

    body_md = re.sub(r"\$([^$\n]+?)\$", save_inline_math, body_md)

    # Clean up abstract text math and markup
    abstract_cleaned = clean_inline_markup(latex_to_unicode(re.sub(r"\$([^$\n]+?)\$", r"\1", abstract_text)))
    index_terms_cleaned = clean_inline_markup(latex_to_unicode(re.sub(r"\$([^$\n]+?)\$", r"\1", index_terms_text)))

    # Convert Markdown to HTML
    body_html = markdown.markdown(
        body_md,
        extensions=["tables", "fenced_code", "sane_lists"]
    )

    # 3. Restore and Render Math Blocks into HTML
    for idx, formula in enumerate(math_blocks):
        rendered = f'<div class="math-block">{latex_to_unicode(formula)}</div>'
        body_html = body_html.replace(f"QQQMATHBLOCK{idx}ZZZ", rendered)

    # 4. Restore and Render Inline Math into HTML
    for idx, formula in enumerate(inline_maths):
        rendered = f'<span class="inline-math">{latex_to_unicode(formula)}</span>'
        body_html = body_html.replace(f"QQQMATHINLINE{idx}ZZZ", rendered)

    # Enhance Section Headings (H2 -> Roman numeral IEEE style)
    def style_h2(match):
        header_text = match.group(1).strip()
        return f'<h2 class="ieee-section-title">{header_text}</h2>'
    body_html = re.sub(r"<h2>(.*?)</h2>", style_h2, body_html)

    # Enhance Subsection Headings (H3 -> Italic IEEE style)
    def style_h3(match):
        header_text = match.group(1).strip()
        return f'<h3 class="ieee-subsection-title">{header_text}</h3>'
    body_html = re.sub(r"<h3>(.*?)</h3>", style_h3, body_html)

    # Enhance Table titles (H4 -> IEEE Table Title format)
    def style_table_title(match):
        title_text = match.group(1).strip()
        return f'<div class="table-caption">{title_text}</div>'
    body_html = re.sub(r"<h4>((?:Table|TABLE).*?)</h4>", style_table_title, body_html)

    # Format References list into IEEE style hanging indent
    def format_references_block(match):
        ref_block = match.group(1)
        lines = [l.strip() for l in ref_block.split('\n') if l.strip()]
        output = ['<div class="references-list">']
        for line in lines:
            m = re.match(r"^\[(\d+)\]\s*(.*)", line)
            if m:
                num = m.group(1)
                text = m.group(2)
                output.append(f'<div class="ref-item"><span class="ref-num">[{num}]</span> {text}</div>')
            else:
                output.append(f'<div class="ref-item">{line}</div>')
        output.append('</div>')
        return f'<h2 class="ieee-section-title">REFERENCES</h2>\n' + "\n".join(output)

    body_html = re.sub(
        r'<h2 class="ieee-section-title">(?:References|REFERENCES)</h2>\s*<p>(.*?)</p>',
        format_references_block,
        body_html,
        flags=re.DOTALL
    )

    # Format Fig 1 (Threat Model) as full-column figure with large zoom
    fig1_match = re.search(r'<p><img[^>]+threat_model_diagram\.png[^>]+>.*?<em>(Fig\. 1\..*?)</em></p>', body_html, re.DOTALL)
    if fig1_match:
        fig1_panel = f'''<div class="ieee-column-figure">
          <img src="figures/threat_model_diagram.png" style="max-height: 300px; width: 98%; object-fit: contain;">
          <div class="fig-caption"><em>{fig1_match.group(1)}</em></div>
        </div>'''
        body_html = body_html.replace(fig1_match.group(0), fig1_panel)

    # Format Fig 2 (Speech Aerodynamics) as a wide figure banner
    fig2_match = re.search(r'<p><img[^>]+speech_aerodynamics_diagram\.png[^>]+>.*?<em>(Fig\. 2\..*?)</em></p>', body_html, re.DOTALL)
    if fig2_match:
        fig2_panel = f'''</div><div class="ieee-wide-figure">
          <img src="figures/speech_aerodynamics_diagram.png" style="width: 92%; max-height: 215px; object-fit: contain;">
          <div class="fig-caption"><em>{fig2_match.group(1)}</em></div>
        </div><div class="ieee-columns">'''
        body_html = body_html.replace(fig2_match.group(0), fig2_panel)

    # Format Fig 3 (Trajectories) as a large wide figure banner
    fig3_match = re.search(r'<p><img[^>]+real_bilabial_trajectories\.png[^>]+>.*?<em>(Fig\. 3\..*?)</em></p>', body_html, re.DOTALL)
    if fig3_match:
        fig3_panel = f'''</div><div class="ieee-wide-figure">
          <img src="figures/real_bilabial_trajectories.png" style="max-height: 200px; width: 100%; object-fit: contain; border: 0.25pt solid #CCCCCC;">
          <div class="fig-caption"><em>{fig3_match.group(1)}</em></div>
        </div><div class="ieee-columns">'''
        body_html = body_html.replace(fig3_match.group(0), fig3_panel)

    # Format Fig 4 (System Architecture) as a prominent wide figure banner
    fig4_match = re.search(r'<p><img[^>]+system_architecture_diagram\.png[^>]+>.*?<em>(Fig\. 4\..*?)</em></p>', body_html, re.DOTALL)
    if fig4_match:
        fig4_panel = f'''</div><div class="ieee-wide-figure">
          <img src="figures/system_architecture_diagram.png" style="width: 95%; max-height: 230px; object-fit: contain;">
          <div class="fig-caption"><em>{fig4_match.group(1)}</em></div>
        </div><div class="ieee-columns">'''
        body_html = body_html.replace(fig4_match.group(0), fig4_panel)

    # Side-by-Side Fig 5 & Fig 6 (ROC and EER Curves) - Large format
    fig56_match = re.search(r'<p><img[^>]+roc_curves\.png[^>]+>.*?<em>Fig\. 6\..*?</em></p>', body_html, re.DOTALL)
    if fig56_match:
        fig56_panel = '''</div><div class="ieee-wide-figure">
          <div style="display: flex; gap: 16pt; justify-content: center;">
            <div style="flex: 1; text-align: center;">
              <img src="figures/roc_curves.png" style="max-height: 195px; max-width: 100%; border: 0.25pt solid #CCCCCC;">
              <div class="fig-caption"><em>Fig. 5. Receiver Operating Characteristic (ROC) curves on held-out test split (N=103). Standalone Spatial Bi-LSTM: AUC = 0.7105; Deployed Constrained Late Fusion: AUC = 0.7197; green curve represents standalone DesyncOnlyNet branch (AUC = 0.6175).</em></div>
            </div>
            <div style="flex: 1; text-align: center;">
              <img src="figures/eer_curves.png" style="max-height: 195px; max-width: 100%; border: 0.25pt solid #CCCCCC;">
              <div class="fig-caption"><em>Fig. 6. False Acceptance Rate (FAR) and False Rejection Rate (FRR) curves. Standalone Spatial Bi-LSTM and Constrained Late Fusion both achieve EER = 36.02% (resolution ~2% on 57/46 test clips; Late Fusion t = 0.2943, Spatial t = 0.4675).</em></div>
            </div>
          </div>
        </div><div class="ieee-columns">'''
        body_html = body_html.replace(fig56_match.group(0), fig56_panel)

    # Format Fig 7 (Training Curves) as wide figure banner across two subplots
    fig7_match = re.search(r'<p><img[^>]+training_curve\.png[^>]+>.*?<em>(Fig\. 7\..*?)</em></p>', body_html, re.DOTALL)
    if fig7_match:
        fig7_panel = f'''</div><div class="ieee-wide-figure">
          <img src="figures/training_curve.png" style="max-height: 150px; width: 88%; border: 0.25pt solid #CCCCCC;">
          <div class="fig-caption"><em>{fig7_match.group(1)}</em></div>
        </div><div class="ieee-columns">'''
        body_html = body_html.replace(fig7_match.group(0), fig7_panel)

    # Format wide tables (Table I and Table II)
    def format_wide_table(match):
        table_caption = match.group(1)
        table_body = match.group(2)
        note_body = match.group(3) if (len(match.groups()) >= 3 and match.group(3)) else ""
        return f'</div><div class="ieee-wide-table"><div class="table-caption">{table_caption}</div><table>{table_body}</table>{note_body}</div><div class="ieee-columns">'

    body_html = re.sub(
        r'<div class="table-caption">((?:Table|TABLE)\s+I:.*?)</div>\s*<table>(.*?)</table>(\s*<p><em>\*Note on GOTCHA.*?</em></p>)?',
        format_wide_table,
        body_html,
        flags=re.DOTALL
    )

    body_html = re.sub(
        r'<div class="table-caption">((?:Table|TABLE)\s+II:.*?)</div>\s*<table>(.*?)</table>',
        format_wide_table,
        body_html,
        flags=re.DOTALL
    )

    body_html = re.sub(
        r'<div class="table-caption">((?:Table|TABLE)\s+III:.*?)</div>\s*<table>(.*?)</table>',
        format_wide_table,
        body_html,
        flags=re.DOTALL
    )

    body_html = re.sub(
        r'<div class="table-caption">((?:Table|TABLE)\s+IV:.*?)</div>\s*<table>(.*?)</table>',
        format_wide_table,
        body_html,
        flags=re.DOTALL
    )

    # Clean up any empty columns created by transitions
    body_html = body_html.replace('<div class="ieee-columns">\n</div>', '')
    body_html = body_html.replace('<div class="ieee-columns"></div>', '')

    # Render scientific subscripts such as P_oral, P_ambient, S_desync, W_vis, D_lip across body text, figures, and tables
    body_html = re.sub(
        r"(?<![A-Za-z0-9/])([PSWDprTf])_(oral|ambient|desync|lip|vis|fail|fused|bilabial|max|mid|sync|audio)(?![A-Za-z0-9_])",
        r"\1<sub>\2</sub>",
        body_html
    )

    if anonymized:
        authors_html = '''<div class="authors-container">
    <div class="author-block">
      <div class="author-name">Anonymous Authors</div>
      <div class="author-inst" style="font-size: 9.5pt; margin-top: 3pt;">Paper Under Double-Blind Review</div>
    </div>
  </div>'''
    else:
        authors_html = '''<div class="authors-container">
    <div class="author-block">
      <div class="author-name">Jatin Dhawad</div>
      <div class="author-dept">Dept. of Computer Engineering</div>
      <div class="author-inst">School of Technology Management & Engineering (STME)<br>SVKM\'s NMIMS, Indore, India</div>
      <div class="author-email">jatin.dhawad652@nmims.in</div>
    </div>
    <div class="author-block">
      <div class="author-name">Param Rawal</div>
      <div class="author-dept">Dept. of Computer Engineering</div>
      <div class="author-inst">School of Technology Management & Engineering (STME)<br>SVKM\'s NMIMS, Indore, India</div>
      <div class="author-email">param.rawal480@nmims.in</div>
    </div>
  </div>'''

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<base href="file://{PROJECT_ROOT / 'paper'}/">
<title>{paper_title}</title>
<style>
  @page {{
    size: letter;
    margin-top: 0.56in;
    margin-bottom: 0.56in;
    margin-left: 0.54in;
    margin-right: 0.54in;
    @bottom-center {{
      content: counter(page);
      font-size: 8.5pt;
      font-family: 'Times New Roman', Times, serif;
    }}
  }}

  * {{
    box-sizing: border-box;
  }}

  body {{
    font-family: 'Times New Roman', Times, 'Nimbus Roman No9 L', serif;
    color: #000000;
    background-color: #FFFFFF;
    font-size: 10.0pt;
    line-height: 1.18;
    margin: 0;
    padding: 0;
  }}

  /* Full-width header spanning across columns */
  .ieee-header {{
    width: 100%;
    text-align: center;
    margin-bottom: 8pt;
  }}

  .paper-title {{
    font-size: 19pt;
    font-weight: bold;
    line-height: 1.18;
    margin-top: 0;
    margin-bottom: 8pt;
    text-align: center;
  }}

  .authors-container {{
    display: flex;
    justify-content: center;
    gap: 40pt;
    margin-bottom: 8pt;
    text-align: center;
  }}

  .author-block {{
    flex: 1;
    max-width: 260pt;
  }}

  .author-name {{
    font-size: 10.5pt;
    font-weight: bold;
    margin-bottom: 2pt;
  }}

  .author-dept, .author-inst, .author-email {{
    font-size: 8.8pt;
    font-style: italic;
    line-height: 1.2;
  }}

  .author-email {{
    font-style: normal;
    font-family: monospace;
    font-size: 8pt;
    margin-top: 2pt;
  }}

  .abstract-index-container {{
    text-align: justify;
    margin: 0 auto 9pt auto;
    width: 97%;
    font-size: 8.8pt;
    line-height: 1.18;
  }}

  .abstract-bold {{
    font-weight: bold;
    font-style: italic;
  }}

  .index-terms-bold {{
    font-weight: bold;
    font-style: italic;
  }}

  /* Two Column Body Layout */
  .ieee-columns {{
    column-count: 2;
    column-gap: 0.24in;
    text-align: justify;
    text-justify: inter-word;
    margin-bottom: 3pt;
  }}

  p {{
    text-indent: 1.2em;
    margin-top: 0;
    margin-bottom: 2pt;
    text-align: justify;
  }}

  /* Remove indent for first paragraph after headings or diagrams */
  h2 + p, h3 + p, div + p, pre + p, ul + p, table + p {{
    text-indent: 0;
  }}

  /* Headings */
  .ieee-section-title {{
    text-align: center;
    text-transform: uppercase;
    font-size: 9.3pt;
    font-weight: bold;
    margin-top: 5pt;
    margin-bottom: 1.5pt;
    break-after: avoid;
  }}

  .ieee-subsection-title {{
    text-align: left;
    font-size: 9.1pt;
    font-weight: bold;
    font-style: italic;
    margin-top: 4pt;
    margin-bottom: 1pt;
    break-after: avoid;
  }}

  /* Math Blocks & Equations */
  .math-block {{
    text-align: center;
    font-family: 'Cambria Math', 'Latin Modern Math', 'Times New Roman', serif;
    font-size: 9.2pt;
    margin: 4pt 0;
    padding: 1pt;
    break-inside: avoid;
  }}

  .inline-math {{
    font-family: 'Cambria Math', 'Latin Modern Math', 'Times New Roman', serif;
    font-style: italic;
  }}

  .frac {{
    display: inline-block;
    vertical-align: middle;
    text-align: center;
    font-size: 0.9em;
    padding: 0 1.5pt;
  }}

  .frac .num {{
    display: block;
    border-bottom: 0.6pt solid #000000;
    padding-bottom: 0.5pt;
  }}

  .frac .den {{
    display: block;
    padding-top: 0.5pt;
  }}

  /* Monospace Diagrams */
  pre {{
    font-family: 'Courier New', Courier, monospace;
    font-size: 6.3pt;
    line-height: 1.1;
    background-color: #F8F9FA;
    border: 0.5pt solid #CCCCCC;
    padding: 2.5pt;
    margin: 4pt 0;
    overflow-x: hidden;
    white-space: pre;
    break-inside: avoid;
  }}

  /* Lists */
  ul, ol {{
    margin-top: 2pt;
    margin-bottom: 3pt;
    padding-left: 13pt;
  }}

  li {{
    margin-bottom: 1.5pt;
    text-align: justify;
  }}

  /* Wide Spanning Tables (outside column container) */
  .ieee-wide-table {{
    width: 100%;
    margin: 1.5pt 0;
    break-inside: avoid;
    clear: both;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 7.4pt;
    line-height: 1.14;
    margin: 2pt auto;
  }}

  th, td {{
    padding: 1.8pt 3pt;
    text-align: left;
  }}

  th {{
    font-weight: bold;
    border-top: 1pt solid #000000;
    border-bottom: 0.75pt solid #000000;
    text-align: center;
  }}

  tr:last-child td {{
    border-bottom: 1pt solid #000000;
  }}

  td {{
    border-top: 0.25pt solid #E2E8F0;
  }}

  .table-caption {{
    text-align: center;
    font-size: 8.0pt;
    font-weight: bold;
    margin-bottom: 1.5pt;
    text-transform: uppercase;
  }}

  /* Wide Spanning Figures */
  .ieee-wide-figure {{
    width: 100%;
    margin: 1.5pt 0;
    text-align: center;
    break-inside: avoid;
    clear: both;
  }}

  /* Column Figures */
  .ieee-column-figure {{
    width: 100%;
    margin: 1.5pt 0;
    text-align: center;
    break-inside: avoid;
  }}

  .fig-img img {{
    max-width: 100%;
    height: auto;
    object-fit: contain;
    border: 0.25pt solid #CCCCCC;
  }}

  .fig-caption {{
    font-size: 7.6pt;
    line-height: 1.14;
    text-align: justify;
    margin-top: 2pt;
  }}

  /* References */
  .references-list {{
    font-size: 7.2pt;
    line-height: 1.10;
    margin-top: 2pt;
  }}

  .ref-item {{
    text-indent: -1.3em;
    padding-left: 1.3em;
    margin-bottom: 1.5pt;
    text-align: justify;
  }}

  .ref-num {{
    font-weight: bold;
  }}
</style>
</head>
<body>

<div class="ieee-header">
  <div class="paper-title">{paper_title}</div>

{authors_html}

  <div class="abstract-index-container">
    <p style="text-indent: 0;">
      <span class="abstract-bold">Abstract</span>—{abstract_cleaned}
    </p>
    <p style="text-indent: 0; margin-top: 3pt;">
      <span class="index-terms-bold">Index Terms</span>—{index_terms_cleaned}
    </p>
  </div>
</div>

<div class="ieee-columns">
{body_html}
</div>

</body>
</html>
"""

    PROJECT_ROOT.joinpath("scratch").mkdir(exist_ok=True)
    with open(HTML_TEMP_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Temporary IEEE HTML written to: {HTML_TEMP_PATH}")

    print("Rendering IEEE Two-Column PDF via Playwright Chromium...")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file://{HTML_TEMP_PATH.resolve()}")
        page.wait_for_load_state("networkidle")
        
        pdf_bytes = page.pdf(
            format="Letter",
            print_background=True,
            prefer_css_page_size=True,
            margin={"top": "0.56in", "bottom": "0.56in", "left": "0.54in", "right": "0.54in"}
        )
        browser.close()

    target_out = output_path if output_path else PDF_CAMERA_READY_PATH
    with open(target_out, "wb") as f:
        f.write(pdf_bytes)
    print(f"IEEE Paper PDF ({'Anonymized' if anonymized else 'Camera-Ready'}) generated successfully at: {target_out}")

    target_docs = docs_output_path if docs_output_path else DOCS_CAMERA_READY_PATH
    if target_docs:
        target_docs.parent.mkdir(exist_ok=True)
        shutil.copyfile(target_out, target_docs)
        print(f"PDF mirrored to: {target_docs}")

    downloads_dir = Path.home() / "Downloads"
    if downloads_dir.exists():
        try:
            dest_download = downloads_dir / target_out.name
            shutil.copyfile(target_out, dest_download)
            print(f"PDF mirrored to: {dest_download}")
        except Exception as e:
            print(f"Could not copy to Downloads: {e}")

    size_kb = len(pdf_bytes) / 1024
    print(f"PDF Size: {size_kb:.1f} KB")

def build_pdf():
    build_single_pdf(anonymized=False, output_path=PDF_CAMERA_READY_PATH, docs_output_path=DOCS_CAMERA_READY_PATH)
    build_single_pdf(anonymized=True, output_path=PDF_ANONYMIZED_PATH, docs_output_path=DOCS_ANONYMIZED_PATH)

if __name__ == "__main__":
    build_pdf()
