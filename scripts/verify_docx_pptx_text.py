#!/usr/bin/env python3
"""
Extract text from regenerated .docx and .pptx files and grep for:
["recall", "+23", "23.9", "28.26", "13 of 46", "doubl", "34.94", "0.6423", "0.7040"]
Print every hit with its file, line number, and line text.
"""
from pathlib import Path
import docx
from pptx import Presentation

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCX_FILES = [
    PROJECT_ROOT / "paper" / "LipGuard_AI_IEEE_Research_Paper.docx",
    PROJECT_ROOT / "docs" / "LipGuard_AI_IEEE_Research_Paper.docx",
    PROJECT_ROOT / "paper" / "LipGuard_AI_Review_2_Report.docx",
    PROJECT_ROOT / "docs" / "LipGuard_AI_Review_2_Report.docx",
]

PPTX_FILES = [
    PROJECT_ROOT / "docs" / "LipGuard_AI_Review_2_Presentation.pptx",
]

SEARCH_TERMS = [
    "recall",
    "+23",
    "23.9",
    "28.26",
    "13 of 46",
    "doubl",
    "34.94",
    "0.6423",
    "0.7040",
]

def extract_docx_lines(path):
    doc = docx.Document(path)
    lines = []
    # paragraphs
    for p in doc.paragraphs:
        txt = p.text.strip()
        if txt:
            lines.append(txt)
    # tables
    for table in doc.tables:
        for row in table.rows:
            row_txt = " | ".join(cell.text.strip().replace("\n", " ") for cell in row.cells)
            if row_txt.strip():
                lines.append(row_txt)
    return lines

def extract_pptx_lines(path):
    prs = Presentation(path)
    lines = []
    for slide_idx, slide in enumerate(prs.slides, start=1):
        for shape in slide.shapes:
            if shape.has_text_frame:
                for p in shape.text_frame.paragraphs:
                    txt = p.text.strip()
                    if txt:
                        lines.append(f"[Slide {slide_idx}] {txt}")
            if shape.has_table:
                for row in shape.table.rows:
                    row_txt = " | ".join(cell.text.strip().replace("\n", " ") for cell in row.cells)
                    if row_txt.strip():
                        lines.append(f"[Slide {slide_idx} Table] {row_txt}")
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            for p in slide.notes_slide.notes_text_frame.paragraphs:
                txt = p.text.strip()
                if txt:
                    lines.append(f"[Slide {slide_idx} Notes] {txt}")
    return lines

def main():
    print("=" * 80)
    print("SEARCH REPORT ACROSS REGENERATED BINARIES (.docx and .pptx)")
    print("Search terms:", SEARCH_TERMS)
    print("=" * 80)

    for docx_path in DOCX_FILES:
        rel_path = docx_path.relative_to(PROJECT_ROOT)
        print(f"\nScanning: {rel_path}")
        lines = extract_docx_lines(docx_path)
        hits = 0
        for term in SEARCH_TERMS:
            print(f"  --- Query: '{term}' ---")
            term_hits = 0
            for idx, line in enumerate(lines, start=1):
                if term.lower() in line.lower():
                    print(f"    Line {idx}: {line}")
                    term_hits += 1
                    hits += 1
            if term_hits == 0:
                print("    (0 hits)")
        print(f"Total hits in {rel_path}: {hits}")

    for pptx_path in PPTX_FILES:
        rel_path = pptx_path.relative_to(PROJECT_ROOT)
        print(f"\nScanning: {rel_path}")
        lines = extract_pptx_lines(pptx_path)
        hits = 0
        for term in SEARCH_TERMS:
            print(f"  --- Query: '{term}' ---")
            term_hits = 0
            for idx, line in enumerate(lines, start=1):
                if term.lower() in line.lower():
                    print(f"    Line {idx}: {line}")
                    term_hits += 1
                    hits += 1
            if term_hits == 0:
                print("    (0 hits)")
        print(f"Total hits in {rel_path}: {hits}")

if __name__ == "__main__":
    main()
