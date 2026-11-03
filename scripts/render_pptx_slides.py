#!/usr/bin/env python3
"""
Render specified PPTX slides to PNG using python-pptx and Playwright Chromium.
Renders Slide 8 and Figure Slides (Slides 6, 8, 9, 10).
"""
import os
import io
import base64
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.dml.color import RGBColor
import asyncio
from playwright.async_api import async_playwright

SLIDE_WIDTH_IN = 13.333
SLIDE_HEIGHT_IN = 7.5
SCALE = 144  # 1920 x 1080 resolution
WIDTH_PX = int(SLIDE_WIDTH_IN * SCALE)   # 1920
HEIGHT_PX = int(SLIDE_HEIGHT_IN * SCALE) # 1080

def color_to_hex(color):
    if color is None:
        return None
    try:
        if hasattr(color, 'rgb') and color.rgb is not None:
            return f"#{color.rgb}"
    except Exception:
        pass
    return None

def shape_to_html(shape, slide_idx):
    left = int((shape.left / 914400) * SCALE)
    top = int((shape.top / 914400) * SCALE)
    width = int((shape.width / 914400) * SCALE)
    height = int((shape.height / 914400) * SCALE)
    
    style = f"position: absolute; left: {left}px; top: {top}px; width: {width}px; height: {height}px; box-sizing: border-box;"
    
    # Check shape type
    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
        image_bytes = shape.image.blob
        b64_img = base64.b64encode(image_bytes).decode('utf-8')
        content_type = shape.image.content_type
        return f'<div style="{style}"><img src="data:{content_type};base64,{b64_img}" style="width: 100%; height: 100%; object-fit: contain; border-radius: 6px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);" /></div>'

    if shape.shape_type == MSO_SHAPE_TYPE.TABLE:
        tbl = shape.table
        html_tbl = f'<div style="{style} overflow: hidden; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.06);">'
        html_tbl += '<table style="width: 100%; height: 100%; border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif;">'
        for r_idx, row in enumerate(tbl.rows):
            html_tbl += "<tr>"
            for c_idx, cell in enumerate(row.cells):
                bg_color = "#ffffff"
                try:
                    if cell.fill and cell.fill.type and cell.fill.fore_color:
                        hex_c = color_to_hex(cell.fill.fore_color)
                        if hex_c:
                            bg_color = hex_c
                except Exception:
                    pass
                
                # Cell text
                cell_texts = []
                for p in cell.text_frame.paragraphs:
                    align = "left"
                    if p.alignment == 2 or str(p.alignment) == "CENTER (2)":
                        align = "center"
                    elif p.alignment == 3 or str(p.alignment) == "RIGHT (3)":
                        align = "right"
                    
                    runs_html = ""
                    for r in p.runs:
                        r_font_size = int((r.font.size.pt if r.font.size else (p.font.size.pt if p.font.size else 10.5)) * (SCALE / 96))
                        r_color = color_to_hex(r.font.color) or color_to_hex(p.font.color) or "#1e293b"
                        r_bold = "font-weight: 700;" if (r.font.bold or p.font.bold) else "font-weight: 400;"
                        runs_html += f'<span style="font-size: {r_font_size}px; color: {r_color}; {r_bold}">{r.text}</span>'
                    if not p.runs and p.text:
                        runs_html = f'<span style="font-size: {int(10.5 * (SCALE/96))}px;">{p.text}</span>'
                    cell_texts.append(f'<div style="text-align: {align};">{runs_html}</div>')
                
                border_style = "border: 1px solid #e2e8f0;"
                html_tbl += f'<td style="background-color: {bg_color}; {border_style} padding: 6px 10px; vertical-align: middle;">{"".join(cell_texts)}</td>'
            html_tbl += "</tr>"
        html_tbl += "</table></div>"
        return html_tbl

    # Standard auto shape or text box
    bg_color = "transparent"
    border = "none"
    border_radius = "0px"
    try:
        if shape.fill and shape.fill.type:
            hex_c = color_to_hex(shape.fill.fore_color)
            if hex_c:
                bg_color = hex_c
        if shape.line and shape.line.color:
            line_c = color_to_hex(shape.line.color)
            if line_c:
                border = f"1px solid {line_c}"
        # If it is a rounded rectangle or card
        if shape.auto_shape_type and str(shape.auto_shape_type).startswith("ROUNDED_RECTANGLE"):
            border_radius = "10px"
            border = border if border != "none" else "1px solid #e2e8f0"
            style += " box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);"
    except Exception:
        pass

    style += f" background-color: {bg_color}; border: {border}; border-radius: {border_radius};"

    # Process text frame if any
    text_content = ""
    if shape.has_text_frame:
        style += " padding: 6px 12px; display: flex; flex-direction: column; justify-content: flex-start;"
        p_list = []
        for p in shape.text_frame.paragraphs:
            align = "left"
            if p.alignment == 2 or str(p.alignment) == "CENTER (2)":
                align = "center"
            elif p.alignment == 3 or str(p.alignment) == "RIGHT (3)":
                align = "right"
            
            p_style = f"text-align: {align}; margin: 2px 0; line-height: 1.3;"
            runs_html = ""
            for r in p.runs:
                r_font_size = int((r.font.size.pt if r.font.size else (p.font.size.pt if p.font.size else 11)) * (SCALE / 96))
                r_color = color_to_hex(r.font.color) or color_to_hex(p.font.color) or "#1e293b"
                r_bold = "font-weight: 700;" if (r.font.bold or p.font.bold) else "font-weight: 400;"
                runs_html += f'<span style="font-size: {r_font_size}px; color: {r_color}; {r_bold} font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif;">{r.text}</span>'
            if not p.runs and p.text:
                r_font_size = int(11 * (SCALE / 96))
                runs_html = f'<span style="font-size: {r_font_size}px; color: #1e293b;">{p.text}</span>'
            p_list.append(f'<div style="{p_style}">{runs_html}</div>')
        text_content = "".join(p_list)

    return f'<div style="{style}">{text_content}</div>'

def slide_to_html(slide, slide_num):
    shapes_html = []
    # Background
    bg_style = "background-color: #f8fafc; background-image: radial-gradient(#e2e8f0 1px, transparent 1px); background-size: 24px 24px;"
    
    for shape in slide.shapes:
        shapes_html.append(shape_to_html(shape, slide_num))
        
    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    width: {WIDTH_PX}px;
    height: {HEIGHT_PX}px;
    overflow: hidden;
    {bg_style}
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  }}
  .slide-container {{
    position: relative;
    width: {WIDTH_PX}px;
    height: {HEIGHT_PX}px;
    overflow: hidden;
  }}
</style>
</head>
<body>
<div class="slide-container">
{"".join(shapes_html)}
</div>
</body>
</html>"""
    return full_html

async def main():
    prs = Presentation('docs/LipGuard_AI_Review_2_Presentation.pptx')
    out_dir = Path("scratch/rendered_slides")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    slides_to_render = [6, 8, 9, 10]
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": WIDTH_PX, "height": HEIGHT_PX})
        
        for slide_num in slides_to_render:
            slide = prs.slides[slide_num - 1]
            html_content = slide_to_html(slide, slide_num)
            html_path = out_dir / f"slide_{slide_num}.html"
            html_path.write_text(html_content, encoding="utf-8")
            
            await page.goto(f"file://{html_path.resolve()}")
            await page.wait_for_timeout(500)
            
            png_path = out_dir / f"slide_{slide_num}.png"
            await page.screenshot(path=str(png_path), full_page=True)
            print(f"[✓] Rendered Slide {slide_num} -> {png_path} ({png_path.stat().st_size / 1024:.1f} KB)")
            
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
