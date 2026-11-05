"""
generate_paper_docx.py
Compiles the full academic research paper into an editable, beautifully formatted
Microsoft Word document (research_paper/Stock_Market_Predictor_Research_Paper.docx).
Incorporates all 21 sections, 13 formatted tables, 12 embedded figures, and IEEE references.
"""

import os
import re
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUTPUT_DOCX = "research_paper/Stock_Market_Predictor_Research_Paper.docx"
MD_FILE = "research_paper/research_paper_final.md"
FIGURES_DIR = "research_paper/figures"
TABLES_DIR = "research_paper/tables"

def clean_xml_string(s):
    if not isinstance(s, str):
        s = str(s)
    # Remove control characters that violate XML 1.0 specification
    return re.sub(r'[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]', '', s)

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_styled_table(doc, csv_path, caption):
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)

    # Caption / Title
    p_cap = doc.add_paragraph()
    p_cap.paragraph_format.space_before = Pt(12)
    p_cap.paragraph_format.space_after = Pt(4)
    run_cap = p_cap.add_run(clean_xml_string(caption))
    run_cap.bold = True
    run_cap.font.size = Pt(10)
    run_cap.font.name = 'Times New Roman'
    run_cap.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D) # Navy

    table = doc.add_table(rows=len(df)+1, cols=len(df.columns))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    # Header Row
    hdr_cells = table.rows[0].cells
    for i, col_name in enumerate(df.columns):
        clean_name = clean_xml_string(str(col_name).replace('_', ' '))
        hdr_cells[i].text = clean_name
        set_cell_background(hdr_cells[i], '1A365D') # Deep Navy
        set_cell_margins(hdr_cells[i], 120, 120, 120, 120)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.bold = True
            r.font.size = Pt(8.5)
            r.font.name = 'Times New Roman'
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    # Data Rows
    for row_idx, row in df.iterrows():
        row_cells = table.rows[row_idx + 1].cells
        bg_color = 'F8FAFC' if row_idx % 2 == 1 else 'FFFFFF'
        for col_idx, val in enumerate(row):
            text_val = clean_xml_string(str(val) if pd.notna(val) else '—')
            row_cells[col_idx].text = text_val
            set_cell_background(row_cells[col_idx], bg_color)
            set_cell_margins(row_cells[col_idx], 80, 80, 100, 100)
            p = row_cells[col_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.size = Pt(8.5)
                r.font.name = 'Times New Roman'
                r.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)

    p_post = doc.add_paragraph()
    p_post.paragraph_format.space_after = Pt(8)

def add_styled_figure(doc, img_name, caption):
    img_path = os.path.join(FIGURES_DIR, img_name)
    if not os.path.exists(img_path):
        return

    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(14)
    p_img.paragraph_format.space_after = Pt(4)
    run_img = p_img.add_run()
    run_img.add_picture(img_path, width=Inches(6.2))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(12)
    run_cap = p_cap.add_run(caption)
    run_cap.italic = True
    run_cap.font.size = Pt(9)
    run_cap.font.name = 'Times New Roman'
    run_cap.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)

def main():
    print("Building Word Document (.docx)...")
    doc = Document()

    # Page setup: Standard Letter with 1-inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.different_first_page_header_footer = True

        # Subsequent pages header
        header = section.header
        p_head = header.paragraphs[0]
        p_head.text = "Stock Market Predictor Research Paper — Indian Equity Markets"
        p_head.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        for r in p_head.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(8.5)
            r.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

        # Subsequent pages footer with Page number
        footer = section.footer
        p_foot = footer.paragraphs[0]
        p_foot.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_f = p_foot.add_run()
        fldSimple = OxmlElement('w:fldSimple')
        fldSimple.set(qn('w:instr'), 'PAGE')
        run_f._r.append(fldSimple)
        run_f.font.name = 'Times New Roman'
        run_f.font.size = Pt(9)
        run_f.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    # Read Markdown source
    with open(MD_FILE, 'r', encoding='utf-8') as f:
        md_text = f.read()

    lines = md_text.split('\n')

    i = 0
    in_references = False

    while i < len(lines):
        line = lines[i].strip()

        # Blank line
        if not line:
            i += 1
            continue

        # Title (starts with # Predictability...)
        if line.startswith('# Predictability'):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(20)
            p.paragraph_format.space_after = Pt(10)
            run = p.add_run("Predictability, Explainability, and Risk-Aware Trading in Indian Equity Markets: An Empirical Study of Technical Indicators, Financial Sentiment, and Overnight Market Signals")
            run.bold = True
            run.font.size = Pt(17)
            run.font.name = 'Times New Roman'
            run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A) # Deep Navy
            i += 1
            continue

        # Author / Metadata block
        if line.startswith('**Vedant Desai**') or line.startswith('**Vedant Kumar**'):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run("Vedant Desai\n")
            run.bold = True
            run.font.size = Pt(11)
            run.font.name = 'Times New Roman'

            run_aff = p.add_run("Department of Computer Engineering\nK. J. Somaiya College of Engineering, Somaiya Vidyavihar University, Mumbai, India\nvedant.ud@somaiya.edu\nWorking Paper — Pre-Submission Manuscript • November 2026")
            run_aff.font.size = Pt(9.5)
            run_aff.font.italic = True
            run_aff.font.name = 'Times New Roman'
            run_aff.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
            i += 1
            while i < len(lines) and not (lines[i].strip().startswith('## ') or lines[i].strip().startswith('---')):
                i += 1
            if i < len(lines) and lines[i].strip().startswith('---'):
                i += 1
            continue

        # Abstract & Keywords in DOCX
        if line.startswith('## 1. Abstract'):
            i += 1
            abstract_paras = []
            while i < len(lines) and not lines[i].strip().startswith('## 2. Keywords'):
                if lines[i].strip():
                    abstract_paras.append(lines[i].strip())
                i += 1
            keywords_text = ""
            if i < len(lines) and lines[i].strip().startswith('## 2. Keywords'):
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('## '):
                    if lines[i].strip():
                        keywords_text += lines[i].strip() + " "
                    i += 1
            clean_kw = clean_xml_string(keywords_text.replace('`', '').strip())

            p_abs_title = doc.add_paragraph()
            p_abs_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_abs_title.paragraph_format.space_before = Pt(12)
            p_abs_title.paragraph_format.space_after = Pt(4)
            r_abs_t = p_abs_title.add_run("ABSTRACT")
            r_abs_t.bold = True
            r_abs_t.font.name = 'Times New Roman'
            r_abs_t.font.size = Pt(10)
            r_abs_t.font.color.rgb = RGBColor(0x0F, 0x29, 0x4A)

            p_abs = doc.add_paragraph()
            p_abs.paragraph_format.left_indent = Inches(0.4)
            p_abs.paragraph_format.right_indent = Inches(0.4)
            p_abs.paragraph_format.space_after = Pt(6)
            p_abs.paragraph_format.line_spacing = 1.15
            p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            r_abs = p_abs.add_run(clean_xml_string(' '.join(abstract_paras)))
            r_abs.font.name = 'Times New Roman'
            r_abs.font.size = Pt(9.5)

            p_kw = doc.add_paragraph()
            p_kw.paragraph_format.left_indent = Inches(0.4)
            p_kw.paragraph_format.right_indent = Inches(0.4)
            p_kw.paragraph_format.space_after = Pt(14)
            p_kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            r_kwt = p_kw.add_run("Index Terms / Keywords— ")
            r_kwt.bold = True
            r_kwt.font.name = 'Times New Roman'
            r_kwt.font.size = Pt(9)
            r_kw = p_kw.add_run(clean_kw)
            r_kw.font.name = 'Times New Roman'
            r_kw.font.size = Pt(9)
            r_kw.font.italic = True
            continue

        # Section Heading 1 (## X. Title)
        if line.startswith('## '):
            heading_text = line[3:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(18)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run(heading_text)
            run.bold = True
            run.font.size = Pt(13)
            run.font.name = 'Times New Roman'
            run.font.color.rgb = RGBColor(0x0F, 0x29, 0x4A) # Navy
            if '21. References' in heading_text:
                in_references = True
            i += 1
            continue

        # Subsection Heading 2 (### X.Y Title)
        if line.startswith('### '):
            heading_text = line[4:].strip()
            # If it's a table header, we will handle table separately
            if heading_text.startswith('Table '):
                # Map to corresponding CSV
                tbl_num = heading_text.split(':')[0].replace('Table ', '').strip()
                csv_map = {
                    '1': ('table1_walk_forward_10stocks.csv', heading_text),
                    '2': ('table2_baseline_benchmarks.csv', heading_text),
                    '3': ('table3_seven_architectures.csv', heading_text),
                    '4': ('table4_deep_learning_diagnostics.csv', heading_text),
                    '5': ('table5_entity_finbert_sentfin.csv', heading_text),
                    '6': ('table6_sentiment_ablation.csv', heading_text),
                    '7': ('table7_lead_lag_correlations.csv', heading_text),
                    '8': ('table8_sentiment_quintiles.csv', heading_text),
                    '9': ('table9_big_moves_coverage.csv', heading_text),
                    '10': ('table10_accuracy_pitfalls_deconstruction.csv', heading_text),
                    '11': ('table11_overnight_opening_forecast.csv', heading_text),
                    '12': ('table12_actionable_strategies.csv', heading_text),
                    '13': ('table13_methodological_disclosures.csv', heading_text)
                }
                if tbl_num in csv_map:
                    csv_file, caption = csv_map[tbl_num]
                    add_styled_table(doc, os.path.join(TABLES_DIR, csv_file), caption)
                # Skip markdown table rows until next blank or header
                i += 1
                while i < len(lines) and (lines[i].strip().startswith('|') or not lines[i].strip() or lines[i].strip().startswith('*Note')):
                    i += 1
                continue
            else:
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(12)
                p.paragraph_format.space_after = Pt(4)
                run = p.add_run(heading_text)
                run.bold = True
                run.font.size = Pt(11)
                run.font.name = 'Times New Roman'
                run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x5F)
                i += 1
                continue

        # Figures in markdown: ![Figure X](figures/figX...)
        if line.startswith('![Figure'):
            cap_line = lines[i+1].strip() if i+1 < len(lines) else ""
            clean_cap = cap_line.strip('*')
            m = re.search(r'\((figures/fig[0-9a-zA-Z_]+\.png)\)', line)
            if m:
                fig_filename = os.path.basename(m.group(1))
                add_styled_figure(doc, fig_filename, clean_cap)
            i += 2
            continue

        # Handle ASCII code blocks
        if line.startswith('```'):
            code_block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_block.append(lines[i])
                i += 1
            i += 1 # skip closing ```
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(8)
            p.paragraph_format.left_indent = Inches(0.4)
            run = p.add_run('\n'.join(code_block))
            run.font.name = 'Consolas'
            run.font.size = Pt(7.5)
            run.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)
            continue

        # Display equations in DOCX
        if line.startswith('$$'):
            eq_lines = []
            if line.endswith('$$') and len(line) > 2:
                eq_str = line[2:-2].strip()
            else:
                eq_lines.append(line.lstrip('$').strip())
                i += 1
                while i < len(lines) and not lines[i].strip().endswith('$$'):
                    eq_lines.append(lines[i].strip())
                    i += 1
                if i < len(lines):
                    eq_lines.append(lines[i].strip().rstrip('$').strip())
                eq_str = ' '.join(eq_lines)

            p_eq = doc.add_paragraph()
            p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_eq.paragraph_format.space_before = Pt(6)
            p_eq.paragraph_format.space_after = Pt(6)
            p_eq.paragraph_format.left_indent = Inches(0.5)
            p_eq.paragraph_format.right_indent = Inches(0.5)
            run_eq = p_eq.add_run(clean_xml_string(eq_str))
            run_eq.font.name = 'Cambria Math'
            run_eq.font.size = Pt(10)
            run_eq.font.italic = True
            run_eq.font.color.rgb = RGBColor(0x0F, 0x29, 0x4A)
            i += 1
            continue

        # Regular paragraph
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.15

        # References format [X] Author...
        if in_references and line.startswith('['):
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.first_line_indent = Inches(-0.3)
            p.paragraph_format.space_after = Pt(3.5)
            run = p.add_run(clean_xml_string(line))
            run.font.name = 'Times New Roman'
            run.font.size = Pt(9)
            i += 1
            continue

        # Format bullet list
        if line.startswith('- ') or line.startswith('* '):
            p.paragraph_format.left_indent = Inches(0.25)
            clean_line = line[2:]
            run = p.add_run(clean_xml_string("- " + clean_line))
            run.font.name = 'Times New Roman'
            run.font.size = Pt(9.5)
        elif re.match(r'^[0-9]+\.\s', line):
            p.paragraph_format.left_indent = Inches(0.25)
            run = p.add_run(clean_xml_string(line))
            run.font.name = 'Times New Roman'
            run.font.size = Pt(9.5)
        else:
            run = p.add_run(clean_xml_string(line))
            run.font.name = 'Times New Roman'
            run.font.size = Pt(10)

        i += 1

    doc.save(OUTPUT_DOCX)
    print(f"Word document saved successfully to {OUTPUT_DOCX}")

if __name__ == '__main__':
    main()
