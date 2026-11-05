"""
generate_paper_docx.py
Compiles the complete academic research paper into an editable, publication-grade
Microsoft Word document (research_paper/Stock_Market_Predictor_Research_Paper.docx).
Converts all equations into native Word OMML mathematical objects, formats all inline
Markdown tokens (bold, italic, code) into styled Word runs, and builds publication-styled tables.
"""

import os
import re
import html
import pandas as pd
import latex2mathml.converter
from lxml import etree

import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import qn

OUTPUT_DOCX = "research_paper/Stock_Market_Predictor_Research_Paper.docx"
MD_FILE = "research_paper/research_paper_final.md"
FIGURES_DIR = "research_paper/figures"
TABLES_DIR = "research_paper/tables"

# Locate MML2OMML.XSL stylesheet for native Word MathML -> OMML conversion
XSLT_CANDIDATES = [
    r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL",
    r"C:\Program Files (x86)\Microsoft Office\root\Office16\MML2OMML.XSL",
    r"C:\Program Files\Microsoft Office\Office16\MML2OMML.XSL",
    r"C:\Program Files (x86)\Microsoft Office\Office16\MML2OMML.XSL",
    r"C:\Program Files\Microsoft Office\Office15\MML2OMML.XSL",
    r"C:\Program Files (x86)\Microsoft Office\Office15\MML2OMML.XSL",
]

transform = None
for cand in XSLT_CANDIDATES:
    if os.path.exists(cand):
        try:
            xslt_tree = etree.parse(cand)
            transform = etree.XSLT(xslt_tree)
            break
        except Exception:
            pass

def clean_xml_string(s):
    if not isinstance(s, str):
        s = str(s)
    return re.sub(r'[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]', '', s)

def latex_to_omml(latex, is_display=False):
    """
    Converts LaTeX math to native Word OMML XML.
    """
    if transform is None:
        return None
    try:
        mml = latex2mathml.converter.convert(latex)
        tree = etree.fromstring(mml)
        omml = transform(tree)
        root = omml.getroot()
        xml_str = etree.tostring(root, encoding='utf-8').decode('utf-8')
        if is_display:
            return f'<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">{xml_str}</m:oMathPara>'
        else:
            return xml_str
    except Exception:
        return None

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

def safe_split_table_row(line):
    """
    Splits markdown table row by '|' while protecting pipes inside math formulas.
    """
    def mask_pipes_in_math(m):
        math_content = m.group(1)
        masked = math_content.replace('|', '__PIPE_CHAR__')
        return f"${masked}$"

    line_masked = re.sub(r'\\\|', '__ESCAPED_PIPE__', line)
    line_masked = re.sub(r'(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)', mask_pipes_in_math, line_masked)

    raw_cells = line_masked.strip().strip('|').split('|')
    clean_cells = []
    for c in raw_cells:
        unmasked = c.replace('__PIPE_CHAR__', '|').replace('__ESCAPED_PIPE__', '|').strip()
        clean_cells.append(unmasked)
    return clean_cells

def add_formatted_runs(p, text, default_size=10, default_color=None):
    """
    Tokenizes text and appends formatted runs or native OMML math to paragraph.
    Handles nested math inside **bold** and *italic* tokens seamlessly.
    """
    if not text:
        return

    math_map = {}
    code_map = {}

    # Step 1: extract display equations
    def repl_disp(m):
        k = f"__OMML_DISP_{len(math_map)}__"
        eq = m.group(1).strip()
        omml = latex_to_omml(eq, is_display=False)
        math_map[k] = (omml, eq)
        return k
    text = re.sub(r'\$\$(.*?)\$\$', repl_disp, text, flags=re.DOTALL)

    # Step 2: extract inline equations
    def repl_inl(m):
        k = f"__OMML_INL_{len(math_map)}__"
        eq = m.group(1).strip()
        omml = latex_to_omml(eq, is_display=False)
        math_map[k] = (omml, eq)
        return k
    text = re.sub(r'(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)', repl_inl, text)

    # Step 3: extract code
    def repl_code(m):
        k = f"__CODE_TOK_{len(code_map)}__"
        code_map[k] = m.group(1)
        return k
    text = re.sub(r'`([^`]+?)`', repl_code, text)

    # Step 4: Tokenize markdown bold and italic
    tok_regex = re.compile(r'(\*\*.*?\*\*|\*[^\*\n]+?\*)')
    tokens = tok_regex.split(text)

    for token in tokens:
        if not token:
            continue
        is_bold = False
        is_italic = False
        content = token
        if token.startswith('**') and token.endswith('**') and len(token) >= 4:
            is_bold = True
            content = token[2:-2]
        elif token.startswith('*') and token.endswith('*') and len(token) >= 2:
            is_italic = True
            content = token[1:-1]

        # Inside content, handle math placeholders and code placeholders
        sub_regex = re.compile(r'(__OMML_(?:DISP|INL)_\d+__|__CODE_TOK_\d+__)')
        sub_tokens = sub_regex.split(content)

        for sub in sub_tokens:
            if not sub:
                continue
            if sub in math_map:
                omml_xml, raw_eq = math_map[sub]
                if omml_xml:
                    try:
                        p._p.append(parse_xml(omml_xml))
                        continue
                    except Exception:
                        pass
                r = p.add_run(clean_xml_string(raw_eq))
                r.italic = True
                r.font.name = 'Cambria Math'
            elif sub in code_map:
                code_text = code_map[sub]
                r = p.add_run(clean_xml_string(code_text))
                r.font.name = 'Consolas'
                r.font.size = Pt(default_size * 0.9)
                if default_color:
                    r.font.color.rgb = default_color
            else:
                r = p.add_run(clean_xml_string(sub))
                r.font.name = 'Times New Roman'
                r.font.size = Pt(default_size)
                if is_bold:
                    r.bold = True
                if is_italic:
                    r.italic = True
                if default_color:
                    r.font.color.rgb = default_color

def parse_md_table_to_docx(doc, caption, header_line, delim_line, data_lines, note_line=None):
    if caption:
        p_cap = doc.add_paragraph()
        p_cap.paragraph_format.space_before = Pt(14)
        p_cap.paragraph_format.space_after = Pt(4)
        p_cap.paragraph_format.keep_with_next = True
        add_formatted_runs(p_cap, f"**{caption}**", default_size=10, default_color=RGBColor(0x1A, 0x36, 0x5D))

    headers = safe_split_table_row(header_line)
    delims = safe_split_table_row(delim_line)
    rows = [safe_split_table_row(dl) for dl in data_lines]

    table = doc.add_table(rows=len(rows)+1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    # Header row
    hdr_cells = table.rows[0].cells
    trPr = table.rows[0]._tr.get_or_add_trPr()
    trPr.append(OxmlElement('w:tblHeader'))
    trPr.append(OxmlElement('w:cantSplit'))

    for i, h_text in enumerate(headers):
        set_cell_background(hdr_cells[i], '1A365D') # Deep Navy
        set_cell_margins(hdr_cells[i], 120, 120, 100, 100)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_formatted_runs(p, h_text, default_size=8.5, default_color=RGBColor(0xFF, 0xFF, 0xFF))

    # Data rows
    for row_idx, row in enumerate(rows):
        row_cells = table.rows[row_idx + 1].cells
        r_trPr = table.rows[row_idx + 1]._tr.get_or_add_trPr()
        r_trPr.append(OxmlElement('w:cantSplit'))
        bg_color = 'F8FAFC' if row_idx % 2 == 1 else 'FFFFFF'

        for col_idx in range(len(headers)):
            cell_val = row[col_idx] if col_idx < len(row) else ''
            set_cell_background(row_cells[col_idx], bg_color)
            set_cell_margins(row_cells[col_idx], 80, 80, 90, 90)
            p = row_cells[col_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            add_formatted_runs(p, cell_val, default_size=8.5, default_color=RGBColor(0x2D, 0x37, 0x48))

    if note_line:
        p_note = doc.add_paragraph()
        p_note.paragraph_format.space_before = Pt(3)
        p_note.paragraph_format.space_after = Pt(8)
        clean_note = note_line.strip().lstrip('*').rstrip('*').strip()
        add_formatted_runs(p_note, f"*{clean_note}*", default_size=8.2, default_color=RGBColor(0x64, 0x74, 0x8B))
    else:
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
    p_img.paragraph_format.keep_with_next = True
    run_img = p_img.add_run()
    run_img.add_picture(img_path, width=Inches(6.2))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(12)
    clean_cap = caption.strip('*').strip()
    add_formatted_runs(p_cap, f"*{clean_cap}*", default_size=9, default_color=RGBColor(0x4A, 0x55, 0x68))

def main():
    print("Building Publication-Ready Word Document (.docx)...")
    doc = Document()

    # Page setup: Standard Letter with 1-inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.different_first_page_header_footer = True

        # Header for subsequent pages
        header = section.header
        p_head = header.paragraphs[0]
        p_head.text = "Stock Market Predictor Research Paper — Indian Equity Markets"
        p_head.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        for r in p_head.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(8.5)
            r.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

        # Footer for subsequent pages
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
        raw_line = lines[i]
        line = raw_line.strip()

        if not line:
            i += 1
            continue

        # Title
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

        # Author and Institutional Block (Exactly 3 authors, zero emails, zero roll numbers)
        if line.startswith('**Smit Patil**') or line.startswith('**Vedant Desai**'):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run("Smit Patil, Vedant Narayan, and Vedant Desai\n")
            run.bold = True
            run.font.size = Pt(11)
            run.font.name = 'Times New Roman'
            run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

            run_aff = p.add_run(
                "Department of Computer Engineering\n"
                "K. J. Somaiya College of Engineering, Somaiya Vidyavihar University, Mumbai, India\n"
                "Working Paper — Pre-Submission Manuscript • November 2026"
            )
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

        # Abstract & Keywords Box
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
            clean_kw = keywords_text.replace('`', '').strip()

            p_abs_title = doc.add_paragraph()
            p_abs_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_abs_title.paragraph_format.space_before = Pt(14)
            p_abs_title.paragraph_format.space_after = Pt(4)
            p_abs_title.paragraph_format.keep_with_next = True
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
            add_formatted_runs(p_abs, ' '.join(abstract_paras), default_size=9.5)

            p_kw = doc.add_paragraph()
            p_kw.paragraph_format.left_indent = Inches(0.4)
            p_kw.paragraph_format.right_indent = Inches(0.4)
            p_kw.paragraph_format.space_after = Pt(14)
            p_kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            r_kwt = p_kw.add_run("Index Terms / Keywords— ")
            r_kwt.bold = True
            r_kwt.font.name = 'Times New Roman'
            r_kwt.font.size = Pt(9)
            add_formatted_runs(p_kw, f"*{clean_kw}*", default_size=9)
            continue

        # Section Heading 1 (## X. Title)
        if line.startswith('## '):
            heading_text = line[3:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(18)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            add_formatted_runs(p, heading_text, default_size=13, default_color=RGBColor(0x0F, 0x29, 0x4A))
            for r in p.runs:
                r.bold = True
            if '21. References' in heading_text:
                in_references = True
            i += 1
            continue

        # Subsection or Table Heading (### Table X: ... or ### X.Y Title)
        if line.startswith('### '):
            heading_text = line[4:].strip()
            if heading_text.startswith('Table '):
                caption = heading_text
                i += 1
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if i < len(lines) and lines[i].strip().startswith('|'):
                    header_line = lines[i].strip()
                    i += 1
                    delim_line = lines[i].strip() if i < len(lines) else ""
                    i += 1
                    data_lines = []
                    while i < len(lines) and lines[i].strip().startswith('|'):
                        data_lines.append(lines[i].strip())
                        i += 1
                    note_line = None
                    while i < len(lines) and not lines[i].strip():
                        i += 1
                    if i < len(lines) and lines[i].strip().startswith('*Note'):
                        note_line = lines[i].strip()
                        i += 1
                    parse_md_table_to_docx(doc, caption, header_line, delim_line, data_lines, note_line)
                    continue
                else:
                    p = doc.add_paragraph()
                    p.paragraph_format.space_before = Pt(12)
                    p.paragraph_format.space_after = Pt(4)
                    p.paragraph_format.keep_with_next = True
                    add_formatted_runs(p, heading_text, default_size=11, default_color=RGBColor(0x1E, 0x3A, 0x5F))
                    for r in p.runs:
                        r.bold = True
                    continue
            else:
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(12)
                p.paragraph_format.space_after = Pt(4)
                p.paragraph_format.keep_with_next = True
                add_formatted_runs(p, heading_text, default_size=11, default_color=RGBColor(0x1E, 0x3A, 0x5F))
                for r in p.runs:
                    r.bold = True
                i += 1
                continue

        # Sub-subsection (#### X.Y.Z Title)
        if line.startswith('#### '):
            heading_text = line[5:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
            add_formatted_runs(p, heading_text, default_size=10, default_color=RGBColor(0x33, 0x41, 0x55))
            for r in p.runs:
                r.bold = True
                r.italic = True
            i += 1
            continue

        # Figures: ![Figure X](figures/figX...)
        if line.startswith('![Figure'):
            cap_line = lines[i+1].strip() if i+1 < len(lines) else ""
            clean_cap = cap_line.strip('*')
            m = re.search(r'\((figures/fig[0-9a-zA-Z_]+\.png)\)', line)
            if m:
                fig_filename = os.path.basename(m.group(1))
                add_styled_figure(doc, fig_filename, clean_cap)
            i += 2
            continue

        # ASCII Code Block (``` ... ```)
        if line.startswith('```'):
            code_block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_block.append(lines[i])
                i += 1
            i += 1
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(8)
            p.paragraph_format.left_indent = Inches(0.4)
            run = p.add_run(clean_xml_string('\n'.join(code_block)))
            run.font.name = 'Consolas'
            run.font.size = Pt(7.5)
            run.font.color.rgb = RGBColor(0x2D, 0x37, 0x48)
            continue

        # Display Equations ($$...$$)
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
            p_eq.paragraph_format.space_before = Pt(8)
            p_eq.paragraph_format.space_after = Pt(8)
            p_eq.paragraph_format.left_indent = Inches(0.5)
            p_eq.paragraph_format.right_indent = Inches(0.5)

            omml_xml = latex_to_omml(eq_str, is_display=True)
            if omml_xml:
                try:
                    p_eq._p.append(parse_xml(omml_xml))
                    i += 1
                    continue
                except Exception:
                    pass

            run_eq = p_eq.add_run(clean_xml_string(eq_str))
            run_eq.font.name = 'Cambria Math'
            run_eq.font.size = Pt(10)
            run_eq.font.italic = True
            run_eq.font.color.rgb = RGBColor(0x0F, 0x29, 0x4A)
            i += 1
            continue

        # Standalone Markdown Table (not prefixed by ### Table)
        if line.startswith('|') and i + 1 < len(lines) and lines[i+1].strip().startswith('|'):
            header_line = line
            i += 1
            delim_line = lines[i].strip()
            i += 1
            data_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                data_lines.append(lines[i].strip())
                i += 1
            note_line = None
            if i < len(lines) and lines[i].strip().startswith('*Note'):
                note_line = lines[i].strip()
                i += 1
            parse_md_table_to_docx(doc, None, header_line, delim_line, data_lines, note_line)
            continue

        # References format in Section 21
        if in_references and line.startswith('['):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.first_line_indent = Inches(-0.3)
            p.paragraph_format.space_after = Pt(3.5)
            p.paragraph_format.line_spacing = 1.15
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            add_formatted_runs(p, line, default_size=9)
            i += 1
            continue

        # Bullet List (- or *)
        if line.startswith('- ') or line.startswith('* '):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            run_bullet = p.add_run("•  ")
            run_bullet.font.name = 'Times New Roman'
            run_bullet.font.size = Pt(9.5)
            clean_line = line[2:].strip()
            add_formatted_runs(p, clean_line, default_size=9.5)
            i += 1
            continue

        # Numbered List (1. , 2. )
        if re.match(r'^[0-9]+\.\s', line):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.space_after = Pt(3.5)
            p.paragraph_format.line_spacing = 1.15
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            m_num = re.match(r'^([0-9]+\.)\s+(.*)', line)
            if m_num:
                num_label = m_num.group(1) + "  "
                content_text = m_num.group(2)
                r_num = p.add_run(num_label)
                r_num.bold = True
                r_num.font.name = 'Times New Roman'
                r_num.font.size = Pt(9.5)
                add_formatted_runs(p, content_text, default_size=9.5)
            else:
                add_formatted_runs(p, line, default_size=9.5)
            i += 1
            continue

        # Regular Paragraph
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.15
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        add_formatted_runs(p, line, default_size=10)
        i += 1

    doc.save(OUTPUT_DOCX)
    print(f"Word document saved successfully to {OUTPUT_DOCX}")

if __name__ == '__main__':
    main()
