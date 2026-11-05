"""
generate_paper_pdf.py
Compiles the complete academic research paper into a publication-ready PDF
using headless Microsoft Edge, native MathML typesetting, and professional academic CSS typography.
Eliminates all raw Markdown syntax and raw LaTeX commands from the final PDF.
"""

import os
import re
import html
import subprocess
import pandas as pd
import latex2mathml.converter

MD_FILE = "research_paper/research_paper_final.md"
HTML_FILE = "research_paper/paper_rendered.html"
PDF_FILE = "research_paper/Stock_Market_Predictor_Research_Paper.pdf"
FIGURES_DIR = "research_paper/figures"
TABLES_DIR = "research_paper/tables"

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

def render_inline(text):
    """
    Renders inline formatting (math, code, bold, italic) into semantic HTML.
    Prevents markdown operators inside math or code from being corrupted.
    """
    if not text:
        return ""

    # Step 1: protect and convert display equations $$...$$
    disp_map = {}
    def repl_disp(m):
        key = f"__MATH_DISP_{len(disp_map)}__"
        eq = m.group(1).strip()
        try:
            mml = latex2mathml.converter.convert(eq, display='block')
            disp_map[key] = f'<div class="math-display">{mml}</div>'
        except Exception:
            disp_map[key] = f'<div class="math-display"><em>{html.escape(eq)}</em></div>'
        return f"\n\n{key}\n\n"

    text = re.sub(r'\$\$(.*?)\$\$', repl_disp, text, flags=re.DOTALL)

    # Step 2: protect and convert inline equations $...$
    inl_map = {}
    def repl_inl(m):
        key = f"__MATH_INL_{len(inl_map)}__"
        eq = m.group(1).strip()
        try:
            mml = latex2mathml.converter.convert(eq, display='inline')
            inl_map[key] = mml
        except Exception:
            inl_map[key] = f'<em>{html.escape(eq)}</em>'
        return key

    text = re.sub(r'(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)', repl_inl, text)

    # Step 3: inline code `...`
    code_map = {}
    def repl_code(m):
        key = f"__CODE_{len(code_map)}__"
        code_content = html.escape(m.group(1))
        code_map[key] = f'<code class="inline-code">{code_content}</code>'
        return key

    text = re.sub(r'`([^`]+?)`', repl_code, text)

    # Step 4: bold **...**
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

    # Step 5: italic *...* (not matching asterisks inside words)
    text = re.sub(r'(?<!\*)\*([^\*\n]+?)\*(?!\*)', r'<em>\1</em>', text)

    # Step 6: restore inline code
    for k, v in code_map.items():
        text = text.replace(k, v)

    # Step 7: restore inline math
    for k, v in inl_map.items():
        text = text.replace(k, v)

    # Step 8: restore display math
    for k, v in disp_map.items():
        text = text.replace(k, v)

    return text

def safe_split_table_row(line):
    """
    Safely splits a markdown table row by '|' while ignoring pipes inside math formulas (e.g. $|\\Delta|$)
    or escaped pipes (\\|).
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

def parse_md_table_to_html(caption, header_line, delim_line, data_lines, note_line=None):
    """
    Converts a Markdown table into a beautifully styled academic HTML table.
    """
    headers = safe_split_table_row(header_line)
    delims = safe_split_table_row(delim_line)

    aligns = []
    for d in delims:
        if d.startswith(':') and d.endswith(':'):
            aligns.append('center')
        elif d.endswith(':'):
            aligns.append('right')
        else:
            aligns.append('left')

    out = ['<div class="table-container">']
    if caption:
        out.append(f'<div class="table-caption"><strong>{render_inline(caption)}</strong></div>')
    out.append('<table>')
    out.append('<thead><tr>')
    for i, h in enumerate(headers):
        al = aligns[i] if i < len(aligns) else 'left'
        out.append(f'<th style="text-align: {al};">{render_inline(h)}</th>')
    out.append('</tr></thead>')
    out.append('<tbody>')
    for idx, row_line in enumerate(data_lines):
        row_cells = safe_split_table_row(row_line)
        row_cls = 'even' if idx % 2 == 0 else 'odd'
        out.append(f'<tr class="{row_cls}">')
        for i, val in enumerate(row_cells):
            al = aligns[i] if i < len(aligns) else ('left' if i == 0 else 'center')
            out.append(f'<td style="text-align: {al};">{render_inline(val)}</td>')
        out.append('</tr>')
    out.append('</tbody>')
    out.append('</table>')
    if note_line:
        clean_note = note_line.strip().lstrip('*').rstrip('*').strip()
        out.append(f'<div class="table-note"><em>{render_inline(clean_note)}</em></div>')
    out.append('</div>')
    return '\n'.join(out)

def build_academic_html():
    with open(MD_FILE, 'r', encoding='utf-8') as f:
        md_text = f.read()

    lines = md_text.split('\n')
    body_html = []

    i = 0
    in_references = False

    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.strip()

        if not line:
            i += 1
            continue

        # Paper Title
        if line.startswith('# Predictability'):
            body_html.append(f'<h1 class="paper-title">Predictability, Explainability, and Risk-Aware Trading in Indian Equity Markets: An Empirical Study of Technical Indicators, Financial Sentiment, and Overnight Market Signals</h1>')
            i += 1
            continue

        # Author and Institutional Metadata
        if line.startswith('**Smit Patil**') or line.startswith('**Vedant Desai**'):
            body_html.append("""
            <div class="author-block">
                <div class="author-name">Smit Patil &nbsp;&bull;&nbsp; Vedant Narayan &nbsp;&bull;&nbsp; Vedant Desai</div>
                <div class="author-affiliation">Department of Computer Engineering<br>K. J. Somaiya College of Engineering, Somaiya Vidyavihar University, Mumbai, India</div>
                <div class="paper-meta">Working Paper — Pre-Submission Manuscript &bull; November 2026</div>
            </div>
            """)
            i += 1
            # Skip author metadata lines until separator or next section
            while i < len(lines) and not (lines[i].strip().startswith('## ') or lines[i].strip().startswith('---')):
                i += 1
            if i < len(lines) and lines[i].strip().startswith('---'):
                i += 1
            continue

        # Abstract and Keywords Box
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
            full_abstract = render_inline(' '.join(abstract_paras))
            body_html.append(f"""
            <div class="abstract-box">
                <div class="abstract-title">ABSTRACT</div>
                <div class="abstract-content">{full_abstract}</div>
                <div class="keywords-content"><strong>Index Terms / Keywords— </strong>{render_inline(clean_kw)}</div>
            </div>
            """)
            continue

        # Section Heading (## X. Title)
        if line.startswith('## '):
            heading = line[3:].strip()
            body_html.append(f'<h2 class="section-heading">{render_inline(heading)}</h2>')
            if '21. References' in heading:
                in_references = True
            i += 1
            continue

        # Table Heading or Subsection (### Table X: ... or ### X.Y Title)
        if line.startswith('### '):
            heading = line[4:].strip()
            if heading.startswith('Table '):
                caption = heading
                i += 1
                while i < len(lines) and not lines[i].strip():
                    i += 1
                # Read markdown table
                if i < len(lines) and lines[i].strip().startswith('|'):
                    header_line = lines[i].strip()
                    i += 1
                    delim_line = lines[i].strip() if i < len(lines) else ""
                    i += 1
                    data_lines = []
                    while i < len(lines) and lines[i].strip().startswith('|'):
                        data_lines.append(lines[i].strip())
                        i += 1
                    # Check for note line immediately following table
                    note_line = None
                    while i < len(lines) and not lines[i].strip():
                        i += 1
                    if i < len(lines) and lines[i].strip().startswith('*Note'):
                        note_line = lines[i].strip()
                        i += 1
                    tbl_html = parse_md_table_to_html(caption, header_line, delim_line, data_lines, note_line)
                    body_html.append(tbl_html)
                    continue
                else:
                    body_html.append(f'<h3 class="subsection-heading">{render_inline(heading)}</h3>')
                    continue
            else:
                body_html.append(f'<h3 class="subsection-heading">{render_inline(heading)}</h3>')
                i += 1
                continue

        # Sub-subsection (#### X.Y.Z Title)
        if line.startswith('#### '):
            heading = line[5:].strip()
            body_html.append(f'<h4 class="subsubsection-heading">{render_inline(heading)}</h4>')
            i += 1
            continue

        # Figures: ![Figure X: ...](figures/figX...)
        if line.startswith('![Figure'):
            cap_line = lines[i+1].strip() if i+1 < len(lines) else ""
            clean_cap = cap_line.strip('*')
            m = re.search(r'\((figures/fig[0-9a-zA-Z_]+\.png)\)', line)
            if m:
                fig_filename = os.path.basename(m.group(1))
                abs_fig = os.path.abspath(os.path.join(FIGURES_DIR, fig_filename)).replace('\\', '/')
                body_html.append(f"""
                <div class="figure-container">
                    <img src="file:///{abs_fig}" alt="{clean_cap}" class="paper-figure">
                    <div class="figure-caption"><em>{render_inline(clean_cap)}</em></div>
                </div>
                """)
            i += 2
            continue

        # Code block (``` ... ```)
        if line.startswith('```'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1
            code_str = html.escape('\n'.join(code_lines))
            body_html.append(f'<pre class="ascii-diagram"><code>{code_str}</code></pre>')
            continue

        # Standalone Display Equation ($$...$$)
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
            try:
                mml = latex2mathml.converter.convert(eq_str, display='block')
                body_html.append(f'<div class="math-display">{mml}</div>')
            except Exception:
                body_html.append(f'<div class="math-display"><em>{html.escape(eq_str)}</em></div>')
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
            tbl_html = parse_md_table_to_html(None, header_line, delim_line, data_lines, note_line)
            body_html.append(tbl_html)
            continue

        # Bullet List (- or *)
        if line.startswith('- ') or line.startswith('* '):
            list_items = []
            while i < len(lines) and (lines[i].strip().startswith('- ') or lines[i].strip().startswith('* ')):
                item_text = lines[i].strip()[2:].strip()
                list_items.append(f'<li>{render_inline(item_text)}</li>')
                i += 1
            body_html.append(f'<ul class="paper-list">\n' + '\n'.join(list_items) + '\n</ul>')
            continue

        # Numbered List (1. , 2. )
        if re.match(r'^[0-9]+\.\s', line):
            num_items = []
            while i < len(lines) and re.match(r'^[0-9]+\.\s', lines[i].strip()):
                item_text = re.sub(r'^[0-9]+\.\s+', '', lines[i].strip())
                num_items.append(f'<li>{render_inline(item_text)}</li>')
                i += 1
            body_html.append(f'<ol class="paper-list">\n' + '\n'.join(num_items) + '\n</ol>')
            continue

        # Blockquote (> ...)
        if line.startswith('> '):
            quote_lines = []
            while i < len(lines) and lines[i].strip().startswith('>'):
                quote_lines.append(lines[i].strip().lstrip('>').strip())
                i += 1
            full_quote = render_inline(' '.join(quote_lines))
            body_html.append(f'<blockquote class="paper-quote">{full_quote}</blockquote>')
            continue

        # Reference Item in Section 21
        if in_references and line.startswith('['):
            body_html.append(f'<div class="reference-item">{render_inline(line)}</div>')
            i += 1
            continue

        # Regular Paragraph
        body_html.append(f'<p class="paper-p">{render_inline(line)}</p>')
        i += 1

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Predictability, Explainability, and Risk-Aware Trading in Indian Equity Markets</title>
<style>
    @page {{
        size: letter;
        margin: 18mm 16mm 18mm 16mm;
        @top-center {{
            content: "Stock Market Predictor Research Paper — Indian Equity Markets";
            font-family: 'Times New Roman', serif;
            font-size: 8pt;
            color: #64748B;
        }}
        @bottom-center {{
            content: counter(page);
            font-family: 'Times New Roman', serif;
            font-size: 8.5pt;
            color: #1E293B;
        }}
    }}
    @page :first {{
        @top-center {{
            content: '';
        }}
    }}
    body {{
        font-family: 'Times New Roman', 'DejaVu Serif', Georgia, serif;
        font-size: 9.8pt;
        line-height: 1.42;
        color: #1E293B;
        background: #FFFFFF;
        margin: 0;
        padding: 0;
    }}
    .paper-title {{
        font-size: 16pt;
        font-weight: bold;
        text-align: center;
        color: #0F172A;
        margin-top: 4pt;
        margin-bottom: 8pt;
        line-height: 1.22;
    }}
    .author-block {{
        text-align: center;
        margin-bottom: 12pt;
        border-bottom: 1px solid #CBD5E1;
        padding-bottom: 8pt;
    }}
    .author-name {{
        font-size: 10.5pt;
        font-weight: bold;
        color: #0F172A;
        margin-bottom: 2pt;
    }}
    .author-affiliation {{
        font-size: 9pt;
        font-style: italic;
        color: #475569;
        margin-bottom: 3pt;
        line-height: 1.3;
    }}
    .paper-meta {{
        font-size: 8pt;
        color: #64748B;
    }}
    .abstract-box {{
        margin: 8pt 16pt 14pt 16pt;
        padding: 8pt 12pt;
        border-top: 1px solid #CBD5E1;
        border-bottom: 1px solid #CBD5E1;
        background-color: #F8FAFC;
        font-size: 8.5pt;
        line-height: 1.34;
        page-break-inside: avoid;
        break-inside: avoid;
    }}
    .abstract-title {{
        font-weight: bold;
        font-size: 8.5pt;
        text-align: center;
        letter-spacing: 1px;
        color: #0F294A;
        margin-bottom: 4pt;
    }}
    .abstract-content {{
        text-align: justify;
        text-justify: inter-word;
        margin-bottom: 6pt;
    }}
    .keywords-content {{
        font-size: 8pt;
        color: #334155;
        border-top: 1px dashed #CBD5E1;
        padding-top: 4pt;
    }}
    .section-heading {{
        font-size: 12pt;
        font-weight: bold;
        color: #0F294A;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 2pt;
        margin-top: 14pt;
        margin-bottom: 6pt;
        page-break-after: avoid;
        break-after: avoid;
    }}
    .subsection-heading {{
        font-size: 10.5pt;
        font-weight: bold;
        color: #1E3A5F;
        margin-top: 11pt;
        margin-bottom: 4pt;
        page-break-after: avoid;
        break-after: avoid;
    }}
    .subsubsection-heading {{
        font-size: 9.5pt;
        font-weight: bold;
        font-style: italic;
        color: #334155;
        margin-top: 8pt;
        margin-bottom: 3pt;
        page-break-after: avoid;
        break-after: avoid;
    }}
    p.paper-p {{
        margin-top: 0;
        margin-bottom: 6pt;
        text-align: justify;
        text-justify: inter-word;
    }}
    ul.paper-list, ol.paper-list {{
        margin-top: 2pt;
        margin-bottom: 6pt;
        padding-left: 20pt;
    }}
    ul.paper-list li, ol.paper-list li {{
        margin-bottom: 3pt;
        text-align: justify;
        text-justify: inter-word;
    }}
    strong, b {{
        font-weight: bold;
        color: #0F172A;
    }}
    em, i {{
        font-style: italic;
    }}
    code.inline-code {{
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 8.5pt;
        background-color: #F1F5F9;
        padding: 1pt 3pt;
        border-radius: 3px;
        border: 1px solid #E2E8F0;
        color: #0F294A;
    }}
    math {{
        font-family: 'Cambria Math', 'STIX Two Math', 'Latin Modern Math', 'DejaVu Serif', serif;
    }}
    .math-display {{
        text-align: center;
        margin: 8pt 16pt;
        padding: 6pt 10pt;
        background-color: #F8FAFC;
        border-left: 3px solid #1E3A5F;
        border-radius: 2px;
        page-break-inside: avoid;
        break-inside: avoid;
        font-size: 10pt;
    }}
    .table-container {{
        margin-top: 10pt;
        margin-bottom: 12pt;
        page-break-inside: avoid;
        break-inside: avoid;
    }}
    .table-caption {{
        font-size: 8.5pt;
        font-weight: bold;
        color: #0F294A;
        margin-bottom: 3pt;
        text-align: left;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 7.8pt;
        margin-bottom: 3pt;
    }}
    th {{
        background-color: #0F294A;
        color: #FFFFFF;
        font-weight: bold;
        padding: 4.5pt 5pt;
        border: 1px solid #CBD5E1;
        text-align: center;
    }}
    th strong {{
        color: #FFFFFF;
    }}
    td {{
        padding: 3.5pt 4.5pt;
        border: 1px solid #E2E8F0;
        vertical-align: middle;
    }}
    tr.even td {{
        background-color: #F8FAFC;
    }}
    tr.odd td {{
        background-color: #FFFFFF;
    }}
    .table-note {{
        font-size: 7.5pt;
        font-style: italic;
        color: #64748B;
        margin-top: 3pt;
        text-align: justify;
    }}
    .figure-container {{
        text-align: center;
        margin-top: 12pt;
        margin-bottom: 12pt;
        page-break-inside: avoid;
        break-inside: avoid;
    }}
    .paper-figure {{
        max-width: 95%;
        height: auto;
        border: 1px solid #E2E8F0;
        border-radius: 3px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    .figure-caption {{
        font-size: 8.2pt;
        font-style: italic;
        color: #475569;
        margin-top: 3pt;
        text-align: center;
    }}
    .ascii-diagram {{
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 4px;
        padding: 6pt;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 7.2pt;
        line-height: 1.22;
        overflow-x: auto;
        page-break-inside: avoid;
        break-inside: avoid;
        margin-top: 6pt;
        margin-bottom: 8pt;
    }}
    blockquote.paper-quote {{
        margin: 6pt 16pt;
        padding: 4pt 10pt;
        border-left: 3px solid #94A3B8;
        background-color: #F8FAFC;
        font-style: italic;
        color: #334155;
    }}
    .reference-item {{
        font-size: 8.2pt;
        line-height: 1.34;
        margin-left: 20pt;
        text-indent: -20pt;
        margin-bottom: 4pt;
        text-align: justify;
        page-break-inside: avoid;
        break-inside: avoid;
    }}
</style>
</head>
<body>
{''.join(body_html)}
</body>
</html>
"""
    with open(HTML_FILE, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"HTML rendered successfully to {HTML_FILE}")

def convert_html_to_pdf():
    abs_html = os.path.abspath(HTML_FILE)
    abs_pdf = os.path.abspath(PDF_FILE)

    cmd = [
        EDGE_PATH,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={abs_pdf}",
        f"file:///{abs_html}"
    ]
    print(f"Running Microsoft Edge headless to compile PDF...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(abs_pdf):
        size = os.path.getsize(abs_pdf)
        print(f"Publication PDF successfully generated at {PDF_FILE} (Size: {size:,} bytes)")
    else:
        print(f"Error compiling PDF: {res.stderr}")

if __name__ == '__main__':
    build_academic_html()
    convert_html_to_pdf()
