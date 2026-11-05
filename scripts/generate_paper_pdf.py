"""
generate_paper_pdf.py
Renders the complete research paper into an academic publication-ready PDF
using headless Microsoft Edge and professional academic typography CSS.
"""

import os
import re
import subprocess
import pandas as pd

MD_FILE = "research_paper/research_paper_final.md"
HTML_FILE = "research_paper/paper_rendered.html"
PDF_FILE = "research_paper/Stock_Market_Predictor_Research_Paper.pdf"
FIGURES_DIR = "research_paper/figures"
TABLES_DIR = "research_paper/tables"

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

def csv_to_html_table(csv_path, caption):
    if not os.path.exists(csv_path):
        return ""
    df = pd.read_csv(csv_path)

    html = [f'<div class="table-container">']
    html.append(f'<div class="table-caption"><strong>{caption}</strong></div>')
    html.append('<table>')

    # Header
    html.append('<thead><tr>')
    for col in df.columns:
        clean_col = str(col).replace('_', ' ')
        html.append(f'<th>{clean_col}</th>')
    html.append('</tr></thead>')

    # Body
    html.append('<tbody>')
    for idx, row in df.iterrows():
        row_cls = 'even' if idx % 2 == 0 else 'odd'
        html.append(f'<tr class="{row_cls}">')
        for i, val in enumerate(row):
            txt = str(val) if pd.notna(val) else '—'
            align = 'left' if i == 0 else 'center'
            html.append(f'<td style="text-align: {align};">{txt}</td>')
        html.append('</tr>')
    html.append('</tbody>')
    html.append('</table></div>')
    return '\n'.join(html)

def build_academic_html():
    with open(MD_FILE, 'r', encoding='utf-8') as f:
        md_text = f.read()

    lines = md_text.split('\n')
    body_html = []

    i = 0
    in_references = False

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Title
        if line.startswith('# Predictability'):
            body_html.append(f'<h1 class="paper-title">Predictability, Explainability, and Risk-Aware Trading in Indian Equity Markets: An Empirical Study of Technical Indicators, Financial Sentiment, and Overnight Market Signals</h1>')
            i += 1
            continue

        # Author
        if line.startswith('**Vedant Desai**') or line.startswith('**Vedant Kumar**'):
            body_html.append("""
            <div class="author-block">
                <div class="author-name">Vedant Desai</div>
                <div class="author-affiliation">Department of Computer Engineering<br>K. J. Somaiya College of Engineering, Somaiya Vidyavihar University, Mumbai, India<br><i>Email: vedant.ud@somaiya.edu</i></div>
                <div class="paper-meta">Working Paper — Pre-Submission Manuscript &bull; November 2026</div>
            </div>
            """)
            i += 1
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
            body_html.append(f"""
            <div class="abstract-box">
                <div class="abstract-title">ABSTRACT</div>
                <div class="abstract-content">{' '.join(abstract_paras)}</div>
                <div class="keywords-content"><strong>Index Terms / Keywords—</strong> {clean_kw}</div>
            </div>
            """)
            continue

        # Section Heading (## X. Title)
        if line.startswith('## '):
            heading = line[3:].strip()
            body_html.append(f'<h2 class="section-heading">{heading}</h2>')
            if '21. References' in heading:
                in_references = True
            i += 1
            continue

        # Subsection Heading (### X.Y Title)
        if line.startswith('### '):
            heading = line[4:].strip()
            if heading.startswith('Table '):
                tbl_num = heading.split(':')[0].replace('Table ', '').strip()
                csv_map = {
                    '1': ('table1_walk_forward_10stocks.csv', heading),
                    '2': ('table2_baseline_benchmarks.csv', heading),
                    '3': ('table3_seven_architectures.csv', heading),
                    '4': ('table4_deep_learning_diagnostics.csv', heading),
                    '5': ('table5_entity_finbert_sentfin.csv', heading),
                    '6': ('table6_sentiment_ablation.csv', heading),
                    '7': ('table7_lead_lag_correlations.csv', heading),
                    '8': ('table8_sentiment_quintiles.csv', heading),
                    '9': ('table9_big_moves_coverage.csv', heading),
                    '10': ('table10_accuracy_pitfalls_deconstruction.csv', heading),
                    '11': ('table11_overnight_opening_forecast.csv', heading),
                    '12': ('table12_actionable_strategies.csv', heading),
                    '13': ('table13_methodological_disclosures.csv', heading)
                }
                if tbl_num in csv_map:
                    csv_file, caption = csv_map[tbl_num]
                    tbl_html = csv_to_html_table(os.path.join(TABLES_DIR, csv_file), caption)
                    body_html.append(tbl_html)
                i += 1
                while i < len(lines) and (lines[i].strip().startswith('|') or not lines[i].strip() or lines[i].strip().startswith('*Note')):
                    i += 1
                continue
            else:
                body_html.append(f'<h3 class="subsection-heading">{heading}</h3>')
                i += 1
                continue

        # Figures: ![Figure X](figures/figX...)
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
                    <div class="figure-caption"><em>{clean_cap}</em></div>
                </div>
                """)
            i += 2
            continue

        # Code block
        if line.startswith('```'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1
            code_str = '\n'.join(code_lines)
            body_html.append(f'<pre class="ascii-diagram"><code>{code_str}</code></pre>')
            continue

        # Display Equation ($$...$$)
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
            body_html.append(f'<div class="math-display"><em>{eq_str}</em></div>')
            i += 1
            continue

        # Bullet list
        if line.startswith('- ') or line.startswith('* '):
            body_html.append(f'<ul><li>{line[2:]}</li></ul>')
        elif re.match(r'^[0-9]+\.\s', line):
            body_html.append(f'<div class="numbered-item">{line}</div>')
        elif in_references and line.startswith('['):
            body_html.append(f'<div class="reference-item">{line}</div>')
        else:
            body_html.append(f'<p>{line}</p>')

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
    }}
    .subsection-heading {{
        font-size: 10pt;
        font-weight: bold;
        color: #1E3A5F;
        margin-top: 10pt;
        margin-bottom: 3pt;
        page-break-after: avoid;
    }}
    p {{
        margin-top: 0;
        margin-bottom: 6pt;
        text-align: justify;
    }}
    ul {{
        margin-top: 0;
        margin-bottom: 5pt;
        padding-left: 16pt;
    }}
    li {{
        margin-bottom: 2.5pt;
    }}
    .numbered-item {{
        margin-left: 12pt;
        margin-bottom: 3.5pt;
        text-align: justify;
    }}
    .math-display {{
        text-align: center;
        margin: 6pt 16pt;
        padding: 4pt 8pt;
        font-family: 'Cambria Math', 'DejaVu Serif', 'Times New Roman', serif;
        font-size: 9.2pt;
        background-color: #F8FAFC;
        border-left: 3px solid #1E3A5F;
        border-radius: 2px;
        page-break-inside: avoid;
    }}
    .table-container {{
        margin-top: 10pt;
        margin-bottom: 12pt;
        page-break-inside: avoid;
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
        padding: 4pt 5pt;
        border: 1px solid #CBD5E1;
        text-align: center;
    }}
    td {{
        padding: 3.5pt 4.5pt;
        border: 1px solid #E2E8F0;
    }}
    tr.even td {{
        background-color: #F8FAFC;
    }}
    tr.odd td {{
        background-color: #FFFFFF;
    }}
    .figure-container {{
        text-align: center;
        margin-top: 12pt;
        margin-bottom: 12pt;
        page-break-inside: avoid;
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
        margin-top: 6pt;
        margin-bottom: 8pt;
    }}
    .reference-item {{
        font-size: 8.2pt;
        line-height: 1.32;
        margin-left: 18pt;
        text-indent: -18pt;
        margin-bottom: 3.5pt;
        text-align: justify;
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
    print(f"HTML rendered to {HTML_FILE}")

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
