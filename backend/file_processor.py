import pandas as pd
import io
import PyPDF2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import tempfile
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def read_excel(file_bytes):
    return pd.read_excel(io.BytesIO(file_bytes))

def read_pdf(file_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text[:5000]

def generate_chart(df, x_col=None, y_col=None):
    if df.empty:
        return None
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(num_cols) < 1:
        return None
    y_col = y_col or num_cols[0]
    x_col = x_col or df.columns[0]
    plt.figure(figsize=(8, 4))
    plt.plot(df[x_col], df[y_col], marker='o', color='#d4af37', linewidth=2)
    plt.title(f'Tendencia de {y_col}', fontsize=12)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; border-radius:8px; margin-top:10px;" />'

def generate_pdf_report(title, content_lines):
    temp_dir = tempfile.gettempdir()
    pdf_path = os.path.join(temp_dir, f"reporte_{title.replace(' ', '_')}.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 12))
    for line in content_lines.split('\n'):
        story.append(Paragraph(line, styles['Normal']))
        story.append(Spacer(1, 6))
    doc.build(story)
    return pdf_path

def generate_excel_report(df, filename="analisis.xlsx"):
    temp_dir = tempfile.gettempdir()
    excel_path = os.path.join(temp_dir, filename)
    df.to_excel(excel_path, index=False)
    return excel_path