# tools/formatter.py

import os
from fpdf import FPDF
from datetime import datetime

OUTPUT_DIR = "data/appeals"

def generate_appeal_pdf(claim_id: str, appeal_text: str) -> str:
    """
    Converts LLM-generated appeal text into a simple PDF.
    Returns the file path.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    filename = f"appeal_{claim_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.pdf"
    filepath = os.path.join(OUTPUT_DIR, filename)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for line in appeal_text.split('\n'):
        pdf.multi_cell(0, 10, line.strip())

    pdf.output(filepath)
    return filepath
# Convert appeal text to PDF
