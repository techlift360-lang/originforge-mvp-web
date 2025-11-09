import os
from datetime import datetime
from io import BytesIO

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

# This is mainly for local use; Streamlit Cloud runs fine without using the folder.
EXPORT_DIR = "data/exports"
os.makedirs(EXPORT_DIR, exist_ok=True)


def export_csv(df, name="run"):
    """
    Save a CSV file to data/exports and return the file path.
    (Useful for local runs; on Streamlit Cloud you'll normally use df.to_csv directly.)
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(EXPORT_DIR, f"{name}_{ts}.csv")
    df.to_csv(path, index=False)
    return path


def export_pdf_bytes(summary_lines, title="OriginForge Policy Brief (v1)"):
    """
    Create a simple PDF in memory from a list of text lines.
    Returns raw PDF bytes suitable for Streamlit's download_button.
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)
    w, h = LETTER

    # Title
    y = h - 72
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, y, title)
    y -= 28

    # Body
    c.setFont("Helvetica", 10)
    for line in summary_lines:
        # Simple wrap: if line is too long, cut it (for now)
        text = str(line)
        if len(text) > 110:
            text = text[:107] + "..."
        c.drawString(72, y, text)
        y -= 14
        if y < 72:
            c.showPage()
            y = h - 72
            c.setFont("Helvetica", 10)

    c.showPage()
    c.save()

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
