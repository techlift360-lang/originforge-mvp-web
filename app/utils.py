import os
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from datetime import datetime

EXPORT_DIR = "data/exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

def export_csv(df, name="run"):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(EXPORT_DIR, f"{name}_{ts}.csv")
    df.to_csv(path, index=False)
    return path

def export_pdf(summary: dict, name="policy_brief"):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(EXPORT_DIR, f"{name}_{ts}.pdf")
    c = canvas.Canvas(path, pagesize=LETTER)
    w, h = LETTER
    y = h - 72
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, y, "OriginForge — Policy Brief (v1)")
    y -= 28
    c.setFont("Helvetica", 10)
    for k, v in summary.items():
        c.drawString(72, y, f"{k}: {v}")
        y -= 14
        if y < 72:
            c.showPage(); y = h - 72
    c.showPage()
    c.save()
    return path
