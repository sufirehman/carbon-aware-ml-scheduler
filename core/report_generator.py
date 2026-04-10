from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_report(savings, best, worst):
    file = "carbon_report.pdf"
    doc = SimpleDocTemplate(file)

    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Carbon-Aware ML Training Report", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Carbon Savings: {savings:.2f}%", styles["Normal"]))
    content.append(Paragraph(f"Best Window Carbon: {best['avg_carbon']}", styles["Normal"]))
    content.append(Paragraph(f"Worst Window Carbon: {worst['avg_carbon']}", styles["Normal"]))

    doc.build(content)
    return file