"""
Report Generator — produces a self-contained HTML screening report.

The report is designed to be opened in a new browser tab and printed to PDF
via window.print(). No server-side PDF library needed.

This is a NON-CLINICAL PROTOTYPE — the report includes all required disclaimers.
"""

from datetime import datetime, timezone
from typing import Optional


def generate_report_html(
    quality_status: str,
    quality_reasons: list[str],
    dr_grade: Optional[int],
    grade_label: str,
    is_referable: Optional[bool],
    referable_label: str,
    confidence: float,
    key_findings: list[str],
    reasoning: str,
    image_quality_note: str,
    model_used: str,
    original_image_b64: str = "",
    enhanced_image_b64: str = "",
) -> str:
    """
    Generate a self-contained HTML report string for the screening result.

    Returns:
        Complete HTML document as a string.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    grade_display = str(dr_grade) if dr_grade is not None else "N/A"
    confidence_pct = f"{confidence * 100:.1f}%" if confidence else "N/A"

    referable_color = "#ef4444" if is_referable else "#22c55e"
    if is_referable is None:
        referable_color = "#f59e0b"

    quality_color = "#22c55e" if quality_status == "GOOD" else "#f59e0b"
    quality_reasons_html = "".join(f"<li>{r}</li>" for r in quality_reasons) if quality_reasons else "<li>No issues detected</li>"
    findings_html = "".join(f"<li>{f}</li>" for f in key_findings) if key_findings else "<li>None reported</li>"

    recommendation = (
        "Refer to ophthalmologist for confirmation."
        if is_referable
        else "Routine follow-up recommended per clinical guidelines."
    )

    images_section = ""
    if original_image_b64 or enhanced_image_b64:
        images_section = f"""
        <div class="images-row">
            {"<div class='image-box'><h4>Original</h4><img src='data:image/jpeg;base64," + original_image_b64 + "' alt='Original fundus image'></div>" if original_image_b64 else ""}
            {"<div class='image-box'><h4>Enhanced (CLAHE)</h4><img src='data:image/jpeg;base64," + enhanced_image_b64 + "' alt='Enhanced fundus image'></div>" if enhanced_image_b64 else ""}
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DR Screening Report — PROTOTYPE</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'Inter', sans-serif;
    color: #1e293b;
    background: #f8fafc;
    padding: 40px;
    line-height: 1.6;
  }}

  .report {{
    max-width: 800px;
    margin: 0 auto;
    background: white;
    border-radius: 12px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    overflow: hidden;
  }}

  .header {{
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    color: white;
    padding: 32px 40px;
  }}

  .header h1 {{
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 4px;
  }}

  .header .subtitle {{
    font-size: 13px;
    color: #94a3b8;
  }}

  .disclaimer-banner {{
    background: #fef3c7;
    border-left: 4px solid #f59e0b;
    padding: 12px 40px;
    font-size: 12px;
    color: #92400e;
    font-weight: 500;
  }}

  .body {{ padding: 32px 40px; }}

  .meta-row {{
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: #64748b;
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid #e2e8f0;
  }}

  .section {{ margin-bottom: 28px; }}

  .section h3 {{
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #475569;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid #e2e8f0;
  }}

  .result-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }}

  .result-item {{
    padding: 16px;
    background: #f8fafc;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
  }}

  .result-item .label {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #64748b;
    margin-bottom: 4px;
  }}

  .result-item .value {{
    font-size: 20px;
    font-weight: 700;
  }}

  .badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    color: white;
  }}

  ul {{ padding-left: 20px; }}
  li {{ margin-bottom: 4px; font-size: 14px; }}

  .reasoning-text {{
    background: #f1f5f9;
    padding: 16px;
    border-radius: 8px;
    font-size: 14px;
    line-height: 1.7;
    color: #334155;
  }}

  .images-row {{
    display: flex;
    gap: 16px;
    margin-top: 12px;
  }}

  .image-box {{
    flex: 1;
    text-align: center;
  }}

  .image-box h4 {{
    font-size: 12px;
    color: #64748b;
    margin-bottom: 8px;
  }}

  .image-box img {{
    max-width: 100%;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
  }}

  .disclaimers {{
    background: #f8fafc;
    padding: 24px 40px;
    border-top: 1px solid #e2e8f0;
  }}

  .disclaimers h3 {{
    font-size: 12px;
    font-weight: 600;
    color: #64748b;
    margin-bottom: 8px;
  }}

  .disclaimers ul {{
    font-size: 11px;
    color: #94a3b8;
  }}

  .disclaimers li {{ margin-bottom: 4px; }}

  .print-btn {{
    position: fixed;
    bottom: 24px;
    right: 24px;
    padding: 12px 24px;
    background: #0f172a;
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
  }}

  .print-btn:hover {{ background: #1e293b; }}

  @media print {{
    body {{ padding: 0; background: white; }}
    .report {{ box-shadow: none; }}
    .print-btn {{ display: none; }}
  }}
</style>
</head>
<body>
<div class="report">
  <div class="header">
    <h1>DR Screening Report</h1>
    <div class="subtitle">Diabetic Retinopathy Screening — Prototype Analysis</div>
  </div>

  <div class="disclaimer-banner">
    ⚠ PROTOTYPE — NOT FOR CLINICAL USE. This report is generated by an AI model for demonstration purposes only.
  </div>

  <div class="body">
    <div class="meta-row">
      <span>Generated: {timestamp}</span>
      <span>Model: {model_used}</span>
    </div>

    <div class="section">
      <h3>Screening Results</h3>
      <div class="result-grid">
        <div class="result-item">
          <div class="label">Image Quality</div>
          <div class="value">
            <span class="badge" style="background: {quality_color}">{quality_status}</span>
          </div>
        </div>
        <div class="result-item">
          <div class="label">DR Grade</div>
          <div class="value">{grade_display} — {grade_label}</div>
        </div>
        <div class="result-item">
          <div class="label">Referable DR</div>
          <div class="value">
            <span class="badge" style="background: {referable_color}">{referable_label}</span>
          </div>
        </div>
        <div class="result-item">
          <div class="label">Model Confidence</div>
          <div class="value">{confidence_pct}</div>
          <div style="font-size:11px;color:#94a3b8;margin-top:2px">Model-reported, not clinically calibrated</div>
        </div>
      </div>
    </div>

    <div class="section">
      <h3>Image Quality Details</h3>
      <ul>{quality_reasons_html}</ul>
      {f'<p style="margin-top:8px;font-size:13px;color:#64748b"><strong>VLM note:</strong> {image_quality_note}</p>' if image_quality_note else ''}
    </div>

    {images_section}

    <div class="section">
      <h3>Key Findings</h3>
      <ul>{findings_html}</ul>
    </div>

    <div class="section">
      <h3>Model Reasoning</h3>
      <div class="reasoning-text">{reasoning if reasoning else "No reasoning provided."}</div>
    </div>

    <div class="section">
      <h3>Recommendation</h3>
      <p style="font-size:14px;font-weight:500">{recommendation}</p>
      <p style="font-size:12px;color:#94a3b8;margin-top:4px">
        All results should be confirmed by a qualified ophthalmologist regardless of grade.
      </p>
    </div>
  </div>

  <div class="disclaimers">
    <h3>Disclaimers</h3>
    <ul>
      <li>This is a prototype for demonstration purposes only — not a validated diagnostic tool.</li>
      <li>Image quality checks are simple heuristics, not a clinically validated quality classifier.</li>
      <li>Model confidence is self-reported by the model, not a calibrated clinical confidence score.</li>
      <li>The reasoning panel reflects the model's stated observations, not a pixel-level attention map or proven causal explanation.</li>
      <li>Any "Referable DR" result should be confirmed by an ophthalmologist.</li>
    </ul>
  </div>
</div>

<button class="print-btn" onclick="window.print()">🖨 Print / Save PDF</button>
</body>
</html>"""
