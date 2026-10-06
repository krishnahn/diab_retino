# DR Screening — VLM Web Dashboard

> **Prototype** — Uses a quality-aware preprocessing pipeline followed by a
> vision-language model for diabetic retinopathy grading and a referable-DR
> screening decision, delivered through a web dashboard with model-stated
> reasoning and an automated report for ophthalmologist-assisted review.

⚠ **NOT FOR CLINICAL USE** — This is a hackathon/presentation prototype only.

---

## Architecture

```
Browser (frontend/index.html)
      │  POST /api/analyze (multipart image)
      ▼
FastAPI (backend/main.py)
      │
      ├── 1. Quality Check (quality_check.py)
      │       └── Laplacian blur, brightness, FOV heuristics
      │
      ├── 2. Preprocessing (preprocessing.py)
      │       └── Crop black border → Resize 512×512 → CLAHE enhancement
      │
      ├── 3. VLM Classification (vlm_client.py)
      │       └── DeepSeek Flash (vision) → Strict JSON grading
      │
      ├── 4. Referable Decision
      │       └── Grade 0-1 → Non-referable, 2-4 → Referable
      │
      └── 5. Report (report.py)
              └── Self-contained HTML → print-to-PDF
```

## Quick Start

### Prerequisites
- Python 3.11+ 
- A DeepSeek API key ([get one here](https://platform.deepseek.com/))

### Setup (one-time)

```bash
# 1. Virtual environment (already created if you followed setup)
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r backend\requirements.txt

# 4. Create your .env file from the template
copy .env.example .env
# Then edit .env and paste your DEEPSEEK_API_KEY
```

### Run the App

```bash
# Make sure your venv is activated, then:
cd backend
python main.py
```

Open **http://localhost:8000** in your browser.

That's it — the FastAPI server serves both the API and the frontend from a single port.

### Usage

1. **Upload** a retinal fundus image (JPEG or PNG)
2. Click **"Analyze Image"**
3. Wait for the pipeline to complete (quality check → enhancement → VLM analysis)
4. View results: DR grade, referable decision, model reasoning
5. Click **"Download Report"** for a printable HTML report (use browser Print → Save as PDF)

---

## Project Structure

```
Diabetic retinopathy/
├── backend/
│   ├── main.py              # FastAPI app — routes + pipeline orchestration
│   ├── quality_check.py     # Focus / brightness / FOV heuristics (OpenCV)
│   ├── preprocessing.py     # Crop, resize, CLAHE enhancement (OpenCV)
│   ├── vlm_client.py        # DeepSeek Flash vision API integration
│   ├── report.py            # HTML report generator (for print-to-PDF)
│   └── requirements.txt     # Python dependencies
├── frontend/
│   └── index.html           # Single-file SPA (HTML + CSS + JS)
├── .env                     # Your API key (gitignored — create from .env.example)
├── .env.example             # Template for environment variables
├── venv/                    # Python virtual environment
└── README.md                # This file
```

## Key Technologies

| Component | Technology |
|---|---|
| Backend | Python 3 + FastAPI + Uvicorn |
| Vision-Language Model | DeepSeek Flash (via OpenAI-compatible SDK) |
| Image Processing | OpenCV (headless) + NumPy + Pillow |
| Frontend | Vanilla HTML/CSS/JS — dark glassmorphism theme |
| Report Export | In-browser HTML → PDF via `window.print()` |

## Disclaimers

- This is a prototype for demonstration purposes only — not a validated diagnostic tool.
- Image quality checks are simple heuristics, not a clinically validated quality classifier.
- Model confidence is self-reported by the model, not a calibrated clinical confidence score.
- The reasoning panel reflects the model's stated observations, not a pixel-level attention map or proven causal explanation.
- Any "Referable DR" result should be confirmed by an ophthalmologist.

## Future Work (Out of Scope)

- Vessel/microaneurysm/exudate/hemorrhage segmentation models
- Clinical calibration and regulatory validation
- Accuracy benchmarking against validated datasets with clinician review
- Multi-model ensemble approaches
