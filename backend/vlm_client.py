"""
VLM Client — DeepSeek Flash Vision API integration for DR grading.

Sends an enhanced fundus image to DeepSeek's vision-capable model and
parses the structured JSON response for diabetic retinopathy grading.

This is a NON-CLINICAL PROTOTYPE — not a validated diagnostic tool.
"""

import base64
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from project root or current dir
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VLMResult:
    """Structured result from the VLM analysis."""
    dr_grade: Optional[int] = None
    grade_label: str = "Inconclusive"
    confidence: float = 0.0
    key_findings: list[str] = field(default_factory=list)
    approx_regions: list[str] = field(default_factory=list)
    reasoning: str = ""
    image_quality_note: str = ""
    is_referable: Optional[bool] = None  # None when grade is null
    referable_label: str = "Inconclusive"
    model_used: str = ""
    raw_response: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Prompt template (from spec Section 3)
# ---------------------------------------------------------------------------

DR_PROMPT = """You are assisting with a NON-CLINICAL PROTOTYPE for diabetic retinopathy (DR) screening
from a fundus photograph. This is for a student/hackathon demo, not real patient care.

Examine the attached fundus image and assess it using the standard 5-level International
Clinical Diabetic Retinopathy (ICDR) severity scale:
0 = No apparent DR
1 = Mild NPDR (microaneurysms only)
2 = Moderate NPDR
3 = Severe NPDR
4 = Proliferative DR (PDR)

Respond with STRICT JSON only, matching this schema, and nothing else:

{
  "dr_grade": <integer 0-4 or null>,
  "grade_label": "<string>",
  "confidence": <float 0-1>,
  "key_findings": ["<finding 1>", "<finding 2>"],
  "approx_regions": ["<e.g. superior-temporal quadrant>"],
  "reasoning": "<2-4 sentence explanation>",
  "image_quality_note": "<comment on whether the image is gradable>"
}

If the image is not a fundus photograph, or is too poor quality to assess, set dr_grade
to null and explain why in "reasoning"."""


# ---------------------------------------------------------------------------
# Grade labels & referable logic
# ---------------------------------------------------------------------------

GRADE_LABELS = {
    0: "No apparent DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR (PDR)",
}


def _apply_referable_rule(grade: Optional[int]) -> tuple[Optional[bool], str]:
    """Grade 0-1 → Non-referable, 2-4 → Referable, None → Inconclusive."""
    if grade is None:
        return None, "Inconclusive"
    if grade <= 1:
        return False, "Non-Referable"
    return True, "Referable"


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """
    Try to extract a JSON object from the VLM response text.
    Handles cases where the model wraps JSON in markdown code blocks.
    """
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in markdown code blocks
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not extract valid JSON from response: {text[:200]}")


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("VLM_MODEL", "google/gemini-2.5-flash-lite")


def analyze_fundus(enhanced_image_bytes: bytes, mime_type: str = "image/jpeg") -> VLMResult:
    """
    Send an enhanced fundus image to the VLM via OpenRouter and return structured grading.

    Args:
        enhanced_image_bytes: JPEG/PNG bytes of the preprocessed fundus image.
        mime_type: MIME type of the image (default: image/jpeg).

    Returns:
        VLMResult with grading, findings, reasoning, and referable decision.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return VLMResult(
            error="DEEPSEEK_API_KEY environment variable is not set.",
            reasoning="Cannot perform analysis without a valid API key.",
        )

    base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    b64_image = base64.b64encode(enhanced_image_bytes).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": DR_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{b64_image}",
                    },
                },
            ],
        }
    ]

    # -- First attempt --
    result = _call_vlm(client, messages)
    if result is not None:
        return result

    # -- Retry with a nudge for clean JSON --
    messages.append({
        "role": "assistant",
        "content": "(previous response was not valid JSON)",
    })
    messages.append({
        "role": "user",
        "content": "Please respond with ONLY valid JSON matching the schema above. No other text.",
    })

    result = _call_vlm(client, messages)
    if result is not None:
        return result

    # -- Total failure fallback --
    return VLMResult(
        reasoning="The model could not produce a valid structured response after two attempts. "
                  "This may be due to image quality or a transient API issue.",
        error="Failed to parse VLM response after retry.",
        model_used=MODEL_NAME,
    )


def _call_vlm(client: OpenAI, messages: list[dict]) -> Optional[VLMResult]:
    """Make a single VLM API call and attempt to parse the response."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1024,
            temperature=0.2,  # Low temp for more deterministic grading
        )

        raw_text = response.choices[0].message.content or ""

        try:
            data = _extract_json(raw_text)
        except ValueError:
            return None  # Signal caller to retry

        grade = data.get("dr_grade")
        if grade is not None:
            grade = int(grade)
            if grade < 0 or grade > 4:
                grade = None

        is_referable, referable_label = _apply_referable_rule(grade)

        return VLMResult(
            dr_grade=grade,
            grade_label=data.get("grade_label", GRADE_LABELS.get(grade, "Inconclusive")),
            confidence=float(data.get("confidence", 0.0)),
            key_findings=data.get("key_findings", []),
            approx_regions=data.get("approx_regions", []),
            reasoning=data.get("reasoning", ""),
            image_quality_note=data.get("image_quality_note", ""),
            is_referable=is_referable,
            referable_label=referable_label,
            model_used=MODEL_NAME,
            raw_response=raw_text,
        )

    except Exception as e:
        return VLMResult(
            error=f"VLM API call failed: {str(e)}",
            reasoning=f"An error occurred while calling the VLM API: {str(e)}",
            model_used=MODEL_NAME,
        )
