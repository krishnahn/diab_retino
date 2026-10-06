"""
Image Quality Check — deterministic heuristics for fundus image quality.

Checks focus (blur), brightness, and field-of-view (retinal circle coverage).
These are simple heuristics for a prototype demo, NOT a validated quality classifier.
"""

import cv2
import numpy as np


def check_quality(image_bytes: bytes) -> dict:
    """
    Run quality heuristics on a raw fundus image.

    Args:
        image_bytes: Raw image file bytes.

    Returns:
        dict with:
            - status: "GOOD" or "POOR"
            - reasons: list of strings explaining any issues
            - scores: dict of individual metric values (for debugging / display)
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {
            "status": "POOR",
            "reasons": ["Could not decode the uploaded file as an image."],
            "scores": {},
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    reasons = []
    scores = {}

    # ---- 1. Focus / Blur Detection ----
    # Variance of Laplacian — low value means blurry
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    scores["focus_score"] = round(float(laplacian_var), 2)

    if laplacian_var < 100:
        reasons.append(
            f"Image appears blurry (focus score: {laplacian_var:.0f}, threshold: 100)."
        )

    # ---- 2. Brightness ----
    mean_brightness = float(gray.mean())
    scores["mean_brightness"] = round(mean_brightness, 2)

    if mean_brightness < 40:
        reasons.append(
            f"Image is underexposed/too dark (brightness: {mean_brightness:.0f}/255)."
        )
    elif mean_brightness > 220:
        reasons.append(
            f"Image is overexposed/has glare (brightness: {mean_brightness:.0f}/255)."
        )

    # ---- 3. Field of View (retinal circle coverage) ----
    # Threshold to find the bright retinal region against the dark border
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = img.shape[0] * img.shape[1]

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        retina_area = cv2.contourArea(largest_contour)
        coverage = retina_area / total_area
    else:
        coverage = 0.0

    scores["retina_coverage"] = round(float(coverage), 3)

    if coverage < 0.15:
        reasons.append(
            f"Retinal region is very small or not detected "
            f"(coverage: {coverage:.0%} of frame). "
            f"This may not be a fundus photograph."
        )

    # ---- Final verdict ----
    status = "POOR" if reasons else "GOOD"

    return {
        "status": status,
        "reasons": reasons,
        "scores": scores,
    }
