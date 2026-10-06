"""
Image Preprocessing — crop, resize, and CLAHE enhancement for fundus images.

Produces a contrast-enhanced version of the fundus image to make
vessels, lesions, and other features more visible for the VLM and for display.

Deterministic — no AI model involved.
"""

import cv2
import numpy as np


def preprocess(image_bytes: bytes, target_size: int = 512) -> tuple[bytes, dict]:
    """
    Preprocess a raw fundus image: crop black border, resize, apply CLAHE.

    Args:
        image_bytes: Raw image file bytes.
        target_size: Target dimension for the square output (default 512×512).

    Returns:
        Tuple of (enhanced_jpeg_bytes, metadata_dict).
        metadata_dict contains crop/resize info for display.
    """
    # Decode
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image bytes.")

    original_h, original_w = img.shape[:2]
    metadata = {
        "original_size": f"{original_w}×{original_h}",
    }

    # ---- 1. Crop black border around retinal circle ----
    img = _crop_retina(img)
    crop_h, crop_w = img.shape[:2]
    metadata["cropped_size"] = f"{crop_w}×{crop_h}"

    # ---- 2. Resize to target_size × target_size ----
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    metadata["final_size"] = f"{target_size}×{target_size}"

    # ---- 3. CLAHE on LAB L-channel ----
    img = _apply_clahe(img)
    metadata["enhancement"] = "CLAHE (clipLimit=2.0, grid=8×8) on LAB L-channel"

    # Encode back to JPEG
    success, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise ValueError("Failed to encode enhanced image to JPEG.")

    return encoded.tobytes(), metadata


def _crop_retina(img: np.ndarray) -> np.ndarray:
    """
    Detect the retinal disc region and crop to its bounding rectangle.
    Falls back to the full image if no clear retinal region is found.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to separate retina (bright-ish) from black border
    _, binary = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img  # No contour found — return original

    largest = max(contours, key=cv2.contourArea)
    total_area = img.shape[0] * img.shape[1]

    # Only crop if the retinal region is a meaningful portion of the image
    if cv2.contourArea(largest) < 0.05 * total_area:
        return img  # Too small — probably not a real retinal boundary

    x, y, w, h = cv2.boundingRect(largest)

    # Add a small margin (5% of each dimension)
    margin_x = int(w * 0.05)
    margin_y = int(h * 0.05)
    x = max(0, x - margin_x)
    y = max(0, y - margin_y)
    w = min(img.shape[1] - x, w + 2 * margin_x)
    h = min(img.shape[0] - y, h + 2 * margin_y)

    return img[y : y + h, x : x + w]


def _apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    on the L-channel of the LAB color space.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return enhanced
