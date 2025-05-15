#!/usr/bin/env python3
"""
extract_copyright_flexible.py
-----------------------------
Locate and read the Street-View string  © YYYY Google  anywhere in a panorama.

Output :  "<year>\t<confidence>"          (confidence ∈ [0, 1])
Exit-1 :  watermark not found.

Dependencies
------------
opencv-python, pytesseract ≥ 0 .3, Tesseract ≥ 4.

Algorithm (concise)
-------------------
1. full-frame LAB-CLAHE → enhances faint glyphs
2. white–tophat (31 × 31) + Otsu → mask of thin bright lines
3. connected components → long/thin blobs are candidate boxes
4. each candidate (with a 25 px margin) is up-scaled ×4 and sent to
   Tesseract (digits only).  
   First 4-digit token ∈ [2008, 2025] wins; its mean digit confidence is
   returned.
"""

from __future__ import annotations
import sys, re
from pathlib import Path
import cv2, pytesseract
import numpy as np
from typing import Tuple, Optional

YEAR_RX  = re.compile(r"^\d{4}$")
VALID    = range(2008, 2026)
TESS_CFG = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"
PAD      = 25                           # px padding round each candidate box


# --------------------------------------------------------------------------- #
def _preproc_full(img: np.ndarray) -> np.ndarray:
    l = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]
    l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
    k  = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    return cv2.morphologyEx(l, cv2.MORPH_TOPHAT, k)


def _boxes(mask: np.ndarray) -> list[Tuple[int, int, int, int]]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 3 * h and 60 <= w <= 800 and 10 <= h <= 120:   # quick heuristics
            out.append((x, y, w, h))
    return out


def _ocr(patch: np.ndarray) -> Tuple[Optional[int], float]:
    data = pytesseract.image_to_data(
        patch, config=TESS_CFG, output_type=pytesseract.Output.DICT
    )
    for txt, conf in zip(data["text"], data["conf"]):
        if conf == "-1" or not YEAR_RX.match(txt):            # not a 4-digit token
            continue
        yr = int(txt)
        if yr in VALID:
            return yr, int(conf) / 100.0                      # scale to [0,1]
    return None, 0.0


def extract_year(path: str | Path) -> Tuple[Optional[int], float]:
    src = cv2.imread(str(path))
    if src is None:
        raise FileNotFoundError(path)

    topo = _preproc_full(src)
    _, m = cv2.threshold(topo, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    m = cv2.dilate(m, None, iterations=2)

    best = (None, 0.0)
    for x, y, w, h in _boxes(m):
        x0 = max(x - PAD, 0)
        y0 = max(y - PAD, 0)
        x1 = min(x + w + PAD, src.shape[1])
        y1 = min(y + h + PAD, src.shape[0])

        patch = cv2.resize(
            src[y0:y1, x0:x1], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC
        )
        yr, conf = _ocr(patch)
        if yr is not None and conf > best[1]:
            best = (yr, conf)

    return best
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage:  python extract_copyright_flexible.py  <image>")

    year, conf = extract_year(sys.argv[1])
    if year is None:
        sys.exit("Watermark year not detected")

    print(f"{year}\t{conf:.2f}")

