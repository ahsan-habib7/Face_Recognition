# ============================================================
# bangla_ocr.py  — Bangla Field Extractor for Bangladesh NID
# ============================================================
#
# Pipeline (3 strategies, best-candidate voting):
#
#   Strategy 1 — EAST Deep Learning Text Detection
#     Uses frozen_east_text_detection.pb to precisely locate
#     text regions on the NID, then runs Tesseract (ben+eng)
#     on each cropped region. Inspired by the reference notebook
#     NID-Information-Extraction-using-Deep-Learning.
#
#   Strategy 2 — Multi-Pass Full-Card Tesseract OCR
#     9 passes (3 preprocessing variants × 3 PSM modes).
#     Falls back gracefully when EAST model is unavailable.
#
#   Strategy 3 — Spatial Label-Anchor (bounding-box search)
#     pytesseract.image_to_data locates label words by pixel
#     position, then collects the value text beside / below.
#
# Public API:
#   extract_bangla_fields(image_bgr) -> dict
#       keys: name_bn, father_bn, mother_bn
#
# All logic is isolated — never touches PaddleOCR, ArcFace,
# or the validation pipeline.  Wrapped in try/except throughout
# so a failure here can never crash the main service.
# ============================================================

import os
import re
import time
import cv2
import numpy as np

try:
    import pytesseract
    from PIL import Image as _PILImage
    _TESS_AVAILABLE = True
except ImportError:
    _TESS_AVAILABLE = False

try:
    from imutils.object_detection import non_max_suppression as _NMS
    _IMUTILS_AVAILABLE = True
except ImportError:
    _IMUTILS_AVAILABLE = False

_EAST_MODEL_PATH = os.path.join(os.path.dirname(__file__), "frozen_east_text_detection.pb")
_EAST_NET = None  

def _get_east_net():
    """Load the EAST model once and cache it for all subsequent calls."""
    global _EAST_NET
    if _EAST_NET is None:
        if not os.path.isfile(_EAST_MODEL_PATH):
            return None
        print("[EAST] loading model into memory (one-time)...", flush=True)
        _EAST_NET = cv2.dnn.readNet(_EAST_MODEL_PATH)
        print("[EAST] model ready.", flush=True)
    return _EAST_NET

_BN_RE  = re.compile(r'[\u0980-\u09FF]')
_JNK_RE = re.compile(r'[^\u0980-\u09FF\u200C\u200D\s\.\-\(\)]')

def _bn_count(s: str) -> int:
    return len(_BN_RE.findall(s))

def _is_garbage(s: str) -> bool:
    if not s:
        return True
    junk = len(_JNK_RE.findall(s))
    return junk > len(s) * 0.35 or _bn_count(s) < 2

def _clean_value(s: str) -> str:
    s = re.sub(r'[\u200C\u200D]', '', s)         
    s = re.sub(r'[^\u0980-\u09FF\s\.\-\(\)]', ' ', s)  
    s = re.sub(r'\s{2,}', ' ', s).strip()
    s = re.sub(r'[\s\.\-]+$', '', s).strip()
    return s

_BN_CORRECTIONS = [
    (re.compile('\u0986\u09A8\u09C0\u09B9\u09B8\u09BE\u09A8'), '\u0986\u09B9\u09B8\u09BE\u09A8'),
    (re.compile('\u0986\u09A8\u09BF\u09B9\u09B8\u09BE\u09A8'), '\u0986\u09B9\u09B8\u09BE\u09A8'),
    (re.compile('\u0983\u0983+'), '\u0983'),
    (re.compile(r'[a-zA-Z0-9]+'), ''),
]

def _apply_corrections(s: str) -> str:
    """Apply conjunct corrections and final noise stripping to a best candidate."""
    for pat, repl in _BN_CORRECTIONS:
        s = pat.sub(repl, s)
    s = re.sub(r'[\s\.\-]+$', '', s).strip() 
    s = re.sub(r'^[\s\.\-]+', '', s).strip()   
    s = re.sub(r'\s{2,}', ' ', s)              
    return s

_LABELS = {
    "name_bn": [
        "নাম", "নামঃ", "নাম:", "নাম :", "নাম ঃ",
    ],
    "father_bn": [
        "পিতা", "পিতাঃ", "পিতা:", "পিতার নাম", "পিতার নামঃ", "পিতার নাম:",
        "বাবার নাম", "বাবার নামঃ",
    ],
    "mother_bn": [
        "মাতা", "মাতাঃ", "মাতা:", "মাতার নাম", "মাতার নামঃ", "মাতার নাম:",
        "মায়ের নাম", "মায়ের নামঃ",
    ],
}

_LABEL_RE: dict = {}
for _field, _forms in _LABELS.items():
    _sorted = sorted(_forms, key=len, reverse=True)
    _pat = '|'.join(re.escape(f) for f in _sorted)
    _LABEL_RE[_field] = re.compile(
        r'^(?:' + _pat + r')\s*[ঃ:\s]?\s*(.*)', re.UNICODE
    )

_ALL_LABEL_TEXTS = []
for _field, _forms in _LABELS.items():
    for _f in _forms:
        _ALL_LABEL_TEXTS.append((_f, _field))
_ALL_LABEL_TEXTS.sort(key=lambda x: len(x[0]), reverse=True)

def _upscale(image_bgr: np.ndarray, min_width: int = 2400) -> np.ndarray:
    """Upscale image so width >= min_width. Tesseract needs high-DPI for Bangla conjuncts."""
    h, w = image_bgr.shape[:2]
    if w < min_width:
        scale = min_width / w
        image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LANCZOS4)
    return image_bgr


def _preproc_variants(image_bgr: np.ndarray):
    image_bgr = _upscale(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variants = []

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v1 = clahe.apply(gray)
    _, v1 = cv2.threshold(v1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("clahe_otsu", _PILImage.fromarray(v1)))

    v2 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    variants.append(("adaptive", _PILImage.fromarray(v2)))

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    v3g = cv2.filter2D(gray, -1, kernel)
    v3 = cv2.adaptiveThreshold(
        v3g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 8
    )
    variants.append(("sharpen_adaptive", _PILImage.fromarray(v3)))

    v4 = cv2.bilateralFilter(gray, 9, 75, 75)
    _, v4 = cv2.threshold(v4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("bilateral_otsu", _PILImage.fromarray(v4)))

    return variants


def _merge_row_boxes(boxes: list, origH: int, origW: int) -> list:
    if not boxes:
        return []

    # Sort top-to-bottom
    boxes = sorted(boxes, key=lambda b: b[1])

    heights = [b[3] - b[1] for b in boxes]
    median_h = sorted(heights)[len(heights) // 2]
    row_gap = max(int(median_h * 0.4), 8)

    groups = []
    current = [boxes[0]]
    for box in boxes[1:]:
        prev_cy = (current[-1][1] + current[-1][3]) / 2
        this_cy = (box[1] + box[3]) / 2
        if abs(this_cy - prev_cy) <= row_gap:
            current.append(box)
        else:
            groups.append(current)
            current = [box]
    groups.append(current)

    merged = []
    for grp in groups:
        sX = max(0,     min(b[0] for b in grp) - 30) 
        sY = max(0,     min(b[1] for b in grp) - 8)
        eX = min(origW, max(b[2] for b in grp) + 30)  
        eY = min(origH, max(b[3] for b in grp) + 12)
        merged.append((sX, sY, eX, eY))

    return merged


def _east_detect_regions(image_bgr: np.ndarray,
                         min_confidence: float = 0.4) -> list:

    orig = image_bgr.copy()
    (origH, origW) = orig.shape[:2]

    newW, newH = 320, 320
    rW = origW / float(newW)
    rH = origH / float(newH)

    resized = cv2.resize(image_bgr, (newW, newH))

    layer_names = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3",
    ]

    net = _get_east_net()
    if net is None:
        return []
    blob = cv2.dnn.blobFromImage(
        resized, 1.0, (newW, newH),
        (123.68, 116.78, 103.94), swapRB=True, crop=False
    )
    t0 = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layer_names)
    print(f"[EAST] detection took {time.time()-t0:.3f}s", flush=True)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xD0 = geometry[0, 0, y]
        xD1 = geometry[0, 1, y]
        xD2 = geometry[0, 2, y]
        xD3 = geometry[0, 3, y]
        angles = geometry[0, 4, y]

        for x in range(numCols):
            if scoresData[x] < min_confidence:
                continue
            offsetX, offsetY = x * 4.0, y * 4.0
            angle = angles[x]
            cos, sin = np.cos(angle), np.sin(angle)
            h = xD0[x] + xD2[x]
            w = xD1[x] + xD3[x]
            endX   = int(offsetX + cos * xD1[x] + sin * xD2[x])
            endY   = int(offsetY - sin * xD1[x] + cos * xD2[x])
            startX = int(endX - w)
            startY = int(endY - h)
            rects.append((startX, startY, endX, endY))
            confidences.append(float(scoresData[x]))

    if not rects:
        return []

    if _IMUTILS_AVAILABLE:
        boxes = _NMS(np.array(rects), probs=confidences)
    else:
        boxes = rects

    scaled = []
    for (sX, sY, eX, eY) in boxes:
        scaled.append((
            int(sX * rW),
            int(sY * rH),
            int(eX * rW),
            int(eY * rH),
        ))

    result = _merge_row_boxes(scaled, origH, origW)

    # Sort top-to-bottom for reading order
    result.sort(key=lambda b: b[1])
    return result


def _s1_east(image_bgr: np.ndarray) -> list[str]:
    if _get_east_net() is None:
        print(f"[EAST] model not found at {_EAST_MODEL_PATH} — skipping S1", flush=True)
        return []

    texts = []
    try:
        regions = _east_detect_regions(image_bgr)
        print(f"[EAST] detected {len(regions)} text regions", flush=True)

        for (x1, y1, x2, y2) in regions:
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # Upscale small crops
            ch, cw = crop.shape[:2]
            if cw < 200:
                crop = cv2.resize(crop, None, fx=3.0, fy=3.0,
                                  interpolation=cv2.INTER_LANCZOS4)
            elif cw < 400:
                crop = cv2.resize(crop, None, fx=2.0, fy=2.0,
                                  interpolation=cv2.INTER_LANCZOS4)

            # Preprocess crop for Tesseract
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            gray = clahe.apply(gray)
            _, gray = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            pil_crop = _PILImage.fromarray(gray)

            for cfg in ["--psm 7 --oem 1", "--psm 6 --oem 1", "--psm 8 --oem 1"]:
                try:
                    txt = pytesseract.image_to_string(
                        pil_crop, lang="ben+eng", config=cfg
                    )
                    if txt.strip():
                        print(f"[EAST][ROW] cfg={cfg} → {txt.strip()!r}", flush=True)
                        texts.append(txt)
                        break
                except Exception:
                    pass

    except Exception as e:
        print(f"[EAST][S1] error: {e}", flush=True)

    return texts

def _s2_multipass(image_bgr: np.ndarray) -> list[str]:
    PSM_MODES = ["--psm 6 --oem 1", "--psm 4 --oem 1", "--psm 3 --oem 1", "--psm 8 --oem 1", "--psm 11 --oem 1"]
    passes = []

    for _name, pil in _preproc_variants(image_bgr):
        for cfg in PSM_MODES:
            try:
                txt = pytesseract.image_to_string(pil, lang="ben+eng", config=cfg)
                passes.append(txt)
            except Exception:
                pass

    h = image_bgr.shape[0]
    crop_top = int(h * 0.30)
    bottom_half = image_bgr[crop_top:, :]
    for _name, pil in _preproc_variants(bottom_half):
        for cfg in PSM_MODES:
            try:
                txt = pytesseract.image_to_string(pil, lang="ben+eng", config=cfg)
                passes.append(txt)
            except Exception:
                pass

    return passes


def _s3_spatial(image_bgr: np.ndarray) -> dict:
    result = {"name_bn": "", "father_bn": "", "mother_bn": ""}
    try:
        img = _upscale(image_bgr)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pil = _PILImage.fromarray(gray)

        data = None
        for cfg in ["--psm 6 --oem 1", "--psm 4 --oem 1", "--psm 3 --oem 1"]:
            try:
                data = pytesseract.image_to_data(
                    pil, lang="ben+eng",
                    config=cfg,
                    output_type=pytesseract.Output.DICT,
                )
                if data and data['text']:
                    break
            except Exception:
                pass
        if not data:
            return result

        words   = data['text']
        lefts   = data['left']
        widths  = data['width']
        tops    = data['top']
        confs   = data['conf']
        lines   = data['line_num']
        blocks  = data['block_num']

        line_groups: dict = {}
        for i, txt in enumerate(words):
            txt = txt.strip()
            if not txt or int(confs[i]) < 10:
                continue
            key = (blocks[i], lines[i])
            line_groups.setdefault(key, []).append(
                (txt, lefts[i], tops[i], widths[i])
            )

        line_list = []
        for key in sorted(line_groups.keys()):
            wds = line_groups[key]
            full  = ' '.join(w[0] for w in wds)
            min_l = min(w[1] for w in wds)
            avg_t = sum(w[2] for w in wds) // len(wds)
            max_r = max(w[1] + w[3] for w in wds)
            line_list.append((full, min_l, avg_t, max_r))

        for idx, (line_text, _l, _t, _r) in enumerate(line_list):
            for label_text, field in _ALL_LABEL_TEXTS:
                if result[field]:
                    continue
                if line_text.startswith(label_text):
                    after = line_text[len(label_text):].lstrip('ঃ: \t')
                    if after and _bn_count(after) >= 2:
                        result[field] = _clean_value(after)
                    elif idx + 1 < len(line_list):
                        nxt = line_list[idx + 1][0]
                        if _bn_count(nxt) >= 2:
                            result[field] = _clean_value(nxt)
                    break

    except Exception as exc:
        print(f"[BANGLA_OCR][S3] spatial error: {exc}", flush=True)

    return result


def _parse_passes(passes: list[str]) -> dict:
    candidates: dict = {"name_bn": [], "father_bn": [], "mother_bn": []}

    for raw in passes:
        if not raw:
            continue
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        for i, line in enumerate(lines):
            for field, pat in _LABEL_RE.items():
                m = pat.match(line)
                if not m:
                    continue
                val = m.group(1).strip()
                if not val:
                    for j in range(i + 1, min(i + 4, len(lines))):
                        nxt = lines[j].strip()
                        if nxt and _bn_count(nxt) >= 2:
                            val = nxt
                            break
                val = _clean_value(val)
                if val and not _is_garbage(val):
                    score = _bn_count(val) * 10 + len(val)
                    candidates[field].append((score, val))
                break

    return candidates


def _best_candidate(pool: list) -> str:
    if not pool:
        return ""
    return max(pool, key=lambda x: x[0])[1]

def extract_bangla_fields(image_bgr: np.ndarray) -> dict:
    """
    Pipeline:
      S1 — EAST deep learning text detector → per-region Tesseract OCR
      S2 — Multi-pass full-card Tesseract OCR (9 passes + bottom-half crop)
      S3 — Spatial label-anchor (bounding-box search)
    """
    result = {"name_bn": "", "father_bn": "", "mother_bn": ""}

    if not _TESS_AVAILABLE:
        print("[BANGLA_OCR] pytesseract not available — returning empty fields", flush=True)
        return result

    try:
        all_candidates: dict = {"name_bn": [], "father_bn": [], "mother_bn": []}

        # S1: EAST text detection
        s1_texts = _s1_east(image_bgr)
        print(f"[BANGLA_OCR][S1] EAST → {len(s1_texts)} region texts", flush=True)
        cands1 = _parse_passes(s1_texts)
        for f in all_candidates:
            all_candidates[f].extend(cands1[f])

        # S2: Multi-pass full-card
        s2_texts = _s2_multipass(image_bgr)
        print(f"[BANGLA_OCR][S2] multipass → {len(s2_texts)} passes", flush=True)
        cands2 = _parse_passes(s2_texts)
        for f in all_candidates:
            all_candidates[f].extend(cands2[f])

        # S3: Spatial label-anchor
        s3 = _s3_spatial(image_bgr)
        for f in all_candidates:
            if s3.get(f) and not _is_garbage(s3[f]):
                score = _bn_count(s3[f]) * 10 + len(s3[f]) + 20  
                all_candidates[f].append((score, s3[f]))

        # Vote: best candidate per field + apply corrections
        for f in result:
            val = _best_candidate(all_candidates[f])
            result[f] = _apply_corrections(val) if val else ""

        print(
            f"[BANGLA_OCR] FINAL  "
            f"name_bn={result['name_bn']!r}  "
            f"father={result['father_bn']!r}  "
            f"mother={result['mother_bn']!r}",
            flush=True,
        )

    except Exception as exc:
        import traceback
        print(f"[BANGLA_OCR] ERROR (non-fatal): {exc}", flush=True)
        traceback.print_exc()

    return result


def enrich_bangla_fields(nid_data: dict, image_bgr: np.ndarray) -> dict:
    needs = (
        not nid_data.get("name_bn") or
        not nid_data.get("father_bn") or
        not nid_data.get("mother_bn")
    )
    if not needs:
        return nid_data

    bn = extract_bangla_fields(image_bgr)
    for field in ("name_bn", "father_bn", "mother_bn"):
        if not nid_data.get(field) and bn.get(field):
            nid_data[field] = bn[field]
            print(f"[ENRICH] {field} filled by EAST+Tesseract: {bn[field]!r}", flush=True)

    return nid_data