# ============================================================
# Face Recognition & ID Validation API 
# FastAPI + ArcFace (InsightFace buffalo_l) + PaddleOCR
# Bangladesh Bank KYC — Fully Offline
# ============================================================
#
# ArcFace Pipeline (Immich-style):
#   Step 1 — Face Detection   : RetinaFace (det_10g.onnx inside buffalo_l)
#   Step 2 — Face Alignment   : 5-pt landmark alignment
#   Step 3 — Embedding        : ArcFace ResNet-50 → 512-dim vector
#   Step 4 — Matching         : Cosine similarity of L2-normed embeddings
#
# Endpoints:
#   POST /validate-id             — NID card OCR validation (PaddleOCR)
#   POST /verify-face             — Direct ArcFace face comparison
#   POST /verify-with-liveness    — Full KYC: liveness session + ArcFace
#   POST /liveness/start          — Create liveness session
#   POST /liveness/frame          — Process video frame
#   POST /liveness/cancel         — Cancel session
#   GET  /health                  — Health check
# ============================================================

import cv2
import base64
import re
import asyncio
import numpy as np
from insightface.app import FaceAnalysis
from paddleocr import PaddleOCR
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from liveness.router import router as liveness_router
from liveness.session import get_session, delete_session, cleanup_old_sessions

# Bangla OCR — EAST Deep Learning + Tesseract Bengali pipeline
# Uses bangla_ocr.py (EAST text detector + multi-pass Tesseract + spatial anchor)
# to extract নাম / পিতা / মাতা fields.  PaddleOCR handles all English fields.
try:
    from bangla_ocr import enrich_bangla_fields
    print("✅ bangla_ocr (EAST + Tesseract Bengali) loaded successfully.")
except Exception as _bangla_import_err:
    print(f"⚠️  bangla_ocr import failed ({_bangla_import_err}) — Bangla fields may be empty.")
    def enrich_bangla_fields(nid_data, image_bgr):   # type: ignore[misc]
        return nid_data

async def _session_cleanup_task():
    """Runs forever, purging sessions idle for >10 minutes every 5 minutes."""
    while True:
        await asyncio.sleep(300) 
        try:
            removed = cleanup_old_sessions()
            if removed:
                print(f"[SESSION_CLEANUP] removed {removed} stale session(s)", flush=True)
        except Exception as exc:
            print(f"[SESSION_CLEANUP] error (non-fatal): {exc}", flush=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_session_cleanup_task())
    print("[STARTUP] Session cleanup task started (5-min interval).", flush=True)
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title="SecureBank Face Verification API",
    description=(
        "Bangladesh bank KYC: NID OCR validation (PaddleOCR) + "
        "ArcFace face verification (InsightFace buffalo_l) — fully offline."
    ),
    version="6.4.0", 
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5000",
        "http://localhost:7000",
        "http://localhost:5172",
        "http://localhost:3000",
        "http://frontend:5000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(liveness_router, prefix="/liveness")

print("Loading InsightFace ArcFace pipeline (buffalo_l)...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("ArcFace pipeline ready.")

paddle_ocr = None

def get_paddle_ocr():
    global paddle_ocr
    if paddle_ocr is None:
        print("🔄 Lazy loading PaddleOCR (latin + Bengali support)...")
        paddle_ocr = PaddleOCR(
            use_angle_cls=True,
            lang='latin', 
            show_log=False,
            det_db_thresh=0.3,
            det_db_box_thresh=0.6,
            det_db_unclip_ratio=1.8,
            rec_batch_num=6,
            use_gpu=False
        )
        print("✅ PaddleOCR ready (lazy loaded).")
    return paddle_ocr

# KEYWORD PATTERNS — Bangladesh ID documents
NID_KEYWORDS = [
    "জাতীয় পরিচয়পত্র", "জাতীয়পরিচয়পত্র",
    "নির্বাচন কমিশন", "নির্বাচনকমিশন",
    "জন্ম তারিখ", "জন্মতারিখ", "জন্মতারিখ:",

    "national id card", "national id", "nid", "nid no", "nid no.",
    "election commission", "election commission bangladesh",
    "voter no", "voter id", "voter no.",
    "smart national id", "smart nid",
    "date of birth", "id no", "id no.", "pin",
    "national identity", "smart card", "identity card", "id card",

    "peoples republic", "people's republic",
    "republic of bangladesh",
    "govt. of bangladesh", "govt of bangladesh",
    "government of bangladesh",
    "bangladesh national",

    "blood group", "blood grp",
    "issued date", "issue date",
    "father", "mother",
]

def decode_image(file_bytes: bytes, label: str) -> np.ndarray:
    np_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode {label}. Please upload a valid JPEG or PNG."
        )
    return image

def image_to_base64(image_bgr: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok or buf is None:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def preprocess_for_paddleocr(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if w < 1600:
        scale = 1600 / w
        image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(image_bgr, -1, kernel)
    image_bgr = cv2.addWeighted(image_bgr, 0.7, sharpened, 0.3, 0)
    return image_bgr


def extract_paddle_text(ocr_result) -> str:
    if not ocr_result or not ocr_result[0]:
        return ""
    lines = []
    for line_boxes in ocr_result:
        for box_info in line_boxes:
            if len(box_info) >= 2:
                text = box_info[1][0].strip()
                confidence = box_info[1][1]
                if confidence > 0.5 and len(text) > 1:
                    lines.append(text)
    return " ".join(lines)


def extract_text_paddle(image_bgr: np.ndarray) -> list:
    ocr = get_paddle_ocr()  # 🔥 MAGIC: Loads on first request
    results = []
    
    # Pass 1: Standard OCR
    processed = preprocess_for_paddleocr(image_bgr)
    ocr_result = ocr.ocr(processed, cls=True)
    text0 = extract_paddle_text(ocr_result)
    results.append(text0)
    
    # Pass 2: High-res
    h, w = image_bgr.shape[:2]
    if w < 2000:
        scale = 2000 / w
        high_res = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    else:
        high_res = image_bgr
    ocr_result = ocr.ocr(high_res, cls=True)
    text1 = extract_paddle_text(ocr_result)
    results.append(text1)
    
    # Pass 3: ROI detection (robust version)
    roi_result = ocr.ocr(image_bgr, det=True, rec=False, cls=True)
    if roi_result and roi_result[0]:
        roi_texts = []
        for box_info in roi_result[0]:   
            bbox = box_info[0] if isinstance(box_info[0], list) else box_info
            if isinstance(bbox, list) and len(bbox) >= 4:
                # Use top-left and bottom-right corners
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x1, y1 = int(min(x_coords)), int(min(y_coords))
                x2, y2 = int(max(x_coords)), int(max(y_coords))
                roi_crop = image_bgr[y1:y2+10, x1:x2+10] 
                if roi_crop.size > 0:
                    roi_ocr = ocr.ocr(roi_crop, cls=True)
                    roi_text = extract_paddle_text(roi_ocr)
                    if roi_text:
                        roi_texts.append(roi_text)
        results.append(" ".join(roi_texts))
    else:
        results.append("")

    return results 


def extract_text(image_bgr: np.ndarray) -> str:
    passes = extract_text_paddle(image_bgr)
    if passes is None:
        passes = []
    merged = "\n".join(p for p in passes if p and p.strip())
    print(f"[OCR] extracted {len(merged)} chars across {len(passes)} passes", flush=True)
    print(f"[OCR] raw text:\n{merged[:800]}", flush=True)
    return merged

def detect_id_type(text: str) -> dict:
    text_lower = text.lower()

    if any(kw.lower() in text_lower for kw in NID_KEYWORDS):
        return {
            "valid": True,
            "id_type": "Bangladesh NID",
            "message": "Valid Bangladesh National ID Card detected"
        }

    import re as _re
    nid_nums = _re.findall(r'\b(\d{10}|\d{13}|\d{17})\b', text)
    if nid_nums:
        return {
            "valid": True,
            "id_type": "Bangladesh NID",
            "message": "Valid Bangladesh National ID Card detected (by ID number)"
        }

    return {
        "valid": False,
        "id_type": "unknown",
        "message": (
            "No valid Bangladesh NID detected. "
            "Please upload a clear photo of your NID, Passport, or Driving License."
        )
    }


def extract_nid_fields(ocr_text: str) -> dict:
    return extract_nid_fields_from_passes([ocr_text])


def extract_nid_fields_from_passes(passes: list) -> dict:
    VISARGA = '\u0983'
    FIELDS  = ["name_en", "name_bn", "father_bn", "mother_bn", "nid_number", "dob"]
    candidates: dict = {f: [] for f in FIELDS}

    HEADER_WORDS = {
        "গণপ্রজাতন্ত্রী", "বাংলাদেশ", "সরকার", "government", "peoples",
        "republic", "national", "election", "commission", "জাতীয়", "পরিচয়",
        "নির্বাচন", "কমিশন", "voter", "identity", "card", "people's",
    }

    def is_header(val: str) -> bool:
        return any(w in HEADER_WORDS for w in val.lower().split())

    def value_after_stem(line: str, stem_len: int) -> str:
        if stem_len >= len(line):
            return ""
        rest = line[stem_len:]
        if rest and rest[0] in (VISARGA, ':', ' ', '।'):
            return rest[1:].strip()
        return rest.strip()

    def next_nonempty(lines: list, i: int, limit: int = 3) -> str:
        for j in range(i + 1, min(i + 1 + limit, len(lines))):
            v = lines[j].strip()
            if v:
                return v
        return ""

    def clean_en_name(val: str) -> str:
        m = re.search(r'[A-Za-z][A-Za-z\s\.\-]*', val)
        if m:
            cand = m.group(0).strip()
            if len(cand) >= 3:
                return cand
        return ""

    def score_value(val: str, field: str) -> int:
        if not val or len(val) < 2:
            return 0 
        if is_header(val):
            return 0

        junk = sum(1 for c in val
                   if not re.match(r'[a-zA-Z0-9 .,\'\-\u0980-\u09FF\u200c\u200d]', c))
        if junk > len(val) * 0.3:
            return 0

        zwj = val.count('\u200c') + val.count('\u200d')
        score = len(val) - zwj * 10   # heavy penalty per ZWJ

        if field in ("name_en", "name_bn", "father_bn", "mother_bn"):
            reject_kw = ["nid", "id no", "date of birth", "voter", "তারিখ",
                         "birth", "card", "/", "\\", "{", "}", "[", "]"]
            if any(kw in val.lower() for kw in reject_kw):
                return 0

            if field in ("name_bn", "father_bn", "mother_bn"):
                bc = len(re.findall(r'[\u0980-\u09FF]', val))
                if bc == 0:
                    return 0
                score += bc * 2

            if field == "name_en":
                lc = len(re.findall(r'[A-Za-z]', val))
                if lc < 2:
                    return 0
                if re.search(r'[\u0980-\u09FF]', val):
                    return 0 
                score += lc

        elif field == "dob":
            if not re.search(r'\d', val):
                return 0
            digits_only = re.sub(r'\D', '', val)
            if len(digits_only) >= 10 and val.strip() == digits_only:
                return 0  

        elif field == "nid_number":
            d = re.sub(r'[\s\-\.]', '', val)
            d = re.sub(r'\D', '', d)
            if len(d) < 10:
                return 0
            score = len(d)

        return max(score, 0)

    LABELS = [
        ("নাম",             "bn",   "name_bn",    3),
        ("name",            "en",   "name_en",    4),
        ("পিতা",            "bn",   "father_bn",  4),
        ("পিতার নাম",       "bn",   "father_bn",  8),
        ("father",          "en",   "father_bn",  6),
        ("মাতা",            "bn",   "mother_bn",  4),
        ("মাতার নাম",       "bn",   "mother_bn",  8),
        ("mother",          "en",   "mother_bn",  6),
        ("id no",           "en",   "nid_number", 5),
        ("nid no",          "en",   "nid_number", 6),
        ("nid",             "en",   "nid_number", 3),
        ("voter no",        "en",   "nid_number", 8),
        ("voter id",        "en",   "nid_number", 8),
        ("pin",             "en",   "nid_number", 3),
        ("date of birth",   "both", "dob",        13),
        ("dob",             "both", "dob",        3),
        ("d.o.b",           "both", "dob",        5),
        ("জন্ম তারিখ",      "both", "dob",        9),
        ("জন্মতারিখ",       "both", "dob",        8),
        ("birth date",      "both", "dob",        10),
    ]

    FUZZY_LABELS = [
        (re.compile(r'^(?:fret|ffet|fffq|pita|fata|freta)\s*[:\u0983]', re.I),
         "father_bn", 4),
        (re.compile(r'^(?:mata|mets)\s*[:\u0983]', re.I),
         "mother_bn", 4),
        (re.compile(r'^[^a-zA-Z\u0980-\u09FF]{0,4}(?:name|nane|naoe)\s*:', re.I),
         "name_en", None), 
    ]

    def stem_matches(line: str, stem: str, sep_type: str) -> bool:
        ll = line.lower()
        sl = stem.lower()
        if not (line.startswith(stem) or ll.startswith(sl)):
            return False
        end = len(stem)
        if end >= len(line):
            return True
        nxt = line[end]
        has_vis = (nxt == VISARGA)
        has_col = (nxt == ':')
        has_sp  = (nxt == ' ')
        if sep_type == "bn":   return has_vis or has_col or has_sp
        if sep_type == "en":   return has_col or has_sp
        return has_vis or has_col or has_sp

    def parse_one_pass(text: str):
        lines = [re.sub(r'[ \t]{2,}', ' ', l.strip())
                 for l in text.splitlines() if l.strip()]

        for i, line in enumerate(lines):
            matched_any = False

            for pattern, field, stem_len in FUZZY_LABELS:
                if not pattern.match(line):
                    continue
                if stem_len is not None:
                    val = value_after_stem(line, stem_len)
                else:
                    # Fuzzy English name — find ':' and take everything after
                    ci = line.find(':')
                    val = line[ci+1:].strip() if ci != -1 else ""
                    val = clean_en_name(val)

                if not val:
                    val = next_nonempty(lines, i)
                sc = score_value(val, field)
                if sc > 0:
                    candidates[field].append((sc, val))
                matched_any = True
                break

            if matched_any:
                continue

            for stem, sep_type, field, stem_len in LABELS:
                if not stem_matches(line, stem, sep_type):
                    continue

                val = value_after_stem(line, stem_len)
                if not val:
                    val = next_nonempty(lines, i)
                if not val:
                    break

                # Field-specific post-processing
                if field == "nid_number":
                    d = re.sub(r'\D', '', val)
                    if len(d) >= 10:
                        val = d
                    else:
                        break
                elif field == "dob":
                    val = re.sub(r'[IiLl](?=\d)', '1', val)
                elif field == "name_en":
                    val = clean_en_name(val) or val

                sc = score_value(val, field)
                if sc > 0:
                    candidates[field].append((sc, val))
                break

    _NAME_STOP_WORDS = {
        "date", "of", "birth", "father", "mother", "nid", "id",
        "card", "national", "bangladesh", "voter", "republic", "government",
    }

    def _extract_inline_name(text: str) -> str:
        # Pattern 1: Standard names
        pat = re.compile(r'\bName\s*:\s*([A-Za-z][A-Za-z\s\.\-]{1,60})', re.I)
        m = pat.search(text)
        if m:
            raw = m.group(1).strip()
            words = raw.split()
            clean = []
            for w in words:
                stripped = re.sub(r'[^A-Za-z\.]', '', w)
                if not stripped:
                    break
                if len(stripped) < 2:   # single-letter initial like 'S' or 'M' 
                    clean.append(stripped)
                    continue
                if stripped.lower() in _NAME_STOP_WORDS:
                    break
                if (stripped == stripped.upper() or 
                    stripped == stripped.capitalize() or 
                    len(stripped) <= 2):
                    clean.append(stripped)
                else:
                    break
            if len(clean) >= 2: 
                return ' '.join(clean)
    
        return ""
    
    INLINE_PATTERNS = [
        ("name_en",    re.compile(r'\bName\s*:', re.I),   _extract_inline_name),
        ("dob",        re.compile(
            r'\bDate\s+of\s+Birth\s*:\s*'
            r'(\d{1,2}\s+\w{3,9}\s+\d{4}|\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}|\d{4}[\/\-]\d{2}[\/\-]\d{2})',
            re.I), None),
        ("nid_number", re.compile(r'\b(?:ID\s*NO|NID\s*NO|NID)\s*[:\.]?\s*(\d{10,17})', re.I), None),
        ("father_bn",  re.compile(r'\bFather\s*(?:Name)?\s*:\s*([A-Za-z\u0980-\u09FF][A-Za-z\u0980-\u09FF\s\.\-]{1,50}?)(?=\s*[:\|]|$)', re.I), None),
        ("mother_bn",  re.compile(r'\bMother\s*(?:Name)?\s*:\s*([A-Za-z\u0980-\u09FF][A-Za-z\u0980-\u09FF\s\.\-]{1,50}?)(?=\s*[:\|]|$)', re.I), None),
    ]

    for pass_text in passes:
        if not pass_text:
            continue
        for field, pat, extractor in INLINE_PATTERNS:
            if extractor is not None:
                val = extractor(pass_text)
            else:
                m = pat.search(pass_text)
                if not m:
                    continue
                val = m.group(1).strip()
            if not val:
                continue
            if field == "nid_number":
                val = re.sub(r'\D', '', val)
            sc = score_value(val, field)
            if sc > 0:
                candidates[field].append((sc, val))

    for pass_text in passes:
        if pass_text:
            parse_one_pass(pass_text)

    def best(field: str) -> str:
        pool = candidates[field]
        if not pool:
            return ""
        pool.sort(key=lambda x: x[0], reverse=True)
        return pool[0][1]

    name_en    = best("name_en")
    name_bn    = best("name_bn")
    father_bn  = best("father_bn")
    mother_bn  = best("mother_bn")
    nid_number = best("nid_number")
    dob        = best("dob")

    if not nid_number:
        for text in passes:
            # Search full text first (handles single-line OCR output)
            nums = re.findall(r'\b(\d{10}|\d{13}|\d{17})\b', text or "")
            if nums:
                nid_number = nums[0]
                break

    if not dob:
        date_pats = [
            r'\b(\d{1,2}\s+\w{3,9}\s+\d{4})\b',
            r'\b(\d{1,2}[\-/]\w{3,9}[\-/]\d{4})\b',
            r'\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})\b',
            r'\b(\d{4}[/\-\.]\d{2}[/\-\.]\d{2})\b',
        ]
        for text in passes:
            full_text = text or ""
            for pat in date_pats:
                for m in re.finditer(pat, full_text, re.IGNORECASE):
                    cand = m.group(1)
                    ctx_start = max(0, m.start() - 20)
                    ctx = full_text[ctx_start:m.start()].lower()
                    if re.search(r'\b(id\s*no|nid\s*no)\b', ctx):
                        continue
                    if not re.match(r'^\d{10,}$', re.sub(r'\D', '', cand)):
                        dob = cand
                        break
                if dob:
                    break
            if dob:
                break

    if not name_en:
        en_cands = []
        for text in passes:
            bangla_seen = False
            for line in (text or "").splitlines():
                if re.search(r'[\u0980-\u09FF]', line):
                    bangla_seen = True
                    continue
                if bangla_seen:
                    ll = line.strip()
                    if (re.match(r'^[A-Za-z][A-Za-z\s\.\-]{2,}$', ll)
                            and not re.search(r'\d', ll)
                            and not is_header(ll)
                            and len(ll) >= 4
                            and not any(kw in ll.lower() for kw in [
                                "nid", "id", "date", "birth", "dob",
                                "bangladesh", "election", "commission",
                                "voter", "republic", "government",
                                "national", "card"])):
                        sc = score_value(ll, "name_en")
                        if sc > 0:
                            en_cands.append((sc, ll))
        if en_cands:
            en_cands.sort(key=lambda x: x[0], reverse=True)
            name_en = en_cands[0][1]

    if not name_bn:
        SKIP_BN = {"গণপ্রজাতন্ত্রী", "বাংলাদেশ", "সরকার", "জাতীয়",
                   "নির্বাচন", "কমিশন", "পরিচয়", "পিতা", "মাতা",
                   "জন্ম", "তারিখ", "রক্ত"}
        bn_cands = []
        for text in passes:
            header_passed = False
            for line in (text or "").splitlines():
                if any(w in line for w in ["বাংলাদেশ সরকার", "National ID", "জাতীয় পরিচয়"]):
                    header_passed = True
                    continue
                if header_passed and re.match(r'^[\u0980-\u09FF\s\.\-]+$', line):
                    words = set(line.split())
                    if not (words & SKIP_BN) and len(line.strip()) >= 4:
                        sc = score_value(line.strip(), "name_bn")
                        if sc > 0:
                            bn_cands.append((sc, line.strip()))
        if bn_cands:
            bn_cands.sort(key=lambda x: x[0], reverse=True)
            name_bn = bn_cands[0][1]

    if name_en:
        name_en = re.split(r'[,\d]', name_en)[0].strip()
        if len(name_en) > 60:
            name_en = ""

    print(
        f"[NID_PARSE] name_en='{name_en}'  name_bn='{name_bn}'  "
        f"father='{father_bn}'  mother='{mother_bn}'  "
        f"nid='{nid_number}'  dob='{dob}'",
        flush=True
    )

    return {
        "name":       name_en,
        "name_en":    name_en,
        "name_bn":    name_bn,
        "father_bn":  father_bn,
        "mother_bn":  mother_bn,
        "nid_number": nid_number,
        "dob":        dob,
    }
    
def compute_image_quality(image_bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    brightness = float(np.mean(gray))

    # Brightness penalty: too dark (<60) or too bright (>200) reduces score
    brightness_score = 1.0
    if brightness < 60:
        brightness_score = brightness / 60.0
    elif brightness > 200:
        brightness_score = (255 - brightness) / 55.0

    score = (sharpness * 0.7) + (brightness_score * 30 * 0.3)

    return {
        "sharpness":  round(sharpness, 2),
        "brightness": round(brightness, 2),
        "score":      round(score, 2),
    }

# ARCFACE FACE MATCHING
def get_face_data(image_bgr: np.ndarray, label: str) -> tuple:
    faces = face_app.get(image_bgr)

    if not faces:
        raise HTTPException(
            status_code=400,
            detail=f"No face detected in {label}. Please ensure the face is clearly visible and well-lit."
        )

    best_face = max(faces, key=lambda f: f.det_score)

    x1, y1, x2, y2 = [int(v) for v in best_face.bbox]
    h, w = image_bgr.shape[:2]
    pad = 30
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    face_crop = image_bgr[y1:y2, x1:x2]

    return best_face.normed_embedding, face_crop


def arcface_confidence_label(similarity: float) -> str:
    if similarity >= 0.65:
        return "Very High"
    elif similarity >= 0.55:
        return "High"
    elif similarity >= 0.45:
        return "Medium"
    else:
        return "Low"


def verify_faces(id_img: np.ndarray, selfie_img: np.ndarray) -> dict:
    id_embedding, id_face_crop = get_face_data(id_img, label="ID card image")

    selfie_embedding, selfie_face_crop = get_face_data(selfie_img, label="selfie")

    similarity = float(np.dot(id_embedding, selfie_embedding))

    THRESHOLD = 0.45
    matched   = similarity >= THRESHOLD

    return {
        "match":           matched,
        "message":         "Faces match" if matched else "Faces do not match",
        "similarity":      round(similarity, 4),
        "confidence":      arcface_confidence_label(similarity),
        "threshold":       THRESHOLD,
        "method":          "ArcFace (InsightFace buffalo_l — RetinaFace + ArcFace ResNet-50)",
        "nid_face_image":  image_to_base64(id_face_crop),
        "live_face_image": image_to_base64(selfie_face_crop),
    }

@app.post("/validate-id")
async def validate_id(id_image: UploadFile = File(...)):
    image_bytes = await id_image.read()
    image_bgr = decode_image(image_bytes, "ID image")
    extracted_text = extract_text(image_bgr)
    result = detect_id_type(extracted_text)

    if not result["valid"]:
        raise HTTPException(status_code=400, detail=result["message"])

    ocr_passes = extract_text_paddle(image_bgr)
    nid_data = extract_nid_fields_from_passes(ocr_passes)
    nid_data = enrich_bangla_fields(nid_data, image_bgr)

    return {
        "valid": result["valid"],
        "id_type": result["id_type"],
        "message": result["message"],
        "nid_data": nid_data,
    }

@app.post("/verify-face")
async def verify_face(
    id_image:     UploadFile = File(...),
    selfie_image: UploadFile = File(...)
):
    id_bytes     = await id_image.read()
    selfie_bytes = await selfie_image.read()
    id_img       = decode_image(id_bytes,     "ID image")
    selfie_img   = decode_image(selfie_bytes, "selfie image")
    return verify_faces(id_img, selfie_img)


@app.post("/verify-with-liveness")
async def verify_with_liveness(
    session_id:   str,
    id_image:     UploadFile = File(...),
    selfie_image: UploadFile = File(None),
):
    session = get_session(session_id)

    if not session:
        raise HTTPException(
            status_code=400,
            detail="Session not found or expired. Please complete the liveness check again."
        )

    # SECURITY CHECK 1: Spoof flag 
    if session.get("spoof_flagged"):
        scores = session.get("consistency_scores", {})
        delete_session(session_id)
        raise HTTPException(
            status_code=403,
            detail=(
                f"Identity spoofing attempt detected: different faces were present "
                f"during the liveness verification stages. "
                f"Consistency scores: {scores}. "
                f"Please restart the verification process."
            )
        )

    # SECURITY CHECK 2: Liveness must have passed
    if not session.get("liveness_passed"):
        stage = session.get("stage", "unknown")
        raise HTTPException(
            status_code=400,
            detail=(
                f"Liveness check not completed (current stage: {stage}). "
                f"Please finish all 3 challenges: blink, head turn, motion."
            )
        )

    if session.get("consistency_ok") is False:
        scores = session.get("consistency_scores", {})
        delete_session(session_id)
        raise HTTPException(
            status_code=403,
            detail=(
                f"Face consistency check failed: captures during liveness "
                f"check do not all match the same person. "
                f"Scores: {scores}."
            )
        )

    id_bytes = await id_image.read()
    id_img   = decode_image(id_bytes, "ID image")

    auto_selfie_bytes = session.get("auto_selfie")

    if auto_selfie_bytes:
        selfie_img = decode_image(auto_selfie_bytes, "auto-captured selfie")
    elif selfie_image is not None:
        selfie_bytes = await selfie_image.read()
        selfie_img   = decode_image(selfie_bytes, "uploaded selfie")
    else:
        raise HTTPException(
            status_code=400,
            detail="No selfie available in session. Please redo the liveness check."
        )

    result = verify_faces(id_img, selfie_img)

    ocr_passes = extract_text_paddle(id_img)
    nid_data = extract_nid_fields_from_passes(ocr_passes)
    nid_data = enrich_bangla_fields(nid_data, id_img)
    result["nid_data"] = nid_data

    result["consistency_ok"]     = session.get("consistency_ok", True)
    result["consistency_scores"] = session.get("consistency_scores", {})
    result["liveness_captures"]  = 3

    delete_session(session_id)

    return result

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "securebank-face-verification-api",
        "version": "6.3.0",
        "face_recognition_pipeline": {
            "library": "InsightFace", "model_pack": "buffalo_l",
            "detector": "RetinaFace", "embedder": "ArcFace ResNet-50",
            "threshold": 0.35,
        },
        "ocr": {
            "english_engine": "PaddleOCR PP-OCRv4 (latin, lazy-loaded)",
            "bangla_engine": "bangla_ocr.py — EAST DL + Tesseract ben+eng",
            "bangla_strategies": [
                "S1: EAST frozen_east_text_detection.pb → per-region Tesseract",
                "S2: Multi-pass full-card Tesseract (9 passes: 3 preproc x 3 PSM)",
                "S3: Spatial label-anchor (pytesseract.image_to_data bbox search)",
            ],
        },
        "endpoints": [
            "POST /validate-id", "POST /verify-face", "POST /verify-with-liveness",
            "POST /liveness/start", "GET /health"
        ]
    }