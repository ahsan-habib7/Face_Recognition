# RecognizeMe — Bangladesh Bank KYC Identity Verification System

A fully offline, bank-grade KYC (Know Your Customer) identity verification system built for Bangladesh financial institutions. It combines **ArcFace biometric face matching**, a **4-stage liveness detection pipeline**, and **dual-engine OCR** (PaddleOCR + EAST + Tesseract) to verify a person's identity against their NID card in real time — entirely within your own infrastructure. No external API calls, no data leaves your server.

---

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Backend — Python FastAPI](#backend--python-fastapi)
- [Frontend — ASP.NET Core MVC](#frontend--aspnet-core-mvc)
- [Models & Dependencies](#models--dependencies)
- [Prerequisites](#prerequisites)
- [Installation & Running](#installation--running)
  - [Docker (Recommended)](#docker-recommended)
  - [Local Development (Without Docker)](#local-development-without-docker)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Third-Party Notices](#third-party-notices)

---

## Features

- **ArcFace Face Matching** — InsightFace buffalo_l (RetinaFace detector + ArcFace ResNet-50 embedder). Produces 512-dimensional L2-normalised embeddings matched via cosine similarity.
- **4-Stage Liveness Detection** — Align → Blink (×3) → Head Turn (R→L→R) → Motion. Fully passive, no special hardware required.
- **Multi-Stage Anti-Spoofing** — Three ArcFace embeddings are captured at different challenge boundaries and compared pairwise to detect subject-swap attacks.
- **Dual-Engine OCR** — PaddleOCR (PP-OCRv4, latin mode) for English and numeric fields; EAST deep learning detector + Tesseract Bengali for Bangla fields (name, father, mother).
- **Bangladesh ID Support** — Detects and extracts data from NID cards.
- **Fully Offline** — All ML models run locally. No internet connection required after the first model download.
- **No Permanent Storage** — Images are processed entirely in memory and discarded after verification.
- **CSRF Protection** — All form POST endpoints use ASP.NET's built-in anti-forgery token validation.
- **Session Auto-Expiry** — Liveness sessions expire after 10 minutes; a background cleanup task runs every 5 minutes.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                      BROWSER (User's device)                         │
│                                                                      │
│   Index.cshtml                                                       │
│   ┌─────────────────┐   ┌─────────────────────────────────────────┐ │
│   │  ID Card Upload │   │  Camera Modal (Liveness Check)          │ │
│   │  (drag / click) │   │                                         │ │
│   │                 │   │  Stage 1: ALIGN  — hold face frontal    │ │
│   │                 │   │  Stage 2: BLINK  — blink 3 times       │ │
│   │                 │   │  Stage 3: TURN   — look L → R → L      │ │
│   │                 │   │  Stage 4: MOTION — natural movement     │ │
│   │                 │   │  Stage 5: CAPTURE — auto selfie taken   │ │
│   └─────────────────┘   └─────────────────────────────────────────┘ │
│                                       │ AJAX frames @ 120 ms        │
│                               ┌───────▼───────┐                     │
│                               │  Form submit  │                     │
│                               │  POST /Verify │                     │
│                               └───────┬───────┘                     │
└───────────────────────────────────────│─────────────────────────────┘
                                        │ HTTP POST
┌───────────────────────────────────────▼─────────────────────────────┐
│             ASP.NET Core MVC (.NET 8) — Port 5000                   │
│                                                                      │
│  VerificationController                                              │
│    GET  /Verification/Index        → KYC start page                 │
│    POST /Verification/ValidateId   → AJAX → JSON (ID check)         │
│    POST /Verification/LivenessStart  ─────────────── proxy ───┐     │
│    POST /Verification/LivenessFrame  ─ forward to Python ─────┤     │
│    POST /Verification/LivenessCancel ─────────────────────────┘     │
│    POST /Verification/Verify       → ArcFace + OCR → Result view    │
│                                                                      │
│  FaceVerificationService (typed HttpClient, 90 s timeout)           │
│    ValidateIdAsync()           → POST /validate-id                  │
│    VerifyWithLivenessAsync()   → POST /verify-with-liveness         │
└────────────────────────────┬────────────────────────────────────────┘
                             │ Internal Docker network
                             │ http://face-api:8000
┌────────────────────────────▼────────────────────────────────────────┐
│             Python FastAPI — Port 8000 (v6.4.0)                     │
│                                                                      │
│  POST /validate-id            PaddleOCR + keyword match → ID type   │
│  POST /verify-face            Direct ArcFace face comparison        │
│  POST /verify-with-liveness   Full KYC: session + ArcFace + OCR     │
│  POST /liveness/start         Create UUID session                   │
│  POST /liveness/frame         Process one webcam frame              │
│  POST /liveness/cancel        Cancel and delete session             │
│  GET  /health                 Health check                          │
│                                                                      │
│  Models loaded at startup:                                           │
│    buffalo_l/det_10g.onnx      RetinaFace face detector             │
│    buffalo_l/2d106det.onnx     106-point landmark detector          │
│    buffalo_l/w600k_r50.onnx    ArcFace ResNet-50 embedder           │
│    face_landmarker.task        MediaPipe 478-point mesh (blink EAR) │
│    frozen_east_text_detection.pb  EAST text detector (Bangla OCR)   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
RecognizeMe/
├── Backend/
│   ├── main.py                          API entry point; all core endpoints
│   ├── bangla_ocr.py                    Bangla field extractor (EAST + Tesseract)
│   ├── requirements.txt                 Python package dependencies
│   ├── Dockerfile                       Python container build
│   ├── docker-compose.yml               Docker Compose for the backend
│   ├── face_landmarker.task             MediaPipe 478-pt face mesh model (~3.6 MB)
│   ├── frozen_east_text_detection.pb    EAST text detector for Bangla OCR (~90 MB)
│   ├── EAST_MODEL_SETUP.md              EAST model download & Tesseract setup guide
│   ├── THIRD_PARTY_NOTICES.txt          Open-source licence notices
│   └── liveness/
│       ├── router.py                    FastAPI router: /start, /frame, /cancel
│       ├── service.py                   Core liveness logic; all 5 challenge stages
│       ├── session.py                   In-memory session store with auto-cleanup
│       └── utils.py                     Shared helper utilities
│
└── Frontend/
    ├── Dockerfile                       .NET 8 multi-stage Docker build
    └── Files/
        ├── FaceRecognition_Frontend.csproj
        ├── Program.cs                   App startup: DI, HttpClients, Kestrel limits
        ├── appsettings.json             App configuration (FaceApiBaseUrl)
        ├── appsettings.Development.json Development overrides
        ├── Controllers/
        │   ├── VerificationController.cs  Main KYC controller
        │   └── HomeController.cs          Error page controller
        ├── Services/
        │   └── FaceVerificationService.cs Typed HttpClient + all ViewModels
        ├── Models/
        │   └── ErrorViewModel.cs
        ├── Views/
        │   ├── Verification/
        │   │   ├── Index.cshtml         KYC workflow page (upload + liveness)
        │   │   └── Result.cshtml        Verification result page
        │   ├── Shared/
        │   │   ├── _Layout.cshtml
        │   │   ├── _Layout.cshtml.css
        │   │   ├── _ValidationScriptsPartial.cshtml
        │   │   └── Error.cshtml
        │   ├── _ViewImports.cshtml
        │   └── _ViewStart.cshtml
        ├── wwwroot/
        │   ├── css/site.css
        │   ├── js/liveness.js           Client-side liveness frame loop & UI
        │   └── favicon.ico
        └── Properties/
            └── launchSettings.json      Dev launch profiles
```

---

## Backend — Python FastAPI

The backend is a single FastAPI application (`main.py`) running on Uvicorn with one worker process.

### Face Recognition

The ArcFace pipeline runs in four steps:

1. **Detection** — RetinaFace (`det_10g.onnx`) at 640×640 input detects all faces and selects the one with the highest confidence score.
2. **Alignment** — 5-point landmark alignment is performed automatically by InsightFace.
3. **Embedding** — ArcFace ResNet-50 (`w600k_r50.onnx`) produces a 512-dimensional L2-normalised vector.
4. **Matching** — Cosine similarity (dot product of two L2-normalised embeddings).

Similarity thresholds used for the result badge:

| Similarity | Status |
|------------|--------|
| ≥ 0.55 | Verified |
| 0.35 – 0.54 | Needs Review |
| < 0.35 | Not Verified |

### OCR Pipeline

**PaddleOCR (PP-OCRv4, latin mode)** handles all English and numeric fields. It is lazy-loaded on the first request to avoid startup delays. Three passes run per image — a standard pass, a high-resolution pass at 2000px width, and a per-region ROI pass. All results feed the multi-pass field parser.

**Fields extracted from the ID card:**

| Field | Description |
|-------|-------------|
| `name_en` | Cardholder name in English |
| `name_bn` | Cardholder name in Bangla (নাম) |
| `father_bn` | Father's name in Bangla (পিতা) |
| `mother_bn` | Mother's name in Bangla (মাতা) |
| `nid_number` | 10, 13, or 17-digit NID number |
| `dob` | Date of birth |

**Bangla OCR** (`bangla_ocr.py`) runs three independent strategies in parallel and picks the best candidate per field by scoring:

- **Strategy 1 — EAST Text Detector**: Locates text regions with `frozen_east_text_detection.pb`, crops each region, and runs Tesseract (`ben+eng`, PSM 7) on each crop individually. The EAST model is lazy-loaded and cached after the first load.
- **Strategy 2 — Multi-Pass Tesseract**: 9 full-card passes (3 preprocessing variants × 3 PSM modes: uniform block, single column, auto) plus a bottom-half crop pass.
- **Strategy 3 — Spatial Label-Anchor**: Uses `pytesseract.image_to_data()` bounding boxes to find label words (`নাম`, `পিতা`, `মাতা`) by pixel position and reads the value text beside or below each label.

If Tesseract is not installed, all three Bangla strategies are skipped silently — the main API continues working with English fields only.

### Liveness Detection

The liveness pipeline has 5 sequential stages, processed at ~8 fps (120 ms frame interval, 80 ms minimum enforced server-side):

| Stage | Challenge | Completion Criteria |
|-------|-----------|---------------------|
| ALIGN | Hold face inside the oval | Yaw ≤ ±14°, Pitch ≤ ±12°, Laplacian variance ≥ 60, held for 8 consecutive frames |
| BLINK | Blink naturally 3 times | EAR < 0.21 for ≥ 2 frames (closed) → EAR > 0.27 for ≥ 3 frames (open), repeated 3 times |
| TURN | Look right, then left, then right | Nose/eye-midpoint ratio: < −0.18 (left, 5 frames) → > +0.18 (right, 5 frames) → return center |
| MOTION | Natural head movement | ≥ 2 of the last 5 frames exceed optical flow magnitude of 3.0 |
| DONE | Auto-selfie captured | JPEG stored in session; anti-spoofing consistency check runs |

### Anti-Spoofing — Multi-Stage Consistency Check

Three selfies are captured at distinct challenge boundaries (blink→turn, turn→motion, and the final motion capture). Their ArcFace embeddings are compared pairwise before the session is marked as passed:

| Pair | Type | Threshold |
|------|------|-----------|
| Capture 1 vs 2 | Frontal vs Profile | 0.22 |
| Capture 1 vs 3 | Frontal vs Frontal | 0.30 |
| Capture 2 vs 3 | Profile vs Frontal | 0.22 |

If any pair fails, the session is flagged and verification is refused with HTTP 403. The result page renders a "Spoofing Detected" badge.

### Session Management

Sessions are stored in a Python in-memory dictionary (`liveness/session.py`). Each session holds all liveness stage state variables, three JPEG captures with ArcFace embeddings, consistency check results, and the auto-selfie used by the verify endpoint. A background asyncio task removes sessions idle for more than **10 minutes**, running every **5 minutes**.

---

## Frontend — ASP.NET Core MVC

The frontend is a .NET 8 MVC application. It serves the UI and proxies all communication with the Python backend — the browser never contacts the Python API directly.

### Controllers

**`VerificationController`** handles all KYC actions:

| Action | Route | Method | Description |
|--------|-------|--------|-------------|
| `Index` | `/Verification/Index` | GET | Renders the KYC start page |
| `ValidateId` | `/Verification/ValidateId` | POST | AJAX: sends ID image to Python OCR; returns JSON |
| `Verify` | `/Verification/Verify` | POST | Full KYC: calls Python ArcFace + OCR; renders Result view |
| `LivenessStart` | `/Verification/LivenessStart` | POST | Proxy to Python `/liveness/start` |
| `LivenessFrame` | `/Verification/LivenessFrame` | POST | Proxy: forwards webcam frame to Python |
| `LivenessCancel` | `/Verification/LivenessCancel` | POST | Proxy: cancels liveness session |

### Services

**`FaceVerificationService`** is a typed HttpClient service. It exposes:

- `ValidateIdAsync(IFormFile)` — posts to `/validate-id`; returns `IdValidationResult`
- `VerifyWithLivenessAsync(IFormFile, string sessionId)` — posts to `/verify-with-liveness?session_id=`; returns `FaceVerificationResult`

**ViewModels** (in `FaceVerificationService.cs`):

- **`NidData`** — OCR fields: `Name`, `NameEn`, `NameBn`, `FatherBn`, `MotherBn`, `NidNumber`, `Dob`
- **`FaceVerificationResult`** — full result ViewModel with `Match`, `Similarity`, `MatchPercentage` (computed), `MatchStatus` (computed), `Confidence`, `NidFaceImage`, `LiveFaceImage`, `NidData`, field match flags (`NameMatch`, `NidMatch`, `DobMatch`, etc.), echoed user-input fields, `Error`, and anti-spoofing data (`ConsistencyOk`, `ConsistencyScores`, `SpoofFlagged`)

### Views

**`Index.cshtml`** — The KYC entry page. Dark navy and gold design, two-column layout:
- Left: hero text and a 3-step progress indicator (Upload ID → Liveness Check → Verify)
- Right: drag-and-drop ID card upload zone, optional identity input fields (Name, NID number, DOB, Father's name, Mother's name), and a camera modal for liveness
- The camera modal shows an animated oval overlay, pill-shaped challenge indicators, blink dot counter, and head-turn direction boxes updated in real time
- `liveness.js` manages the 120 ms frame-sending loop and all UI state transitions

**`Result.cshtml`** — The verification result page:
- Left: step indicators, side-by-side face crops (NID vs selfie), OCR data table with match icons (✅ / ❌) per field
- Right: status badge (Verified / Needs Review / Not Verified / Spoofing Detected), animated similarity percentage meter, confidence label, KYC use-case grid, and Try Again / Home navigation

---

## Models & Dependencies

### Python Packages

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.111.0 | Web framework |
| uvicorn[standard] | 0.29.0 | ASGI server |
| insightface | 0.7.3 | ArcFace face recognition (buffalo_l) |
| onnxruntime | 1.17.3 | ONNX model inference |
| opencv-python-headless | 4.9.0.80 | Image processing, EAST inference, optical flow |
| numpy | 1.26.4 | Array operations |
| mediapipe | 0.10.14 | 478-point face mesh for blink EAR |
| scipy | 1.13.1 | Euclidean distance for EAR computation |
| paddlepaddle | 2.6.2 | PaddleOCR inference backend |
| paddleocr | 2.8.1 | OCR engine (PP-OCRv4, latin mode) |
| pytesseract | 0.3.13 | Bangla OCR wrapper |
| Pillow | ≥ 10.0.0 | Image preprocessing for Tesseract |
| imutils | ≥ 0.5.4 | Non-max suppression for EAST bounding boxes |
| Cython | 3.0.10 | Build dependency for pyclipper |
| python-multipart | 0.0.9 | Multipart form parsing for FastAPI |

### ML Model Files

| File | Size | How to obtain |
|------|------|---------------|
| `buffalo_l/` (3 ONNX files) | ~300 MB | Downloaded automatically by InsightFace on first run |
| `face_landmarker.task` | ~3.6 MB | Bundled in `Backend/` |
| `frozen_east_text_detection.pb` | ~90 MB | Bundled in `Backend/` (see `EAST_MODEL_SETUP.md` if missing) |

### System Packages (installed in Docker)

- `tesseract-ocr`, `tesseract-ocr-ben`, `tesseract-ocr-eng`
- `libgl1`, `libglib2.0-0`, `libgomp1`, `libsm6`, `libxext6`, `libxrender-dev`
- `gcc`, `g++`, `build-essential`, `libstdc++6`

### .NET Frontend

- Target framework: **net8.0** (ASP.NET Core MVC)
- No additional NuGet packages — all HTTP client, MVC, and JSON functionality is included in `Microsoft.NET.Sdk.Web`

---

## Prerequisites

- **Docker Desktop** (Windows / macOS) or **Docker Engine + Compose** (Linux)
- Minimum **8 GB RAM**
- A **webcam** for liveness detection
- An internet connection **on first run only** to download the InsightFace buffalo_l model pack (~300 MB)

---

## Installation & Running

### Docker (Recommended)

**Step 1 — Start the Python backend**

```bash
cd RecognizeMe/Backend
docker-compose up --build
```

The first build takes **3–5 minutes** (installs all packages and caches ArcFace models). Subsequent starts take ~30 seconds. The backend will be available at `http://localhost:8000`.

**Step 2 — Build and run the .NET frontend**

```bash
cd RecognizeMe/Frontend/Files
dotnet run .
```

> **Linux note:** Replace `host.docker.internal` with your actual host IP, or place both containers on the same Docker network and use `http://face-api:8000`.

**Step 3 — Open the application**

```
http://localhost:5077
```

**Viewing logs:**

```bash
# Python backend (face detection, OCR output, liveness stage transitions)
docker logs -f face_api

# .NET frontend
docker logs -f face_frontend
```

---

### Local Development (Without Docker)

**Python backend:**

```bash
cd RecognizeMe/Backend
pip install -r requirements.txt
```

Install Tesseract with Bengali language support:

```bash
# Ubuntu / Debian
sudo apt-get install -y tesseract-ocr tesseract-ocr-ben

# macOS
brew install tesseract tesseract-lang

# Windows — download from https://github.com/UB-Mannheim/tesseract/wiki
# Ensure ben.traineddata is in your tessdata/ folder
```

Start the server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

On the first request, InsightFace will automatically download the `buffalo_l` model pack (~300 MB) to `~/.insightface/models/`.

**ASP.NET frontend:**

```bash
cd RecognizeMe/Frontend/Files
dotnet run
```

Available at `http://localhost:5077` (HTTP) or `https://localhost:7264` (HTTPS). `appsettings.json` already points `FaceApiBaseUrl` to `http://localhost:8000`.

---

## Configuration

### Backend

No configuration file. CORS allowed origins in `main.py` include `localhost:5000`, `localhost:7000`, `localhost:5172`, `localhost:3000`, and `frontend:5000`.

### Frontend

**`appsettings.json`:**
```json
{
  "FaceApiBaseUrl": "http://localhost:8000"
}
```

In Docker, override this via the `FaceApiBaseUrl` environment variable:
```
FaceApiBaseUrl=http://face-api:8000
```

**Request body size limit:** 50 MB (configured in `Program.cs`). This accommodates large ID card images and base64 face crops.

**HttpClient timeouts:**
- `FaceVerificationService` (typed client): **90 seconds** — allows for ArcFace model cold-start on the first request
- `"PythonApi"` named client (liveness proxy): **30 seconds**

---

## API Reference

### `POST /validate-id`

Validates an uploaded ID card image using OCR and returns extracted fields.

**Request:** `multipart/form-data` — field `id_image` (JPEG or PNG)

**Response:**
```json
{
  "valid": true,
  "id_type": "Bangladesh NID",
  "message": "Valid Bangladesh National ID Card detected",
  "nid_data": {
    "name": "Mohammad Ali",
    "name_en": "Mohammad Ali",
    "name_bn": "মোহাম্মদ আলী",
    "father_bn": "আবদুল করিম",
    "mother_bn": "ফাতেমা বেগম",
    "nid_number": "1234567890",
    "dob": "01 Jan 1990"
  }
}
```

### `POST /verify-with-liveness?session_id=`

Runs the full KYC verification using the auto-selfie stored in the liveness session.

**Request:** `multipart/form-data` — field `id_image` (JPEG or PNG) + `session_id` query parameter

**Response:**
```json
{
  "match": true,
  "message": "Faces match",
  "similarity": 0.7241,
  "confidence": "High",
  "threshold": 0.45,
  "method": "ArcFace (InsightFace buffalo_l — RetinaFace + ArcFace ResNet-50)",
  "nid_face_image": "<base64 JPEG>",
  "live_face_image": "<base64 JPEG>",
  "nid_data": { "..." },
  "consistency_ok": true,
  "consistency_scores": { "1v2": 0.51, "1v3": 0.68, "2v3": 0.45 },
  "liveness_captures": 3
}
```

### `POST /liveness/start`

Creates a new liveness session.

**Response:** `{ "session_id": "uuid", "blink_needed": 3 }`

### `POST /liveness/frame?session_id=`

Processes one webcam frame.

**Request:** `multipart/form-data` — field `frame` (JPEG)

**Response:** `{ "stage": "blink", "progress": 33, "icon": "👁", "message": "Blink 2 more times", "blink_count": 1, "blink_needed": 3 }`

### `POST /liveness/cancel?session_id=`

Cancels and deletes a liveness session.

**Response:** `{ "cancelled": true }`

### `GET /health`

Returns service status, loaded model info, and available endpoints.

---

## Third-Party Notices

- **MediaPipe Face Landmarker** by Google — Apache License 2.0
- **InsightFace buffalo_l** (RetinaFace + ArcFace ResNet-50) — MIT License
- **PaddleOCR PP-OCRv4** — Apache License 2.0
- **Tesseract OCR** — Apache License 2.0
- **EAST Frozen Text Detection Model** — see [source repository](https://github.com/oyyd/frozen_east_text_detection.pb)
- **FastAPI** — MIT License
- **OpenCV** — Apache License 2.0

See `Backend/THIRD_PARTY_NOTICES.txt` for the notices bundled with this project.
