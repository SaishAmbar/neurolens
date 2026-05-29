# NeuroLens — Project Reference

> **Purpose of this file:** Quick-reference doc for understanding the full codebase at a glance. Generated after a full read-through of every file.

---

## 🧠 What Is NeuroLens?

**NeuroLens** is a real-time, browser-based multimodal AI perception system with four core features:

| Feature | Input | Model / Method |
|---|---|---|
| Emotion Recognition | Webcam (face) | MediaPipe FaceLandmarker + blendshape scoring |
| Focus / Attention Detection | Webcam (face) | MediaPipe FaceLandmarker + blink/gaze/head heuristics |
| Sign Language Recognition | Webcam (hands) | MediaPipe HandLandmarker + custom rule-based classifier |
| Obstacle Detection | Webcam (scene) | TensorFlow.js COCO-SSD (MobileNet v2, lazy-loaded) |
| Text Scanner (OCR + Summary) | Webcam (frozen frame) | Tesseract.js v5 OCR + Ollama (qwen2.5:7b) |

Everything runs **locally in the browser** — no data is sent to any server.  
There is also an optional **Raspberry Pi Camera Module 3** MJPEG stream backend for external camera support.

---

## 📁 File Map

```
Backup-Saish - Gemini/
├── index.html               ← Main UI shell (all 5 tabs, onboarding, theme switcher)
├── index.css                ← All styles (dark / light / calm themes, panels, animations)
├── requirements.txt         ← Python deps for the Pi-side server (opencv, flask)
├── camera_stream.py         ← Pi Camera Module 3 Flask MJPEG server (run on Raspberry Pi)
├── deploy_to_pi.py          ← Script to deploy camera_stream.py to the Pi over SSH
│
├── js/                      ← Browser JavaScript (ES modules)
│   ├── app.js               ← ★ MAIN ORCHESTRATOR — camera, MediaPipe init, detect loop, UI
│   ├── emotion-analyzer.js  ← Emotion recognition logic
│   ├── focus-analyzer.js    ← Focus / attention detection logic
│   ├── sign-language-analyzer.js  ← Sign language top-level analyzer
│   ├── neurolens-analyzer.js      ← Obstacle detection (COCO-SSD wrapper + HUD)
│   ├── text-scanner.js            ← Text Scanner (Tesseract.js OCR + Ollama summarization)
│   ├── visual-quality.js          ← Frame quality assessment (lighting, blur, occlusion)
│   └── sign/                      ← Sign language sub-modules
│       ├── features.js            ← Hand landmark → feature vector extractor
│       ├── motion.js              ← Motion tracker (velocity, pattern detection)
│       ├── static-classifier.js   ← Rule-based sign classifier (30-word vocab)
│       ├── dynamic-classifier.js  ← Motion-pattern sign classifier (supplementary)
│       ├── word-engine.js         ← Stability + locking + cooldown engine
│       └── sentence-builder.js    ← ASL gloss → English sentence converter
│
├── models/                  ← YOLO model files (for Python-side obstacle detection)
│   ├── yolov4-tiny.cfg      ← YOLOv4-tiny architecture config
│   ├── yolov4-tiny.weights  ← YOLOv4-tiny weights (~24MB)
│   └── coco.names           ← COCO class labels (80 classes)
│
└── neurolens/               ← Python obstacle detection package (standalone, OpenCV-based)
    ├── __init__.py
    ├── config.py            ← Camera index, resolution, confidence thresholds
    ├── main.py              ← Python entry point (cv2 window display loop)
    ├── detector.py          ← YOLOv4-tiny detector wrapper
    ├── distance_estimator.py← Focal-length distance estimation from bounding boxes
    ├── hud_overlay.py       ← OpenCV HUD rendering (alerts, boxes, stats)
    └── models/
        └── yolov4-tiny.weights  ← Duplicate weights file (also in /models/)
```

---

## 1. 😄 Emotion Recognition

### Files involved
- [`js/emotion-analyzer.js`](js/emotion-analyzer.js) — **Core logic**
- [`js/app.js`](js/app.js) — Lines ~403–412 (integration), `updateEmotionUI()` ~L555–580

### How it works
1. **MediaPipe FaceLandmarker** runs in VIDEO mode, outputting **52 blendshape scores** per frame.
2. `EmotionAnalyzer.analyze(blendshapes)` scores 6 base emotions (**Happy, Sad, Angry, Surprised, Fear, Disgust**) using weighted sums of blendshapes.
3. **Contempt** is detected separately via left–right mouth smile asymmetry.
4. **Inhibitors** prevent false positives (e.g. browDown suppresses Sad score).
5. **EMA smoothing** (α=0.35) across frames, plus **15-frame rolling history**.
6. Returns `{ primary, secondary, intensity, confidence, evidence }`.

### Emotions supported
`Happy · Sad · Angry · Surprised · Fear · Disgust · Contempt · Neutral`

### Status: ✅ Fully implemented, no missing dependencies

---

## 2. 🎯 Focus / Attention Detection

### Files involved
- [`js/focus-analyzer.js`](js/focus-analyzer.js) — **Core logic**
- [`js/app.js`](js/app.js) — Lines ~415–424 (integration), `updateFocusUI()` ~L582–597

### How it works
Uses **4 signals** scored 0–100:

| Signal | Max Score | Method |
|---|---|---|
| Blink rate | 30 | Counts `eyeBlinkLeft/Right` threshold crossings over 60s window |
| Gaze variance | 30 | Variance of `eyeLookOut/In/Up/Down` blendshape over last 90 frames |
| Head drift | 25 | Variance of nose tip landmark position over last 90 frames |
| Eye openness | 15 | `1 - eyeBlinkLeft/Right` average |

Maps total score → state:
- ≥85: **Deep Focus** · ≥65: **Focused** · ≥45: **Partially Focused** · ≥25: **Distracted** · <25: **Disengaged**

Uses **majority-vote state smoothing** over a 30-vote window.

### Status: ✅ Fully implemented, no missing dependencies

---

## 3. 🤟 Sign Language Recognition

### Files involved (all in [`js/sign/`](js/sign/))

| File | Role |
|---|---|
| [`sign-language-analyzer.js`](js/sign-language-analyzer.js) | Top-level coordinator |
| [`sign/features.js`](js/sign/features.js) | Extracts 30+ features from 21 hand landmarks |
| [`sign/motion.js`](js/sign/motion.js) | Velocity, trajectory, pattern detection (wave/nod/push/pull/sweep) |
| [`sign/static-classifier.js`](js/sign/static-classifier.js) | Rule-based classifier for 30 whole-word signs |
| [`sign/dynamic-classifier.js`](js/sign/dynamic-classifier.js) | Supplementary motion-pattern classifier |
| [`sign/word-engine.js`](js/sign/word-engine.js) | Stability (4-frame), lock, cooldown (1.5s) engine |
| [`sign/sentence-builder.js`](js/sign/sentence-builder.js) | ASL gloss → English grammar (NOT used in current flow) |

### Vocabulary (30 words)
`I, YOU, HELLO, THANK YOU, PLEASE, YES, NO, STOP, GO, COME, EAT, DRINK, WANT, NEED, HELP, HOME, SCHOOL, FOOD, WATER, GOOD, BAD, SORRY, LOVE, FRIEND, FAMILY, TODAY, TOMORROW, HAPPY, SAD, MORE`

### Pipeline
```
MediaPipe HandLandmarker → extractFeatures() → MotionTracker.update()
  → StaticGestureClassifier.classify() → WordFormationEngine.update()
    → if confirmed → _addWord() → sentence string joined by " → "
```

### Anti-duplicate mechanisms
1. 4-frame stability (same sign for 4 consecutive frames)
2. 70%+ average confidence
3. Gesture lock (frozen until sign changes)
4. 1.5s cooldown between detections
5. 3-frame motion reset to unlock

### ⚠️ Note on `sentence-builder.js` and `dynamic-classifier.js`
These two files exist but are **NOT imported** in the current `sign-language-analyzer.js`. They are extra/unused code. `sentence-builder.js` contains ASL grammar expansion logic that could be wired in, and `dynamic-classifier.js` contains supplementary motion rules.

### Status: ✅ Core pipeline fully implemented. `sentence-builder.js` and `dynamic-classifier.js` are present but unused.

---

## 4. 🚧 Obstacle Detection

### Browser-side (JS — active in the app)

| File | Role |
|---|---|
| [`js/neurolens-analyzer.js`](js/neurolens-analyzer.js) | COCO-SSD wrapper + distance estimation + canvas HUD |
| [`js/app.js`](js/app.js) | Lazy-loads model on tab switch, async detect loop |

**How it works:**
- Uses **TensorFlow.js COCO-SSD (MobileNet v2)** loaded from CDN on first tab switch.
- Detects all 80 COCO classes with ≥35% confidence.
- Estimates distance using focal-length formula: `distance = (knownHeight × focalLength) / bboxHeight`
- Classifies zones: **SAFE (>3m) · NEAR (1.5–3m) · VERY CLOSE (<1.5m)**
- Draws animated HUD on the overlay canvas (grid, scan line, bounding boxes, directional bars, warning panel).
- Sorts by danger score (distance + centrality + object size).

### Python-side (`neurolens/` package — standalone, NOT used by the browser app)

| File | Role |
|---|---|
| [`neurolens/main.py`](neurolens/main.py) | Entry point — OpenCV window display loop |
| [`neurolens/detector.py`](neurolens/detector.py) | YOLOv4-tiny via OpenCV DNN |
| [`neurolens/distance_estimator.py`](neurolens/distance_estimator.py) | Same focal-length distance estimation |
| [`neurolens/hud_overlay.py`](neurolens/hud_overlay.py) | OpenCV HUD drawing |
| [`neurolens/config.py`](neurolens/config.py) | Camera index, thresholds |

Requires `models/yolov4-tiny.weights` (24MB ✅ present) and `models/yolov4-tiny.cfg` (✅ present).

### Status: ✅ Browser JS version fully implemented. Python standalone version fully implemented but separate from the web app.

---

## 5. 📝 Text Scanner (OCR + Summarization)

### Files involved
- [`js/text-scanner.js`](js/text-scanner.js) — **Core logic** (Tesseract.js OCR + Ollama streaming)
- [`js/app.js`](js/app.js) — Scan/Clear button handlers, UI state management, word highlighting

### How it works
1. User clicks **Scan** → current webcam frame is frozen to a canvas
2. **Tesseract.js v5** (loaded from CDN on first use) runs OCR on the frozen frame
3. Detected words are highlighted with bounding boxes on the preview canvas
4. Extracted text is shown in the "Raw OCR" panel
5. Text is sent to **Ollama** (`http://localhost:11434/api/generate`, model `qwen2.5:7b`)
6. Ollama response is **streamed** word-by-word into the "AI Summary" panel
7. Status badge transitions: `IDLE → LOADING → READY → SCANNING → THINKING → DONE`

### Requirements
- Tesseract.js: automatically loaded from CDN
- Ollama: must be running locally with `qwen2.5:7b` model pulled
  ```bash
  ollama pull qwen2.5:7b
  ollama serve
  ```

### Status: ✅ Fully implemented

---

## 6. 🍓 Pi Camera Module 3 Integration

### Files
- [`camera_stream.py`](camera_stream.py) — **Run this on Raspberry Pi**
- [`deploy_to_pi.py`](deploy_to_pi.py) — SSH deploy helper

### Stream URL
Hardcoded in `app.js` line 92:
```js
const PI_STREAM_URL = 'http://10.203.139.112:5000/video';
```

### ⚠️ PORT MISMATCH
`camera_stream.py` serves on **PORT 5001**, but `app.js` connects to **PORT 5000**.  
Either update `app.js` to `5001`, or change `PORT` in `camera_stream.py` to `5000`.

### Stream endpoint
`camera_stream.py` exposes `/stream` (MJPEG), but `app.js` connects to `/video`.  
The URL in `app.js` should be `http://10.203.139.112:5001/stream`.

---

## 🔁 Main App Flow (`app.js`)

```
DOMContentLoaded
  → startBtn click
    → startCamera() [laptop webcam or Pi MJPEG]
      → initModels() [FaceLandmarker + HandLandmarker from Google CDN]
        → requestAnimationFrame(detectLoop)

detectLoop() — every frame:
  ├── if mode=emotion  → faceLandmarker.detectForVideo() → EmotionAnalyzer.analyze()
  ├── if mode=focus    → faceLandmarker.detectForVideo() → FocusAnalyzer.analyze()
  ├── if mode=sign     → handLandmarker.detectForVideo() → SignLanguageAnalyzer.analyze()
  ├── if mode=obstacle → NeuroLensAnalyzer.analyze() [async, COCO-SSD]
  └── always           → VisualQualityAnalyzer.analyze() + JSON output update
```

---

## ⚠️ Issues & Notes

| # | Severity | Issue | Location |
|---|---|---|---|
| 1 | 🟡 Medium | **Pi Camera port mismatch**: app expects port 5000, server runs on 5001 | `app.js` L92 vs `camera_stream.py` L56 |
| 2 | 🟡 Medium | **Pi Camera endpoint mismatch**: app calls `/video`, server exposes `/stream` | `app.js` L92 vs `camera_stream.py` L159 |
| 3 | 🟢 Low | `dynamic-classifier.js` is imported nowhere — dead code | `js/sign/dynamic-classifier.js` |
| 4 | 🟢 Low | `sentence-builder.js` is imported nowhere — unused module | `js/sign/sentence-builder.js` |
| 5 | 🟢 Low | `yolov4-tiny.weights` exists in both `/models/` AND `/neurolens/models/` — duplicate | Both paths |
| 6 | 🟢 Low | `DOM.scanLine` referenced in `app.js` L28 but no `id="scanLine"` element in `index.html` | `app.js` L28 |

---

## 🧩 External Dependencies (Browser)

| Library | Source | Used For |
|---|---|---|
| `@mediapipe/tasks-vision@0.10.18` | jsDelivr CDN | FaceLandmarker + HandLandmarker |
| `@tensorflow/tfjs@4.17.0` | jsDelivr CDN | COCO-SSD runtime (lazy loaded) |
| `@tensorflow-models/coco-ssd@2.2.3` | jsDelivr CDN | Obstacle detection (lazy loaded) |
| `tesseract.js@5` | jsDelivr CDN | OCR text extraction (lazy loaded) |
| Ollama (`qwen2.5:7b`) | localhost:11434 | Text summarization (local LLM) |

---

## 🐍 Python Dependencies (Pi-side)

```
opencv-python>=4.8.0
numpy>=1.24.0
flask (manual install)
flask-cors (manual install)
picamera2 (apt install python3-picamera2)
libcamera (apt install python3-libcamera)
```

---

## 🚀 How to Run

### Browser app (laptop)
Just open `index.html` in a browser (Chrome recommended).  
A local HTTP server may be needed due to ES module imports:
```bash
python -m http.server 8080
# Then open http://localhost:8080
```

### Python obstacle detection (standalone)
```bash
pip install opencv-python numpy
python -m neurolens.main
```

### Pi Camera stream (on Raspberry Pi)
```bash
sudo apt install -y python3-picamera2 python3-libcamera python3-opencv
pip install flask flask-cors numpy
python camera_stream.py
```
Then in `app.js`, fix the URL to: `http://<pi-ip>:5001/stream`
