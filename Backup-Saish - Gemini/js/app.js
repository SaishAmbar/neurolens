/**
 * NeuroSense AI — Main Application
 * Orchestrates webcam capture, MediaPipe models, analysis modules, and UI rendering.
 */

import { FaceLandmarker, HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { EmotionAnalyzer } from './emotion-analyzer.js?v=2';
import { FocusAnalyzer } from './focus-analyzer.js';
import { SignLanguageAnalyzer } from './sign-language-analyzer.js';
import { VisualQualityAnalyzer } from './visual-quality.js';
import { NeuroLensAnalyzer } from './neurolens-analyzer.js';
import { TextScanner } from './text-scanner.js';

// ── DOM Elements ──
const $ = id => document.getElementById(id);

const DOM = {
  webcam: $('webcam'),
  canvas: $('overlayCanvas'),
  loadingOverlay: $('loadingOverlay'),
  loadingText: $('loadingText'),
  permissionScreen: $('permissionScreen'),
  startBtn: $('startBtn'),
  errorMsg: $('errorMsg'),
  statusDot: $('statusDot'),
  statusText: $('statusText'),
  fpsCounter: $('fpsCounter'),
  resolution: $('resolution'),
  scanLine: $('scanLine'),
  // Emotion
  emotionPrimary: $('emotionPrimary'),
  emotionSecondary: $('emotionSecondary'),
  emotionIntensity: $('emotionIntensity'),
  emotionConfBar: $('emotionConfBar'),
  emotionConfVal: $('emotionConfVal'),
  emotionEvidence: $('emotionEvidence'),
  emotionBadge: $('emotionBadge'),
  // Focus
  focusState: $('focusState'),
  focusConfBar: $('focusConfBar'),
  focusConfVal: $('focusConfVal'),
  focusEvidence: $('focusEvidence'),
  focusBadge: $('focusBadge'),
  // Sign Language (sentence translation)
  gestureDisplay: $('gestureDisplay'),
  signInterpretation: $('signInterpretation'),
  signConfBar: $('signConfBar'),
  signConfVal: $('signConfVal'),
  wordConfBar: $('wordConfBar'),
  wordConfVal: $('wordConfVal'),
  sentenceConfBar: $('sentenceConfBar'),
  sentenceConfVal: $('sentenceConfVal'),
  signAlternatives: $('signAlternatives'),
  signBadge: $('signBadge'),
  signStatusBadge: $('signStatusBadge'),
  signFormingRow: $('signFormingRow'),
  signFormingText: $('signFormingText'),
  signCurrentSentence: $('signCurrentSentence'),
  signTranscriptHistory: $('signTranscriptHistory'),
  signResetBtn: $('signResetBtn'),
  signUndoBtn: $('signUndoBtn'),
  signToggleInput: $('signToggleInput'),
  signToggleLabel: $('signToggleLabel'),
  // NeuroLens Obstacle
  obstacleBadge: $('obstacleBadge'),
  obstaclePanel: $('obstaclePanel'),
  obstacleEvidence: $('obstacleEvidence'),
  nlAlertZone: $('nlAlertZone'),
  nlAlertIcon: $('nlAlertIcon'),
  nlAlertLabel: $('nlAlertLabel'),
  nlAlertSub: $('nlAlertSub'),
  nlThreatCard: $('nlThreatCard'),
  nlThreatName: $('nlThreatName'),
  nlThreatDist: $('nlThreatDist'),
  nlThreatDir: $('nlThreatDir'),
  nlThreatDanger: $('nlThreatDanger'),
  nlDetectionCount: $('nlDetectionCount'),
  nlDetectionsList: $('nlDetectionsList'),
  // Quality
  qLighting: $('qLighting'),
  qOcclusion: $('qOcclusion'),
  qClarity: $('qClarity'),
  qualityBadge: $('qualityBadge'),
  // JSON
  jsonToggle: $('jsonToggle'),
  jsonOutput: $('jsonOutput'),
  jsonPre: $('jsonPre'),
  // Mode tabs
  modeTabs: $('modeTabs'),
  // Text Scanner
  textScannerPanel: $('textScannerPanel'),
  textScannerBadge: $('textScannerBadge'),
  tsFrozenCanvas: $('tsFrozenCanvas'),
  tsFramePlaceholder: $('tsFramePlaceholder'),
  tsScanBtn: $('tsScanBtn'),
  tsClearBtn: $('tsClearBtn'),
  tsProgressRow: $('tsProgressRow'),
  tsProgressBar: $('tsProgressBar'),
  tsProgressVal: $('tsProgressVal'),
  tsRawText: $('tsRawText'),
  tsCharCount: $('tsCharCount'),
  tsSummaryBox: $('tsSummaryBox'),
  tsModelLabel: $('tsModelLabel'),
  tsError: $('tsError'),
  tsEvidence: $('tsEvidence'),
};

// ── Pi Camera Config ──
const PI_STREAM_URL = 'http://10.221.228.112:5001/stream';

// ── OLED Display Config ──
const PI_OLED_URL = 'http://10.221.228.112:5002/oled';
let _oledLastSent = 0;

/**
 * Sends display data to the SSD1306 OLED via the Pi Flask server.
 * Throttled to once per 500ms. Silently ignores errors (Pi may be offline).
 * @param {string} mode  - 'emotion'|'focus'|'sign'|'obstacle'|'textscanner'
 * @param {object|null} data - { line1, line2, scroll } or null to clear screen
 */
function sendToOLED(mode, data) {
  const now = Date.now();
  if (now - _oledLastSent < 500) return;
  _oledLastSent = now;

  const body = data
    ? { mode, line1: data.line1 ?? '', line2: data.line2 ?? '', scroll: !!data.scroll }
    : { mode, line1: '', line2: '', scroll: false };

  fetch(PI_OLED_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).catch(() => { /* Pi offline — silently ignore */ });
}

// ── State ──
let faceLandmarker = null;
let handLandmarker = null;
let animFrameId = null;
let lastFrameTime = 0;
let frameCount = 0;
let fpsTime = 0;
let currentFps = 0;
let currentMode = 'emotion';
let neurolensLoading = false;
let signDetectionEnabled = true;
let usePiCamera = false;      // false = laptop webcam, true = Pi Camera Module 3
let piImgEl = null;           // <img> fed by MJPEG stream
let piFrameCanvas = null;     // offscreen canvas bridging MJPEG → video
let piFrameCtx = null;
let piFrameLoopId = null;

const emotionAnalyzer = new EmotionAnalyzer();
const focusAnalyzer = new FocusAnalyzer();
const signAnalyzer = new SignLanguageAnalyzer();
const qualityAnalyzer = new VisualQualityAnalyzer();
const neurolensAnalyzer = new NeuroLensAnalyzer();
const textScanner = new TextScanner();
let textScannerLoading = false;

// Canvas for pixel analysis (offscreen)
const analysisCanvas = document.createElement('canvas');
const analysisCtx = analysisCanvas.getContext('2d', { willReadFrequently: true });


// ── Mode Switching ──
function switchMode(mode) {
  currentMode = mode;
  document.querySelectorAll('.mode-tab').forEach(tab => {
    tab.classList.toggle('active', tab.dataset.mode === mode);
  });
  document.querySelectorAll('.mode-panel').forEach(panel => {
    if (panel.dataset.mode === mode) {
      panel.style.display = '';
      panel.style.animation = 'none';
      panel.offsetHeight;
      panel.style.animation = '';
      // Focus Highlight Mode: mark active panel
      panel.classList.add('focus-active');
    } else {
      panel.style.display = 'none';
      panel.classList.remove('focus-active');
    }
  });
  const modeNames = { emotion: 'Analyzing emotions…', focus: 'Tracking attention…', sign: 'Reading signs…', neurolens: 'Scanning surroundings…', textscanner: 'Text scanner ready…' };
  if (DOM.statusDot.classList.contains('active')) {
    DOM.statusText.textContent = modeNames[mode] || 'Observing…';
  }
  // Lazy-load NeuroLens model when tab is first selected
  if (mode === 'neurolens' && !neurolensAnalyzer.isLoaded && !neurolensLoading) {
    neurolensLoading = true;
    DOM.nlAlertSub.textContent = 'Warming up obstacle model…';
    neurolensAnalyzer.loadModel().then(() => {
      neurolensLoading = false;
      DOM.nlAlertSub.textContent = 'Model ready — scanning…';
      if (DOM.obstacleBadge) {
        DOM.obstacleBadge.className = 'panel-badge active';
        DOM.obstacleBadge.textContent = 'LIVE';
      }
    }).catch(err => {
      neurolensLoading = false;
      DOM.nlAlertSub.textContent = `Model load failed: ${err.message}`;
    });
  }
  // Lazy-load Tesseract worker when Text Scanner tab is first selected
  if (mode === 'textscanner' && !textScanner.isReady && !textScannerLoading) {
    textScannerLoading = true;
    setTSBadge('scanning', '⚙️ LOADING');
    textScanner.initWorker().then(() => {
      textScannerLoading = false;
      setTSBadge('active', 'READY');
    }).catch(err => {
      textScannerLoading = false;
      setTSBadge('error', 'ERROR');
      DOM.tsError.textContent = `Tesseract load failed: ${err.message}`;
      DOM.tsError.style.display = '';
    });
  }
}

DOM.modeTabs.addEventListener('click', (e) => {
  const tab = e.target.closest('.mode-tab');
  if (tab && tab.dataset.mode) {
    switchMode(tab.dataset.mode);
  }
});

// ── JSON Toggle ──
DOM.jsonToggle.addEventListener('click', () => {
  DOM.jsonToggle.classList.toggle('open');
  DOM.jsonOutput.classList.toggle('open');
});

// ── Sign Language Reset Button ──
DOM.signResetBtn.addEventListener('click', () => {
  signAnalyzer.reset();
  updateSignUI(signAnalyzer.analyze(null, null, performance.now()));
});

// ── Sign Language Undo Button ──
DOM.signUndoBtn.addEventListener('click', () => {
  signAnalyzer.undoLastWord();
  updateSignUI(signAnalyzer.analyze(null, null, performance.now()));
});

// ── Sign Language Toggle ──
DOM.signToggleInput.addEventListener('change', (e) => {
  signDetectionEnabled = e.target.checked;
  DOM.signToggleLabel.textContent = signDetectionEnabled ? 'ON' : 'OFF';
});

// ── Text Scanner Buttons ──
let tsProgressPoll = null;

function setTSBadge(cls, text) {
  DOM.textScannerBadge.className = `panel-badge ${cls}`;
  DOM.textScannerBadge.textContent = text;
}

function resetTextScannerUI() {
  textScanner.reset();
  DOM.tsFrozenCanvas.classList.remove('visible');
  DOM.tsFramePlaceholder.classList.remove('hidden');
  DOM.tsRawText.innerHTML = '<span class="ts-placeholder-text">No text extracted yet…</span>';
  DOM.tsRawText.classList.remove('has-content');
  DOM.tsSummaryBox.innerHTML = '<span class="ts-placeholder-text">Summary will appear here…</span>';
  DOM.tsSummaryBox.classList.remove('has-content', 'streaming');
  DOM.tsCharCount.textContent = '';
  DOM.tsProgressRow.style.display = 'none';
  DOM.tsProgressBar.style.width = '0%';
  DOM.tsProgressVal.textContent = '0%';
  DOM.tsError.style.display = 'none';
  DOM.tsScanBtn.disabled = false;
  DOM.tsScanBtn.textContent = '🔍 Scan';
  DOM.tsScanBtn.classList.remove('scanning');
  DOM.tsClearBtn.disabled = true;
  setTSBadge(textScanner.isReady ? 'active' : 'inactive', textScanner.isReady ? 'READY' : 'IDLE');
  if (tsProgressPoll) { clearInterval(tsProgressPoll); tsProgressPoll = null; }
}

function drawWordHighlights(canvas, words, sourceCanvas) {
  // Copy frozen frame then overlay word bounding boxes
  const ctx = canvas.getContext('2d');
  canvas.width = sourceCanvas.width;
  canvas.height = sourceCanvas.height;
  ctx.drawImage(sourceCanvas, 0, 0);

  // Draw semi-transparent highlights for each word
  for (const w of words) {
    if (w.confidence < 40) continue;
    const { x0, y0, x1, y1 } = w.bbox;
    // Highlight background
    ctx.fillStyle = 'rgba(139, 203, 216, 0.18)';
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    // Border
    ctx.strokeStyle = 'rgba(139, 203, 216, 0.7)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  }
}

DOM.tsScanBtn.addEventListener('click', async () => {
  if (textScanner.state === 'scanning' || textScanner.state === 'summarizing') return;

  DOM.tsScanBtn.disabled = true;
  DOM.tsScanBtn.textContent = '⏳ Scanning…';
  DOM.tsScanBtn.classList.add('scanning');
  DOM.tsClearBtn.disabled = true;
  DOM.tsError.style.display = 'none';
  setTSBadge('scanning', '🔍 SCANNING');

  // Show progress bar
  DOM.tsProgressRow.style.display = 'flex';
  DOM.tsProgressBar.style.width = '0%';
  DOM.tsProgressVal.textContent = '0%';

  // Poll for OCR progress
  tsProgressPoll = setInterval(() => {
    const p = textScanner.progress;
    DOM.tsProgressBar.style.width = `${p}%`;
    DOM.tsProgressVal.textContent = `${p}%`;
  }, 200);

  try {
    // Step 1: OCR
    const { canvas: frozenCanvas, words, text } = await textScanner.scan(DOM.webcam);

    clearInterval(tsProgressPoll);
    tsProgressPoll = null;
    DOM.tsProgressBar.style.width = '100%';
    DOM.tsProgressVal.textContent = '100%';

    // Show frozen frame with highlights
    if (words.length > 0) {
      drawWordHighlights(DOM.tsFrozenCanvas, words, frozenCanvas);
    } else {
      const ctx = DOM.tsFrozenCanvas.getContext('2d');
      DOM.tsFrozenCanvas.width = frozenCanvas.width;
      DOM.tsFrozenCanvas.height = frozenCanvas.height;
      ctx.drawImage(frozenCanvas, 0, 0);
    }
    DOM.tsFrozenCanvas.classList.add('visible');
    DOM.tsFramePlaceholder.classList.add('hidden');

    // Show extracted text
    if (text) {
      DOM.tsRawText.textContent = text;
      DOM.tsRawText.classList.add('has-content');
      DOM.tsCharCount.textContent = `${text.length} chars`;
    } else {
      DOM.tsRawText.innerHTML = '<span class="ts-placeholder-text">No readable text detected</span>';
      DOM.tsRawText.classList.remove('has-content');
      DOM.tsCharCount.textContent = '';
    }

    // Show error if no text
    if (textScanner.error) {
      DOM.tsError.textContent = textScanner.error;
      DOM.tsError.style.display = '';
      DOM.tsScanBtn.disabled = false;
      DOM.tsScanBtn.textContent = '🔍 Rescan';
      DOM.tsScanBtn.classList.remove('scanning');
      DOM.tsClearBtn.disabled = false;
      setTSBadge('done', '✓ DONE');
      return;
    }

    // Step 2: Summarize with Ollama
    setTSBadge('summarizing', '🧠 THINKING');
    DOM.tsScanBtn.textContent = '🧠 Summarizing…';
    DOM.tsSummaryBox.innerHTML = '<span class="ts-cursor"></span>';
    DOM.tsSummaryBox.classList.add('streaming');
    DOM.tsSummaryBox.classList.remove('has-content');

    await textScanner.summarize(text, (partial) => {
      DOM.tsSummaryBox.innerHTML = partial + '<span class="ts-cursor"></span>';
      DOM.tsSummaryBox.scrollTop = DOM.tsSummaryBox.scrollHeight;
    });

    // Finalize
    DOM.tsSummaryBox.classList.remove('streaming');
    if (textScanner.summary) {
      DOM.tsSummaryBox.textContent = textScanner.summary;
      DOM.tsSummaryBox.classList.add('has-content');
      sendToOLED('textscanner', {
        line1: 'SUMMARY:',
        line2: textScanner.summary,
        scroll: true,
      });
    } else {
      DOM.tsSummaryBox.innerHTML = '<span class="ts-placeholder-text">No summary generated</span>';
    }

    if (textScanner.error && textScanner.state === 'error') {
      DOM.tsError.textContent = textScanner.error;
      DOM.tsError.style.display = '';
      setTSBadge('error', '⚠ ERROR');
    } else {
      setTSBadge('done', '✓ DONE');
    }

  } catch (err) {
    DOM.tsError.textContent = `Unexpected error: ${err.message}`;
    DOM.tsError.style.display = '';
    setTSBadge('error', '⚠ ERROR');
  } finally {
    if (tsProgressPoll) { clearInterval(tsProgressPoll); tsProgressPoll = null; }
    DOM.tsScanBtn.disabled = false;
    DOM.tsScanBtn.textContent = '🔍 Rescan';
    DOM.tsScanBtn.classList.remove('scanning');
    DOM.tsClearBtn.disabled = false;
    setTimeout(() => { DOM.tsProgressRow.style.display = 'none'; }, 1500);
  }
});

DOM.tsClearBtn.addEventListener('click', () => {
  resetTextScannerUI();
});

// ── Camera Source Toggle ──
const btnLaptop = document.getElementById('btnLaptopCam');
const btnPi = document.getElementById('btnPiCam');

function stopCurrentStream() {
  // FIX: Stop all hardware tracks FIRST so the laptop webcam fully releases
  // before we null srcObject. Without this, the canvas captureStream from the
  // previous mode can keep the underlying getUserMedia alive and "bleed through".
  if (DOM.webcam.srcObject) {
    const stream = DOM.webcam.srcObject;
    DOM.webcam.pause();
    DOM.webcam.srcObject = null;   // detach before stopping to avoid a race
    if (stream && typeof stream.getTracks === 'function') {
      stream.getTracks().forEach(track => track.stop());
    }
  }
  // FIX: Also clear the src attribute — belt-and-suspenders against bleed-through.
  DOM.webcam.removeAttribute('src');
  DOM.webcam.load();               // resets the media element state machine

  // Cancel the Pi MJPEG draw loop
  if (piFrameLoopId) {
    cancelAnimationFrame(piFrameLoopId);
    piFrameLoopId = null;
  }
  // FIX: Abort the MJPEG img load so the browser stops downloading stream data.
  // Setting src='' alone may not abort an in-flight multipart response.
  if (piImgEl) {
    piImgEl.onload = null;
    piImgEl.onerror = null;
    piImgEl.src = 'about:blank';   // forces connection teardown in all browsers
    piImgEl = null;
  }
  // Wipe the offscreen Pi canvas so stale frames can't be read
  if (piFrameCtx && piFrameCanvas) {
    piFrameCtx.clearRect(0, 0, piFrameCanvas.width, piFrameCanvas.height);
  }
  piFrameCanvas = null;
  piFrameCtx = null;
}
async function handleCameraSwitch(targetIsPi) {
  if (usePiCamera === targetIsPi) return;
  usePiCamera = targetIsPi;

  if (usePiCamera) {
    btnPi.classList.add('active', 'pi');
    btnLaptop.classList.remove('active');
    DOM.webcam.style.transform = 'scaleX(1)';
    DOM.canvas.style.transform = 'scaleX(1)';
  } else {
    btnLaptop.classList.add('active');
    btnLaptop.classList.remove('pi');
    btnPi.classList.remove('active', 'pi');
    DOM.webcam.style.transform = 'scaleX(-1)';
    DOM.canvas.style.transform = 'scaleX(-1)';
  }
  if (!DOM.statusDot.classList.contains('active')) return;
  try {
    DOM.statusText.textContent = '🔄 Switching camera source...';
    stopCurrentStream();

    if (usePiCamera) {
      await startPiCameraStream();
    } else {
      await startLaptopCameraStream();
    }

    const vw = usePiCamera ? piFrameCanvas.width : DOM.webcam.videoWidth;
    const vh = usePiCamera ? piFrameCanvas.height : DOM.webcam.videoHeight;
    DOM.canvas.width = vw;
    DOM.canvas.height = vh;
    analysisCanvas.width = vw;
    analysisCanvas.height = vh;

    DOM.resolution.textContent = `${vw}×${vh}${usePiCamera ? ' (Pi Cam)' : ''}`;
    DOM.statusText.textContent = usePiCamera ? '📡 Pi Camera Active' : 'Analyzing emotions…';
  } catch (err) {
    console.error("Camera source swap failed:", err);
    DOM.statusText.textContent = `❌ Switch Error: ${err.message}`;
  }
}
btnLaptop.addEventListener('click', () => handleCameraSwitch(false));
btnPi.addEventListener('click', () => handleCameraSwitch(true));

// ── Start Button ──
DOM.startBtn.addEventListener('click', async () => {
  DOM.startBtn.disabled = true;
  DOM.startBtn.textContent = '⏳ Starting…';
  DOM.errorMsg.style.display = 'none';

  try {
    await startCamera();
  } catch (err) {
    DOM.errorMsg.textContent = `Camera error: ${err.message}`;
    DOM.errorMsg.style.display = 'block';
    DOM.startBtn.disabled = false;
    DOM.startBtn.textContent = '🔄 Retry';
  }
});

// ── Camera Setup ──
async function startCamera() {
  if (usePiCamera) {
    await startPiCameraStream();
  } else {
    await startLaptopCameraStream();
  }

  // Set canvas sizes — for Pi cam, startPiCameraStream already set DOM.canvas dimensions
  const vw = usePiCamera ? piFrameCanvas.width : DOM.webcam.videoWidth;
  const vh = usePiCamera ? piFrameCanvas.height : DOM.webcam.videoHeight;
  DOM.canvas.width = vw;
  DOM.canvas.height = vh;
  analysisCanvas.width = vw;
  analysisCanvas.height = vh;
  DOM.resolution.textContent = `${vw}×${vh}${usePiCamera ? ' (Pi Cam)' : ''}`;

  // Hide permission screen, show loading
  DOM.permissionScreen.classList.add('hidden');
  DOM.loadingText.innerHTML = 'Loading <span>MediaPipe</span> models…';

  await initModels();

  DOM.loadingOverlay.classList.add('hidden');
  DOM.statusDot.classList.add('active');
  DOM.statusText.textContent = usePiCamera ? '📡 Pi Camera Active' : 'Analyzing emotions…';

  // Activate badges
  for (const badge of [DOM.emotionBadge, DOM.focusBadge, DOM.signBadge, DOM.obstacleBadge, DOM.qualityBadge]) {
    if (badge) {
      badge.className = 'panel-badge active';
      badge.textContent = 'LIVE';
    }
  }

  // Apply initial mode
  switchMode(currentMode);

  // Start detection loop
  requestAnimationFrame(detectLoop);
}

// ── Laptop webcam ──
async function startLaptopCameraStream() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
    audio: false,
  });
  DOM.webcam.srcObject = stream;
  await new Promise(r => { DOM.webcam.onloadedmetadata = r; });
  await DOM.webcam.play();
}

// ── Pi Camera Module 3 via MJPEG → canvas → captureStream ──
async function startPiCameraStream() {
  const PI_HOST = PI_STREAM_URL.replace('/stream', '');

  // Step 1: Verify the Pi server is reachable.
  // We probe the root URL — ANY response (even 404) proves the server is up.
  // Only a network error (unreachable, refused, timeout) should block us.
  try {
    await Promise.race([
      fetch(PI_HOST, { mode: 'cors' }).then(res => {
        // Any HTTP response means the Pi Flask server is running
        console.log(`📡 Pi server reachable (HTTP ${res.status})`);
      }),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Pi Camera timed out (8s). Is camera_stream.py running?')), 8000)
      )
    ]);
  } catch (err) {
    // "Failed to fetch" = CORS block OR network error.
    // For CORS blocks the Pi IS reachable, so try proceeding with the MJPEG
    // stream anyway — <img> tags bypass CORS for loading (only drawImage
    // might fail if the server doesn't send Access-Control headers).
    if (err.message.includes('Failed to fetch')) {
      console.warn('⚠️ Pi reachability check got CORS error — Pi may still be reachable, attempting MJPEG stream...');
    } else {
      throw new Error(`Cannot reach Pi Camera at ${PI_STREAM_URL} — ${err.message}`);
    }
  }

  // Step 2: Set up offscreen canvas to hold MJPEG frames
  piFrameCanvas = document.createElement('canvas');
  piFrameCanvas.width = 320;
  piFrameCanvas.height = 240;
  piFrameCtx = piFrameCanvas.getContext('2d');

  // Step 3: Create <img> for the MJPEG stream
  piImgEl = new Image();
  piImgEl.crossOrigin = 'anonymous';

  // Track frame state
  let firstFrameReceived = false;
  let drawErrorLogged = false;
  let lastGoodFrame = 0;

  // Error handler — surfaces connection failures instead of silent swallow
  piImgEl.onerror = (e) => {
    console.error('❌ Pi MJPEG <img> error — stream failed to load.', e);
    DOM.statusText.textContent = '❌ Pi Camera stream failed to load';
  };

  piImgEl.src = PI_STREAM_URL;

  // Step 4: Continuously draw MJPEG frames into the offscreen canvas
  (function drawPiFrame() {
    if (!piImgEl) return;  // cleaned up by stopCurrentStream

    try {
      if (piImgEl.naturalWidth > 0) {
        piFrameCtx.drawImage(piImgEl, 0, 0, piFrameCanvas.width, piFrameCanvas.height);
        if (!firstFrameReceived) {
          firstFrameReceived = true;
          console.log('✅ Pi Camera: first MJPEG frame drawn to canvas');
        }
        lastGoodFrame = performance.now();
      }
    } catch (e) {
      // Log CORS / tainted canvas errors ONCE
      if (!drawErrorLogged) {
        drawErrorLogged = true;
        console.error('❌ Pi Camera drawImage failed (CORS/tainted canvas):', e.message);
        DOM.statusText.textContent = '❌ Pi stream CORS error — update camera_stream.py on Pi';
      }
    }

    // Warn if stream stalls (had frames but stopped for 5s)
    if (firstFrameReceived && (performance.now() - lastGoodFrame > 5000)) {
      DOM.statusText.textContent = '⚠️ Pi Camera stream stalled';
      console.warn('⚠️ Pi Camera: no new frames for 5s');
      lastGoodFrame = performance.now();
    }

    piFrameLoopId = requestAnimationFrame(drawPiFrame);
  })();

  // Step 5: Wait for first real frame before wiring up captureStream.
  // Without this, MediaPipe sees blank frames and the old laptop feed
  // appears to "bleed through".
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      if (!firstFrameReceived) {
        reject(new Error('Pi Camera: no MJPEG frames received in 10s. Check Pi network, CORS, and camera_stream.py.'));
      }
    }, 10000);

    (function waitForFrame() {
      if (firstFrameReceived) { clearTimeout(timeout); resolve(); return; }
      if (!piImgEl) { clearTimeout(timeout); reject(new Error('Pi Camera stream stopped')); return; }
      setTimeout(waitForFrame, 100);
    })();
  });

  // Step 6: Feed canvas as MediaStream into <video> for MediaPipe
  const canvasStream = piFrameCanvas.captureStream(15);
  DOM.webcam.srcObject = canvasStream;
  DOM.canvas.width = 320;
  DOM.canvas.height = 240;
  await DOM.webcam.play();

  console.log('📡 Pi Camera stream active and feeding into <video>');
}

// ── Model Initialization ──
async function initModels() {
  DOM.loadingText.innerHTML = 'Warming up <span>vision engine</span>…';
  const filesetResolver = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
  );

  DOM.loadingText.innerHTML = 'Preparing <span>face recognition</span>…';
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  });

  DOM.loadingText.innerHTML = 'Preparing <span>hand tracking</span>…';
  handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numHands: 2,
  });
}

// ── Main Detection Loop ──
function detectLoop(timestamp) {
  if (!faceLandmarker || !handLandmarker) {
    animFrameId = requestAnimationFrame(detectLoop);
    return;
  }

  // Avoid duplicate timestamps
  if (timestamp === lastFrameTime) {
    animFrameId = requestAnimationFrame(detectLoop);
    return;
  }
  lastFrameTime = timestamp;

  // FPS calculation
  frameCount++;
  if (timestamp - fpsTime >= 1000) {
    currentFps = frameCount;
    frameCount = 0;
    fpsTime = timestamp;
    DOM.fpsCounter.textContent = `${currentFps} FPS`;
  }

  // Draw landmarks on overlay canvas
  const ctx = DOM.canvas.getContext('2d');
  ctx.clearRect(0, 0, DOM.canvas.width, DOM.canvas.height);

  // Draw frame to analysis canvas for pixel analysis
  analysisCtx.drawImage(DOM.webcam, 0, 0);

  // Only run the models needed for the active mode
  const needFace = currentMode === 'emotion' || currentMode === 'focus';
  const needHand = currentMode === 'sign' && signDetectionEnabled;
  const needNeuroLens = currentMode === 'neurolens';

  let faceResults = null;
  let handResults = null;

  if (needFace) {
    faceResults = faceLandmarker.detectForVideo(DOM.webcam, timestamp);
  }
  if (needHand) {
    handResults = handLandmarker.detectForVideo(DOM.webcam, timestamp);
  }

  // ── Results per mode ──
  let emotionResult = null;
  let focusResult = null;
  let signResult = null;
  let neurolensResult = null;

  if (currentMode === 'emotion') {
    if (faceResults?.faceLandmarks?.length > 0) {
      const landmarks = faceResults.faceLandmarks[0];
      const blendshapes = faceResults.faceBlendshapes?.[0]?.categories || [];
      drawFaceMesh(ctx, landmarks);
      emotionResult = emotionAnalyzer.analyze(blendshapes);
    } else {
      emotionResult = emotionAnalyzer.analyze(null);
    }
    updateEmotionUI(emotionResult);
  }

  if (currentMode === 'focus') {
    if (faceResults?.faceLandmarks?.length > 0) {
      const landmarks = faceResults.faceLandmarks[0];
      const blendshapes = faceResults.faceBlendshapes?.[0]?.categories || [];
      drawFaceMesh(ctx, landmarks);
      focusResult = focusAnalyzer.analyze(blendshapes, landmarks, timestamp);
    } else {
      focusResult = focusAnalyzer.analyze(null, null, timestamp);
    }
    updateFocusUI(focusResult);
  }

  if (currentMode === 'sign' && signDetectionEnabled) {
    if (handResults?.landmarks?.length > 0) {
      for (const handLandmarks of handResults.landmarks) {
        drawHandSkeleton(ctx, handLandmarks);
      }
      signResult = signAnalyzer.analyze(handResults.landmarks, handResults.handednesses, timestamp);
    } else {
      signResult = signAnalyzer.analyze(null, null, timestamp);
    }
    updateSignUI(signResult);
  }

  if (needNeuroLens && neurolensAnalyzer.isLoaded) {
    // NeuroLens uses async detection — fire and continue
    neurolensAnalyzer.analyze(DOM.webcam, ctx, DOM.canvas.width, DOM.canvas.height)
      .then(result => {
        neurolensResult = result;
        updateNeuroLensUI(result);
      });
  }

  // ── Visual Quality (always runs) ──
  const faceLandmarksForQuality = faceResults?.faceLandmarks?.[0] || null;
  const qualityResult = qualityAnalyzer.analyze(analysisCanvas, faceLandmarksForQuality, faceResults);
  updateQualityUI(qualityResult);

  // ── JSON ──
  updateJSON(emotionResult, focusResult, signResult, qualityResult, neurolensResult);

  animFrameId = requestAnimationFrame(detectLoop);
}

// ── Drawing Functions ──

function drawFaceMesh(ctx, landmarks) {
  // Draw tessellation dots
  ctx.fillStyle = 'rgba(0, 229, 255, 0.25)';
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    const x = lm.x * DOM.canvas.width;
    const y = lm.y * DOM.canvas.height;
    ctx.beginPath();
    ctx.arc(x, y, 0.8, 0, 2 * Math.PI);
    ctx.fill();
  }

  // Draw key contours (eyes, lips, face oval) with brighter lines
  drawContour(ctx, landmarks, FACE_OVAL, 'rgba(0, 229, 255, 0.15)', 1);
  drawContour(ctx, landmarks, LEFT_EYE, 'rgba(105, 240, 174, 0.5)', 1.2);
  drawContour(ctx, landmarks, RIGHT_EYE, 'rgba(105, 240, 174, 0.5)', 1.2);
  drawContour(ctx, landmarks, LIPS_OUTER, 'rgba(255, 171, 64, 0.4)', 1.2);
  drawContour(ctx, landmarks, LEFT_EYEBROW, 'rgba(179, 136, 255, 0.4)', 1);
  drawContour(ctx, landmarks, RIGHT_EYEBROW, 'rgba(179, 136, 255, 0.4)', 1);

  // Draw iris landmarks (468-477)
  ctx.fillStyle = 'rgba(0, 229, 255, 0.9)';
  for (let i = 468; i < Math.min(478, landmarks.length); i++) {
    const lm = landmarks[i];
    const x = lm.x * DOM.canvas.width;
    const y = lm.y * DOM.canvas.height;
    ctx.beginPath();
    ctx.arc(x, y, 1.5, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function drawContour(ctx, landmarks, indices, color, width) {
  if (!indices || indices.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  const first = landmarks[indices[0]];
  ctx.moveTo(first.x * DOM.canvas.width, first.y * DOM.canvas.height);
  for (let i = 1; i < indices.length; i++) {
    const lm = landmarks[indices[i]];
    ctx.lineTo(lm.x * DOM.canvas.width, lm.y * DOM.canvas.height);
  }
  ctx.closePath();
  ctx.stroke();
}

function drawHandSkeleton(ctx, landmarks) {
  const w = DOM.canvas.width;
  const h = DOM.canvas.height;

  // Draw connections
  const connections = [
    [0, 1], [1, 2], [2, 3], [3, 4],       // thumb
    [0, 5], [5, 6], [6, 7], [7, 8],       // index
    [0, 9], [9, 10], [10, 11], [11, 12],  // middle
    [0, 13], [13, 14], [14, 15], [15, 16],// ring
    [0, 17], [17, 18], [18, 19], [19, 20],// pinky
    [5, 9], [9, 13], [13, 17],          // palm
  ];

  ctx.strokeStyle = 'rgba(105, 240, 174, 0.6)';
  ctx.lineWidth = 2;
  for (const [a, b] of connections) {
    ctx.beginPath();
    ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
    ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
    ctx.stroke();
  }

  // Draw joints
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    const x = lm.x * w;
    const y = lm.y * h;
    const isTip = [4, 8, 12, 16, 20].includes(i);

    ctx.beginPath();
    ctx.arc(x, y, isTip ? 4 : 2.5, 0, 2 * Math.PI);
    ctx.fillStyle = isTip ? 'rgba(105, 240, 174, 0.95)' : 'rgba(105, 240, 174, 0.6)';
    ctx.fill();

    if (isTip) {
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, 2 * Math.PI);
      ctx.strokeStyle = 'rgba(105, 240, 174, 0.25)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }
}

// ── UI Update Functions ──

function updateEmotionUI(result) {
  DOM.emotionPrimary.textContent = result.primary;
  DOM.emotionSecondary.textContent = result.secondary || '—';

  const intensity = result.intensity || '—';
  DOM.emotionIntensity.textContent = intensity;
  DOM.emotionIntensity.className = `intensity-badge ${intensity.toLowerCase()}`;

  const confNum = parseInt(result.confidence) || 0;
  DOM.emotionConfBar.style.width = `${confNum}%`;
  DOM.emotionConfVal.textContent = result.confidence;
  DOM.emotionEvidence.textContent = result.evidence;

  // Color the primary value by emotion
  const colorMap = {
    'Happy': 'var(--accent-orange)',
    'Sad': 'var(--accent-blue)',
    'Angry': 'var(--accent-red)',
    'Surprised': 'var(--accent-purple)',
    'Fear': 'var(--accent-purple)',
    'Disgust': 'var(--accent-green)',
    'Contempt': 'var(--accent-pink)',
    'Neutral': 'var(--text-secondary)',
  };
  DOM.emotionPrimary.style.color = colorMap[result.primary] || 'var(--accent-orange)';

  sendToOLED('emotion', {
    line1: result.primary.toUpperCase(),
    line2: result.confidence,
    scroll: false,
  });
}

function updateFocusUI(result) {
  DOM.focusState.textContent = result.state;
  const confNum = parseInt(result.confidence) || 0;
  DOM.focusConfBar.style.width = `${confNum}%`;
  DOM.focusConfVal.textContent = result.confidence;
  DOM.focusEvidence.textContent = result.evidence;

  const colorMap = {
    'Deep Focus': 'var(--accent-cyan)',
    'Focused': 'var(--accent-blue)',
    'Partially Focused': 'var(--accent-orange)',
    'Distracted': 'var(--accent-red)',
    'Disengaged': 'var(--text-muted)',
  };
  DOM.focusState.style.color = colorMap[result.state] || 'var(--accent-cyan)';

  sendToOLED('focus', {
    line1: result.state.toUpperCase(),
    line2: '',
    scroll: false,
  });
}

function updateSignUI(result) {
  const currentState = result.state || 'idle';

  // ── Status Badge ──
  const statusBadge = DOM.signStatusBadge;
  statusBadge.textContent = result.status;
  statusBadge.className = 'sign-status-badge';
  if (currentState === 'word-confirmed') {
    statusBadge.classList.add('word-recognized');
  } else if (currentState === 'locked') {
    statusBadge.classList.add('processing');
  } else if (currentState === 'stabilizing') {
    statusBadge.classList.add('recording');
  } else {
    statusBadge.classList.add('detecting');
  }

  // ── Live badge ──
  if (currentState === 'word-confirmed') {
    DOM.signBadge.className = 'panel-badge completed';
    DOM.signBadge.textContent = '✓ DONE';
  } else if (currentState === 'locked') {
    DOM.signBadge.className = 'panel-badge processing';
    DOM.signBadge.textContent = '🔒 LOCK';
  } else {
    DOM.signBadge.className = result.detected ? 'panel-badge active' : 'panel-badge inactive';
    DOM.signBadge.textContent = result.detected ? 'LIVE' : 'IDLE';
  }

  // ── Current sign + emoji ──
  const sign = result.current_sign;
  if (sign) {
    const emojiMap = {
      'I': '👤', 'YOU': '👉', 'HELLO': '👋', 'YES': '👍', 'NO': '🙅',
      'STOP': '✋', 'GO': '👉', 'COME': '👈', 'THANK YOU': '🙏',
      'PLEASE': '🙏', 'EAT': '🍽️', 'DRINK': '🥤', 'WANT': '🤲',
      'NEED': '🤲', 'HELP': '🆘', 'HOME': '🏠', 'SCHOOL': '🏫',
      'FOOD': '🍽️', 'WATER': '💧', 'GOOD': '👍', 'BAD': '👎',
      'SORRY': '😔', 'LOVE': '❤️', 'FRIEND': '🤝', 'FAMILY': '👨‍👩‍👧',
      'TODAY': '📅', 'TOMORROW': '📆', 'HAPPY': '😊', 'SAD': '😢', 'MORE': '➕',
    };
    DOM.gestureDisplay.textContent = emojiMap[sign.sign] || '🤚';

    if (currentState === 'word-confirmed') {
      DOM.signInterpretation.textContent = `✅ ${sign.sign}`;
      DOM.signInterpretation.style.color = 'var(--accent-green)';
    } else if (currentState === 'locked') {
      DOM.signInterpretation.textContent = `🔒 ${sign.sign}`;
      DOM.signInterpretation.style.color = 'var(--accent-orange)';
    } else {
      DOM.signInterpretation.textContent = sign.sign;
      DOM.signInterpretation.style.color = 'var(--accent-green)';
    }

    const confNum = sign.confidence || 0;
    DOM.signConfBar.style.width = `${confNum}%`;
    DOM.signConfVal.textContent = `${confNum}%`;
  } else if (!result.detected) {
    DOM.gestureDisplay.textContent = '—';
    DOM.signInterpretation.textContent = 'No hands detected';
    DOM.signInterpretation.style.color = 'var(--accent-green)';
    DOM.signConfBar.style.width = '0%';
    DOM.signConfVal.textContent = '—';
  } else {
    DOM.gestureDisplay.textContent = '🤚';
    DOM.signInterpretation.textContent = 'Analyzing...';
    DOM.signInterpretation.style.color = 'var(--accent-green)';
  }

  // ── Forming row: stabilization progress ──
  if (result.forming_letters && result.forming_letters.length > 0 && currentState === 'stabilizing') {
    DOM.signFormingRow.style.display = 'flex';
    const pct = Math.round((result.forming_progress || 0) * 100);
    DOM.signFormingText.textContent = `${result.forming_letters} — ${pct}%`;
  } else {
    DOM.signFormingRow.style.display = 'none';
  }

  // ── Confidence bars ──
  const wordConf = parseInt(result.word_confidence) || 0;
  DOM.wordConfBar.style.width = `${wordConf}%`;
  DOM.wordConfVal.textContent = result.word_confidence;
  DOM.sentenceConfBar.style.width = '0%';
  DOM.sentenceConfVal.textContent = '—';

  // ── Alternatives ──
  DOM.signAlternatives.innerHTML = '';
  if (result.alternatives && result.alternatives.length > 0) {
    for (const alt of result.alternatives) {
      const chip = document.createElement('span');
      chip.className = 'alt-chip';
      chip.textContent = alt;
      DOM.signAlternatives.appendChild(chip);
    }
  }

  // ── Word history ──
  const sentenceEl = DOM.signCurrentSentence;
  if (result.sentence && result.sentence.length > 0) {
    sentenceEl.textContent = result.sentence;
    sentenceEl.classList.add('has-content');
    if (currentState === 'word-confirmed') {
      sentenceEl.classList.add('sentence-reveal');
    } else {
      sentenceEl.classList.remove('sentence-reveal');
    }
  } else {
    sentenceEl.innerHTML = '<span class="sign-sentence-placeholder">Start signing to see detected words...</span>';
    sentenceEl.classList.remove('has-content', 'sentence-reveal');
  }

  DOM.signTranscriptHistory.innerHTML = '';

  sendToOLED('sign', {
    line1: 'SIGNING:',
    line2: result.sentence || '',
    scroll: (result.sentence?.length ?? 0) > 16,
  });
}

function updateQualityUI(result) {
  DOM.qLighting.textContent = result.lighting;
  DOM.qOcclusion.textContent = result.occlusion;
  DOM.qClarity.textContent = result.clarity;

  // Color coding
  setQualityColor(DOM.qLighting, result.lighting, ['Adequate', 'Bright'], ['Low', 'Flat']);
  setQualityColor(DOM.qOcclusion, result.occlusion, ['None'], ['Partial']);
  setQualityColor(DOM.qClarity, result.clarity, ['Good'], ['Moderate movement']);
}

function setQualityColor(el, value, goodValues, warnValues) {
  el.className = 'q-value';
  if (goodValues.some(v => value.includes(v))) el.classList.add('good');
  else if (warnValues.some(v => value.includes(v))) el.classList.add('adequate');
  else if (value === '—' || value === 'Unknown') el.classList.add('none');
  else el.classList.add('poor');
}

// ── NeuroLens UI Update (closest object only) ──
function updateNeuroLensUI(result) {
  if (!result) return;

  // Alert zone styling
  const zoneColors = {
    'SAFE': { color: 'var(--accent-green)', icon: '✓', bg: 'rgba(105, 240, 174, 0.08)' },
    'NEAR': { color: 'var(--accent-orange)', icon: '⚠', bg: 'rgba(255, 171, 64, 0.08)' },
    'VERY CLOSE': { color: 'var(--accent-red)', icon: '⛔', bg: 'rgba(255, 82, 82, 0.12)' },
  };

  const zoneStyle = zoneColors[result.alertZone] || zoneColors['SAFE'];
  DOM.nlAlertZone.style.background = zoneStyle.bg;
  DOM.nlAlertZone.style.borderColor = zoneStyle.color;
  DOM.nlAlertIcon.textContent = zoneStyle.icon;
  DOM.nlAlertIcon.style.color = zoneStyle.color;
  DOM.nlAlertLabel.textContent = result.alertLabel;
  DOM.nlAlertLabel.style.color = zoneStyle.color;

  DOM.nlDetectionCount.textContent = result.detectionCount;

  // Only show the CLOSEST object (mostDangerous = sorted by danger score)
  if (result.mostDangerous) {
    DOM.nlAlertSub.textContent = `${result.mostDangerous.class} detected — ${result.mostDangerous.direction}`;
    DOM.nlThreatCard.style.display = '';
    DOM.nlThreatName.textContent = result.mostDangerous.class.toUpperCase();
    DOM.nlThreatDist.textContent = `${result.mostDangerous.distanceM}m`;
    DOM.nlThreatDir.textContent = result.mostDangerous.direction;
    DOM.nlThreatDanger.textContent = result.mostDangerous.dangerScore;

    // Color danger score
    const ds = parseFloat(result.mostDangerous.dangerScore);
    DOM.nlThreatDanger.style.color = ds > 0.7 ? 'var(--accent-red)' : ds > 0.4 ? 'var(--accent-orange)' : 'var(--accent-green)';
  } else {
    DOM.nlAlertSub.textContent = 'No obstacles detected';
    DOM.nlThreatCard.style.display = 'none';
  }

  // Detection list (show all but only mark closest)
  DOM.nlDetectionsList.innerHTML = '';
  for (let i = 0; i < Math.min(result.detections.length, 5); i++) {
    const det = result.detections[i];
    const chip = document.createElement('div');
    chip.className = `nl-detection-chip nl-zone-${det.zone.replace(' ', '-').toLowerCase()}`;
    const isClosest = (i === 0);
    chip.innerHTML = `<span class="nl-det-name">${isClosest ? '📍 ' : ''}${det.class}</span>
      <span class="nl-det-zone">${det.zone}</span>
      <span class="nl-det-dist">${det.distanceM}m</span>
      <span class="nl-det-dir">${det.direction}</span>`;
    DOM.nlDetectionsList.appendChild(chip);
  }

  // Panel border animation
  DOM.obstaclePanel.classList.toggle('obstacle-warning', result.alertZone === 'NEAR');
  DOM.obstaclePanel.classList.toggle('obstacle-danger', result.alertZone === 'VERY CLOSE');

  // OLED: clear screen in obstacle/blind mode
  sendToOLED('obstacle', null);
}

// ── JSON Output ──
let jsonUpdateCounter = 0;

function updateJSON(emotion, focus, sign, quality, neurolens) {
  // Throttle JSON updates to every 5 frames
  jsonUpdateCounter++;
  if (jsonUpdateCounter % 5 !== 0) return;

  const output = {
    active_mode: currentMode,
  };

  if (emotion) {
    output.emotion = {
      primary: emotion.primary,
      secondary: emotion.secondary,
      intensity: emotion.intensity,
      confidence: emotion.confidence,
      evidence: emotion.evidence,
    };
  }
  if (focus) {
    output.focus = {
      state: focus.state,
      confidence: focus.confidence,
      evidence: focus.evidence,
    };
  }
  if (sign) {
    output.sign_language = {
      detected: sign.detected,
      current_word: sign.current_word || '',
      sentence: sign.sentence || '',
      alternatives: sign.alternatives || [],
      word_confidence: sign.word_confidence || '—',
      status: sign.status || 'Detecting...',
    };
  }
  if (neurolens) {
    output.neurolens = {
      alert_zone: neurolens.alertZone,
      detection_count: neurolens.detectionCount,
      most_dangerous: neurolens.mostDangerous,
      detections: neurolens.detections,
    };
  }
  if (quality) {
    output.visual_quality = {
      lighting: quality.lighting,
      occlusion: quality.occlusion,
      clarity: quality.clarity,
    };
  }

  DOM.jsonPre.innerHTML = syntaxHighlightJSON(JSON.stringify(output, null, 2));
}

function syntaxHighlightJSON(json) {
  return json
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"([^"]+)":/g, '<span class="json-key">"$1"</span>:')
    .replace(/: "([^"]*)"/g, ': <span class="json-string">"$1"</span>')
    .replace(/: (\d+\.?\d*)/g, ': <span class="json-number">$1</span>')
    .replace(/: (true|false)/g, ': <span class="json-bool">$1</span>')
    .replace(/: (null)/g, ': <span class="json-null">$1</span>');
}

// ── Face Landmark Index Groups ──
const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10];
const LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33];
const RIGHT_EYE = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362];
const LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61];
const LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 70];
const RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276, 300];