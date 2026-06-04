/**
 * NeuroSense AI — Main Application
 * Orchestrates webcam capture, MediaPipe models, analysis modules, and UI rendering.
 */

import { FaceLandmarker, HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { EmotionAnalyzer } from './emotion-analyzer.js?v=3';
import { FocusAnalyzer } from './focus-analyzer.js?v=5';
import { SignLanguageAnalyzer } from './sign-language-analyzer.js';
import { VisualQualityAnalyzer } from './visual-quality.js';
import { NeuroLensAnalyzer } from './neurolens-analyzer.js?v=11';
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
  recoveryActivitySelect: $('recoveryActivitySelect'),
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
  nlModeSelect: $('nlModeSelect'),
  nlAudioToggleInput: $('nlAudioToggleInput'),
  nlAudioToggleLabel: $('nlAudioToggleLabel'),
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
  tsReadAloudBtn: $('tsReadAloudBtn'),
  tsError: $('tsError'),
  tsEvidence: $('tsEvidence'),
};

// ── Pi Camera Config ──
const PI_STREAM_URL = 'http://10.165.194.112:5001/stream';

// ── OLED Display Config ──
const PI_OLED_URL = 'http://10.165.194.112:5002/oled';
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
window.nlAudioEnabled = false;
let usePiCamera = false;      // false = laptop webcam, true = Pi Camera Module 3
let piImgEl = null;           // <img> fed by MJPEG stream
let piFrameCanvas = null;     // offscreen canvas bridging MJPEG → video
let piFrameCtx = null;
let piFrameLoopId = null;
let cameraSwitching = false;  // FIX: guards detectLoop during camera transitions

// ── Focus Recovery Mode State ──
let focusRecoveryActive = false;
let distractionTime = 0;
let recoveryTarget = null;
let recoverySuccessTime = null;
let hasSeenFace = false;
let recoveryCooldownEndTime = 0;
let selectedRecoveryActivity = 'tracking';



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
  sendToOLED(mode, null); // Clear OLED when switching modes
  if (window.updateAIOrbMode) window.updateAIOrbMode(mode);

  if (mode !== 'focus') {
    focusRecoveryActive = false;
    distractionTime = 0;
    recoveryTarget = null;
    recoverySuccessTime = null;
    hasSeenFace = false;
    recoveryCooldownEndTime = 0;
  }

  // Calm Zone specific transition logic
  if (typeof handleCalmZoneSwitch === 'function') {
    handleCalmZoneSwitch(mode);
  }

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
  const modeNames = { 
    emotion: 'Analyzing emotions…', 
    focus: 'Tracking attention…', 
    sign: 'Reading signs…', 
    neurolens: 'Scanning surroundings…', 
    textscanner: 'Text scanner ready…',
    calmzone: 'Calm & Neuro Support active…'
  };
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

// ── Focus Recovery Selector Listener ──
if (DOM.recoveryActivitySelect) {
  DOM.recoveryActivitySelect.addEventListener('change', (e) => {
    selectedRecoveryActivity = e.target.value;
    if (focusRecoveryActive) {
      recoveryTarget = null;
    }
  });
}

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

// ── Obstacle Audio Toggle ──
if (DOM.nlAudioToggleInput) {
  DOM.nlAudioToggleInput.addEventListener('change', (e) => {
    window.nlAudioEnabled = e.target.checked;
    DOM.nlAudioToggleLabel.textContent = window.nlAudioEnabled ? '🔊' : '🔈';
    if (!window.nlAudioEnabled && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
  });
}

// ── Text Scanner Read Aloud ──
if (DOM.tsReadAloudBtn) {
  DOM.tsReadAloudBtn.addEventListener('click', () => {
    if (textScanner.summary && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const msg = new SpeechSynthesisUtterance(textScanner.summary);
      msg.rate = 1.0;
      const voices = window.speechSynthesis.getVoices();
      if (voices.length > 0) {
        msg.voice = voices.find(v => v.lang.startsWith('en')) || voices[0];
      }
      window.speechSynthesis.speak(msg);
    }
  });
}

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
  if (DOM.tsReadAloudBtn) DOM.tsReadAloudBtn.style.display = 'none';
  if ('speechSynthesis' in window) window.speechSynthesis.cancel();
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
      if (DOM.tsReadAloudBtn) DOM.tsReadAloudBtn.style.display = 'inline-block';
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

  // FIX: Block detectLoop BEFORE touching the video element.
  // Without this flag the loop fires between stopCurrentStream() and the new
  // stream being ready, calling detectForVideo on an invalid <video> which
  // throws an unhandled error and kills the loop permanently.
  cameraSwitching = true;

  if (usePiCamera) {
    btnPi.classList.add('active', 'pi');
    btnLaptop.classList.remove('active');
    DOM.webcam.style.transform = 'scaleX(1)';
    DOM.canvas.style.transform = 'scaleX(1)';
  } else {
    btnLaptop.classList.add('active');
    btnLaptop.classList.remove('pi');
    btnPi.classList.remove('active', 'pi');
    // Reset container aspect ratio to 16:9 for laptop webcam
    const _vc = document.getElementById('videoContainer');
    if (_vc) _vc.style.aspectRatio = '16 / 9';
    DOM.webcam.style.transform = 'scaleX(-1)';
    DOM.canvas.style.transform = 'scaleX(1)';
  }
  if (!DOM.statusDot.classList.contains('active')) {
    cameraSwitching = false;
    return;
  }
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
  } finally {
    // FIX: Always re-enable the loop, even on error, so modes keep working
    cameraSwitching = false;
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
  // Start at 1×1 — will auto-resize to the actual stream resolution on the first frame.
  piFrameCanvas = document.createElement('canvas');
  piFrameCanvas.width = 1;
  piFrameCanvas.height = 1;
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
        // FIX: Resize canvas to actual stream resolution (was hardcoded 320×240).
        if (piFrameCanvas.width !== piImgEl.naturalWidth || piFrameCanvas.height !== piImgEl.naturalHeight) {
          piFrameCanvas.width = piImgEl.naturalWidth;
          piFrameCanvas.height = piImgEl.naturalHeight;
          console.log(`📐 Pi Camera canvas resized to native ${piImgEl.naturalWidth}×${piImgEl.naturalHeight}`);
        }
        // Draw at native resolution — no forced downscaling
        piFrameCtx.drawImage(piImgEl, 0, 0);
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
  const canvasStream = piFrameCanvas.captureStream(30);
  DOM.webcam.srcObject = canvasStream;
  // FIX: Canvas dimensions are now set by drawPiFrame to match native stream resolution.
  // Mirror them to the overlay canvas so it lines up correctly.
  DOM.canvas.width = piFrameCanvas.width;
  DOM.canvas.height = piFrameCanvas.height;

  // FIX: Update the container's aspect-ratio to match the Pi cam (640×480 = 4:3),
  // instead of forcing 16:9 which causes CSS to crop/zoom the feed.
  const videoContainer = document.getElementById('videoContainer');
  if (videoContainer && piFrameCanvas.width > 1) {
    videoContainer.style.aspectRatio = `${piFrameCanvas.width} / ${piFrameCanvas.height}`;
  }
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

  // FIX: Skip processing while camera is being swapped out.
  // Without this, detectForVideo fires on an invalid <video> element
  // (srcObject = null, readyState = 0) and throws, killing the loop.
  if (cameraSwitching) {
    animFrameId = requestAnimationFrame(detectLoop);
    return;
  }

  // FIX: Ensure the video element actually has frame data before sending
  // it to MediaPipe. HAVE_CURRENT_DATA (readyState >= 2) means at least
  // one decoded frame is available. This catches the brief gap after a
  // camera switch where srcObject is set but no frame has been decoded yet.
  if (DOM.webcam.readyState < 2) {
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

  // FIX: Wrap detection in try-catch so the loop can NEVER die.
  // MediaPipe's detectForVideo can throw on corrupted frames, canvas
  // capture timing glitches, or GPU context loss. An unhandled throw
  // skips the requestAnimationFrame call at the bottom → loop dies → all
  // modes freeze permanently. The catch lets us survive and retry next frame.
  try {
    if (needFace) {
      faceResults = faceLandmarker.detectForVideo(DOM.webcam, timestamp);
    }
    if (needHand) {
      handResults = handLandmarker.detectForVideo(DOM.webcam, timestamp);
    }
  } catch (detectionErr) {
    console.warn('⚠️ detectForVideo error (retrying next frame):', detectionErr.message);
    animFrameId = requestAnimationFrame(detectLoop);
    return;
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
    const hasFace = faceResults?.faceLandmarks?.length > 0;
    let noseX = 0;
    let noseY = 0;

    if (hasFace) {
      hasSeenFace = true;
      const landmarks = faceResults.faceLandmarks[0];
      const blendshapes = faceResults.faceBlendshapes?.[0]?.categories || [];
      drawFaceMesh(ctx, landmarks);
      focusResult = focusAnalyzer.analyze(blendshapes, landmarks, timestamp);

      const isMirrored = !usePiCamera;
      const w = DOM.canvas.width;
      const h = DOM.canvas.height;
      const noseTip = landmarks[1];
      noseX = isMirrored ? (1 - noseTip.x) * w : noseTip.x * w;
      noseY = noseTip.y * h;
    } else {
      focusResult = focusAnalyzer.analyze(null, null, timestamp);
    }

    const inGracePeriod = timestamp < recoveryCooldownEndTime;
    const isUnfocused = !inGracePeriod && (
                        focusResult.state === 'Distracted' || 
                        focusResult.state === 'Disengaged' || 
                        (focusResult.state === 'Not reliably detectable' && hasSeenFace)
    );

    if (isUnfocused) {
      if (!focusRecoveryActive && !recoverySuccessTime) {
        if (distractionTime === 0) {
          distractionTime = timestamp;
        } else {
          const elapsed = timestamp - distractionTime;
          if (elapsed > 4500) {
            focusRecoveryActive = true;
            recoveryTarget = null;
            distractionTime = 0;
            playNotificationSound();
          } else {
            const pct = (elapsed / 4500) * 100;
            drawDistractionWarningCircle(ctx, pct);
          }
        }
      }
    } else {
      distractionTime = 0;
    }

    if (focusRecoveryActive) {
      if (hasFace) {
        let completed = false;
        if (selectedRecoveryActivity === 'tracking') {
          completed = drawFocusRecovery(ctx, noseX, noseY, timestamp);
        } else if (selectedRecoveryActivity === 'breathing') {
          completed = drawFocusBreathing(ctx, noseX, noseY, timestamp);
        } else if (selectedRecoveryActivity === 'gymnastics') {
          completed = drawFocusGymnastics(ctx, noseX, noseY, timestamp);
        }

        if (completed) {
          focusRecoveryActive = false;
          recoverySuccessTime = timestamp;
          recoveryCooldownEndTime = timestamp + 5000;
          recoveryTarget = null;
          playSuccessSound();
        }
      } else {
        drawFocusRecoveryFaceLost(ctx, timestamp);
      }
    }

    if (recoverySuccessTime) {
      drawFocusSuccess(ctx, timestamp);
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
    const obstacleMode = DOM.nlModeSelect ? DOM.nlModeSelect.value : 'navigation';
    neurolensAnalyzer.analyze(DOM.webcam, ctx, DOM.canvas.width, DOM.canvas.height, !usePiCamera, obstacleMode)
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

function drawDistractionWarningCircle(ctx, progress) {
  const w = DOM.canvas.width;
  const h = DOM.canvas.height;

  // Center of screen
  const cx = w / 2;
  const cy = h / 2;
  const radius = 30;

  // 1. Dark, semi-transparent circle background
  ctx.fillStyle = 'rgba(18, 18, 24, 0.82)';
  ctx.strokeStyle = 'rgba(217, 150, 150, 0.4)';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.arc(cx, cy, radius + 10, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // 2. Draw progress ring (red closing in)
  ctx.strokeStyle = '#d9534f';
  ctx.lineWidth = 3.5;
  ctx.beginPath();
  ctx.arc(
    cx, cy,
    radius + 5,
    -Math.PI / 2,
    -Math.PI / 2 + (Math.PI * 2 * (progress / 100))
  );
  ctx.stroke();

  // 3. Warning icon / text
  ctx.fillStyle = '#d9534f';
  ctx.font = 'bold 12px var(--font)';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('⚠️', cx, cy);
}

function drawFocusBreathing(ctx, noseX, noseY, timestamp) {
  const w = DOM.canvas.width;
  const h = DOM.canvas.height;
  const cx = w / 2;
  const cy = h / 2;

  if (!recoveryTarget) {
    recoveryTarget = {
      progress: 0,
      startTime: timestamp,
      lastUpdate: timestamp
    };
  }

  const elapsed = timestamp - recoveryTarget.startTime;
  const cycleDuration = 10000; // 10 seconds total
  const cycleTime = elapsed % cycleDuration;
  
  // Calculate breath stage
  let stage = 'inhale'; // inhale, hold, exhale
  let targetRadius = 40;
  let label = '';
  let color = 'var(--accent-cyan)';

  if (cycleTime < 4000) {
    // Inhale: 0s to 4s
    stage = 'inhale';
    const pct = cycleTime / 4000;
    targetRadius = 40 + pct * 40; // 40 to 80
    label = 'Breathe In... 🌬️';
    color = 'var(--accent-cyan)';
  } else if (cycleTime < 6000) {
    // Hold: 4s to 6s
    stage = 'hold';
    targetRadius = 80;
    label = 'Hold... 🧘';
    color = 'var(--accent-purple)';
  } else {
    // Exhale: 6s to 10s
    stage = 'exhale';
    const pct = (cycleTime - 6000) / 4000;
    targetRadius = 80 - pct * 40; // 80 to 40
    label = 'Breathe Out... 💨';
    color = 'var(--accent-green)';
  }

  recoveryTarget.progress = Math.min(100, (elapsed / cycleDuration) * 100);

  // --- DRAWING ---
  // 1. Dark background backdrop
  ctx.fillStyle = 'rgba(18, 18, 24, 0.75)';
  ctx.beginPath();
  ctx.arc(cx, cy, 100, 0, Math.PI * 2);
  ctx.fill();

  // 2. Breathing Circle (pulsing/filling size)
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.2;
  ctx.beginPath();
  ctx.arc(cx, cy, targetRadius, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 1.0;

  // Outer ring of breathing circle
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(cx, cy, targetRadius, 0, Math.PI * 2);
  ctx.stroke();

  // 3. Steady target alignment dot (user's nose)
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(noseX, noseY, 10, 0, Math.PI * 2);
  ctx.stroke();

  // Draw guidance ring in center
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.arc(cx, cy, 30, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);

  // 4. Labels
  ctx.fillStyle = '#f0f0f5';
  ctx.font = 'bold 14px var(--font)';
  ctx.textAlign = 'center';
  ctx.fillText(label, cx, cy - 10);

  ctx.fillStyle = 'var(--text-muted)';
  ctx.font = '10px var(--mono)';
  ctx.fillText('KEEP NOSE CENTERED', cx, cy + 20);

  return recoveryTarget.progress >= 100;
}

function drawFocusGymnastics(ctx, noseX, noseY, timestamp) {
  const w = DOM.canvas.width;
  const h = DOM.canvas.height;

  if (!recoveryTarget) {
    recoveryTarget = {
      step: 0,
      stepProgress: 0,
      progress: 0,
      x: w * 0.75,
      y: h / 2
    };
  }

  const steps = [
    { x: w * 0.75, y: h / 2, label: 'Look Right ➔' },
    { x: w * 0.25, y: h / 2, label: '➔ Look Left' },
    { x: w / 2, y: h * 0.25, label: 'Look Up ▲' },
    { x: w / 2, y: h * 0.75, label: '▼ Look Down' }
  ];

  const currentTarget = steps[recoveryTarget.step];
  recoveryTarget.x = currentTarget.x;
  recoveryTarget.y = currentTarget.y;

  // Collision check
  const dist = Math.hypot(noseX - recoveryTarget.x, noseY - recoveryTarget.y);
  const isAligned = dist < 45; // reticle matches dot

  if (isAligned) {
    recoveryTarget.stepProgress += 2.5; // takes ~40 frames (~1.2s)
    if (recoveryTarget.stepProgress >= 100) {
      recoveryTarget.stepProgress = 0;
      recoveryTarget.step++;
      if (recoveryTarget.step < 4) {
        playNote(600, 0.1);
      }
    }
  } else {
    recoveryTarget.stepProgress = Math.max(0, recoveryTarget.stepProgress - 1.0);
  }

  recoveryTarget.progress = (recoveryTarget.step / 4) * 100 + (recoveryTarget.stepProgress / 4);

  // --- DRAWING ---
  // 1. Solid backing circle
  ctx.fillStyle = 'rgba(18, 18, 24, 0.75)';
  ctx.beginPath();
  ctx.arc(recoveryTarget.x, recoveryTarget.y, 40, 0, Math.PI * 2);
  ctx.fill();

  // 2. Target Orb
  ctx.strokeStyle = isAligned ? 'var(--accent-green)' : 'var(--accent-orange)';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.arc(recoveryTarget.x, recoveryTarget.y, 25, 0, Math.PI * 2);
  ctx.stroke();

  // Progress outer arc
  if (recoveryTarget.stepProgress > 0) {
    ctx.strokeStyle = 'var(--accent-cyan)';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(
      recoveryTarget.x, recoveryTarget.y,
      30,
      -Math.PI / 2,
      -Math.PI / 2 + (Math.PI * 2 * (recoveryTarget.stepProgress / 100))
    );
    ctx.stroke();
  }

  // Target core
  ctx.fillStyle = isAligned ? 'var(--accent-green)' : 'var(--accent-orange)';
  ctx.beginPath();
  ctx.arc(recoveryTarget.x, recoveryTarget.y, 8, 0, Math.PI * 2);
  ctx.fill();

  // 3. User Reticle
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(noseX, noseY, 12, 0, Math.PI * 2);
  ctx.stroke();

  // 4. Instructions
  ctx.fillStyle = '#f0f0f5';
  ctx.font = 'bold 12px var(--font)';
  ctx.textAlign = 'center';
  ctx.fillText(currentTarget.label, recoveryTarget.x, recoveryTarget.y - 48);

  ctx.fillStyle = 'var(--text-muted)';
  ctx.font = '10px var(--mono)';
  ctx.fillText(`Target ${recoveryTarget.step + 1} of 4`, recoveryTarget.x, recoveryTarget.y + 48);

  return recoveryTarget.step >= 4;
}

function playNote(freq, duration) {
  try {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
  } catch (e) {}
}

function drawFocusRecovery(ctx, noseX, noseY, timestamp) {
  const w = DOM.canvas.width;
  const h = DOM.canvas.height;

  // Initialize target if not done
  if (!recoveryTarget) {
    recoveryTarget = {
      x: w / 2,
      y: h / 2,
      radius: 40,
      progress: 0,
      targetX: Math.random() * (w - 120) + 60,
      targetY: Math.random() * (h - 120) + 60,
      speed: 2,
      lastUpdate: timestamp
    };
  }

  // Move target towards its destination
  const dx = recoveryTarget.targetX - recoveryTarget.x;
  const dy = recoveryTarget.targetY - recoveryTarget.y;
  const distToDest = Math.hypot(dx, dy);
  
  if (distToDest < 10) {
    // Pick new target destination
    recoveryTarget.targetX = Math.random() * (w - 120) + 60;
    recoveryTarget.targetY = Math.random() * (h - 120) + 60;
  } else {
    recoveryTarget.x += (dx / distToDest) * recoveryTarget.speed;
    recoveryTarget.y += (dy / distToDest) * recoveryTarget.speed;
  }

  // Check collision with nose tip
  const noseDist = Math.hypot(noseX - recoveryTarget.x, noseY - recoveryTarget.y);
  const isLocked = noseDist < recoveryTarget.radius + 20;

  if (isLocked) {
    recoveryTarget.progress += 1.2; // takes ~80 frames (~2.5s) to complete
    if (recoveryTarget.progress > 100) {
      recoveryTarget.progress = 100;
    }
  } else {
    recoveryTarget.progress = Math.max(0, recoveryTarget.progress - 0.4);
  }

  // --- DRAWING ---
  // 0. Solid dark backing circle for high-contrast visibility against webcam
  ctx.fillStyle = 'rgba(18, 18, 24, 0.72)';
  ctx.beginPath();
  ctx.arc(recoveryTarget.x, recoveryTarget.y, recoveryTarget.radius + 6, 0, Math.PI * 2);
  ctx.fill();

  // 1. Draw guidance line connecting nose tip to target
  ctx.strokeStyle = isLocked ? 'rgba(139, 203, 216, 0.5)' : 'rgba(111, 66, 193, 0.3)';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(noseX, noseY);
  ctx.lineTo(recoveryTarget.x, recoveryTarget.y);
  ctx.stroke();
  ctx.setLineDash([]); // reset

  // 2. Draw target outer glow (pulsing)
  const pulseFactor = 1 + 0.1 * Math.sin(timestamp / 150);
  const glowGrad = ctx.createRadialGradient(
    recoveryTarget.x, recoveryTarget.y, 5,
    recoveryTarget.x, recoveryTarget.y, recoveryTarget.radius * 1.8 * pulseFactor
  );
  glowGrad.addColorStop(0, isLocked ? 'rgba(0, 172, 193, 0.6)' : 'rgba(111, 66, 193, 0.4)');
  glowGrad.addColorStop(0.5, isLocked ? 'rgba(0, 172, 193, 0.2)' : 'rgba(111, 66, 193, 0.1)');
  glowGrad.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = glowGrad;
  ctx.beginPath();
  ctx.arc(recoveryTarget.x, recoveryTarget.y, recoveryTarget.radius * 1.8 * pulseFactor, 0, Math.PI * 2);
  ctx.fill();

  // 3. Draw target core (Darker, rich purple/cyan)
  ctx.fillStyle = isLocked ? '#00acc1' : '#6f42c1';
  ctx.beginPath();
  ctx.arc(recoveryTarget.x, recoveryTarget.y, 8, 0, Math.PI * 2);
  ctx.fill();

  // 4. Draw outer target ring (Darker, rich purple/cyan)
  ctx.strokeStyle = isLocked ? '#00acc1' : '#6f42c1';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.arc(recoveryTarget.x, recoveryTarget.y, recoveryTarget.radius, 0, Math.PI * 2);
  ctx.stroke();

  // 5. Draw progress ring
  if (recoveryTarget.progress > 0) {
    ctx.strokeStyle = 'var(--accent-green)';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(
      recoveryTarget.x, recoveryTarget.y,
      recoveryTarget.radius + 5,
      -Math.PI / 2,
      -Math.PI / 2 + (Math.PI * 2 * (recoveryTarget.progress / 100))
    );
    ctx.stroke();
  }

  // 6. Draw "FOCUS TARGET" label
  ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
  ctx.font = '11px var(--font)';
  ctx.textAlign = 'center';
  ctx.fillText(
    isLocked ? `Syncing... ${Math.round(recoveryTarget.progress)}%` : 'Gaze Target',
    recoveryTarget.x,
    recoveryTarget.y - recoveryTarget.radius - 12
  );

  // 7. Draw reticle around user nose tip
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(noseX, noseY, 12, 0, Math.PI * 2);
  ctx.stroke();
  
  // Crosshair ticks
  ctx.beginPath();
  ctx.moveTo(noseX - 18, noseY); ctx.lineTo(noseX - 8, noseY);
  ctx.moveTo(noseX + 8, noseY); ctx.lineTo(noseX + 18, noseY);
  ctx.moveTo(noseX, noseY - 18); ctx.lineTo(noseX, noseY - 8);
  ctx.moveTo(noseX, noseY + 8); ctx.lineTo(noseX, noseY + 18);
  ctx.stroke();

  return recoveryTarget.progress >= 100;
}

function drawFocusRecoveryFaceLost(ctx, timestamp) {
  const w = DOM.canvas.width;
  const h = DOM.canvas.height;
  
  if (recoveryTarget) {
    // Decay progress slowly when face is lost
    recoveryTarget.progress = Math.max(0, recoveryTarget.progress - 0.2);
    
    // Draw target static/semi-transparent
    ctx.strokeStyle = 'rgba(185, 173, 223, 0.2)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(recoveryTarget.x, recoveryTarget.y, recoveryTarget.radius, 0, Math.PI * 2);
    ctx.stroke();
    
    if (recoveryTarget.progress > 0) {
      ctx.strokeStyle = 'rgba(156, 203, 175, 0.3)';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(
        recoveryTarget.x, recoveryTarget.y,
        recoveryTarget.radius + 5,
        -Math.PI / 2,
        -Math.PI / 2 + (Math.PI * 2 * (recoveryTarget.progress / 100))
      );
      ctx.stroke();
    }
  }

  // Draw overlay hint
  ctx.fillStyle = 'rgba(217, 150, 150, 0.85)';
  ctx.font = '12px var(--font)';
  ctx.textAlign = 'center';
  ctx.fillText('Camera view blocked — look back to target', w / 2, h - 30);
}

function drawFocusSuccess(ctx, timestamp) {
  const w = DOM.canvas.width;
  const h = DOM.canvas.height;
  const elapsed = timestamp - recoverySuccessTime;
  if (elapsed > 1500) {
    recoverySuccessTime = null;
    return;
  }

  // Calculate opacity
  let alpha = 1;
  if (elapsed > 1200) {
    alpha = 1 - (elapsed - 1200) / 300; // fade out in last 300ms
  }

  // Glassmorphic modal background
  ctx.fillStyle = `rgba(26, 26, 36, ${0.8 * alpha})`;
  ctx.strokeStyle = `rgba(156, 203, 175, ${0.3 * alpha})`;
  ctx.lineWidth = 2;
  
  const boxW = 260;
  const boxH = 140;
  const boxX = (w - boxW) / 2;
  const boxY = (h - boxH) / 2;
  
  // Rounded rect
  drawRoundedRect(ctx, boxX, boxY, boxW, boxH, 16);
  ctx.fill();
  ctx.stroke();

  // Glow ring
  const circleRadius = 25;
  const circleX = w / 2;
  const circleY = boxY + 45;
  
  ctx.strokeStyle = `rgba(156, 203, 175, ${alpha})`;
  ctx.lineWidth = 3;
  ctx.fillStyle = `rgba(156, 203, 175, ${0.15 * alpha})`;
  ctx.beginPath();
  ctx.arc(circleX, circleY, circleRadius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Draw Checkmark
  ctx.strokeStyle = `rgba(255, 255, 255, ${alpha})`;
  ctx.lineWidth = 4.5;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(circleX - 10, circleY + 1);
  ctx.lineTo(circleX - 2, circleY + 9);
  ctx.lineTo(circleX + 11, circleY - 6);
  ctx.stroke();
  ctx.lineCap = 'butt'; // reset

  // Draw Text
  ctx.fillStyle = `rgba(240, 240, 245, ${alpha})`;
  ctx.font = 'bold 15px var(--font)';
  ctx.textAlign = 'center';
  ctx.fillText('Focus Restored', w / 2, boxY + 105);

  ctx.fillStyle = `rgba(160, 160, 176, ${alpha})`;
  ctx.font = '11px var(--mono)';
  ctx.fillText('ATTENTION RESET COMPLETE', w / 2, boxY + 122);
}

function drawRoundedRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function drawFaceMesh(ctx, landmarks) {
  const isMirrored = !usePiCamera;
  const w = DOM.canvas.width;
  const h = DOM.canvas.height;

  // Draw tessellation dots
  ctx.fillStyle = 'rgba(0, 229, 255, 0.25)';
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    const x = isMirrored ? (1 - lm.x) * w : lm.x * w;
    const y = lm.y * h;
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
    const x = isMirrored ? (1 - lm.x) * w : lm.x * w;
    const y = lm.y * h;
    ctx.beginPath();
    ctx.arc(x, y, 1.5, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function drawContour(ctx, landmarks, indices, color, width) {
  if (!indices || indices.length < 2) return;
  const isMirrored = !usePiCamera;
  const w = DOM.canvas.width;
  const h = DOM.canvas.height;

  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  const first = landmarks[indices[0]];
  ctx.moveTo(isMirrored ? (1 - first.x) * w : first.x * w, first.y * h);
  for (let i = 1; i < indices.length; i++) {
    const lm = landmarks[indices[i]];
    ctx.lineTo(isMirrored ? (1 - lm.x) * w : lm.x * w, lm.y * h);
  }
  ctx.closePath();
  ctx.stroke();
}

function drawHandSkeleton(ctx, landmarks) {
  const isMirrored = !usePiCamera;
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
    ctx.moveTo(isMirrored ? (1 - landmarks[a].x) * w : landmarks[a].x * w, landmarks[a].y * h);
    ctx.lineTo(isMirrored ? (1 - landmarks[b].x) * w : landmarks[b].x * w, landmarks[b].y * h);
    ctx.stroke();
  }

  // Draw joints
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    const x = isMirrored ? (1 - lm.x) * w : lm.x * w;
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
  if (focusRecoveryActive && recoveryTarget) {
    let modeLabel = '🎯 Focus Recovery Mode';
    let evidenceText = 'Align your reticle with the floating target on screen to restore focus.';
    
    if (selectedRecoveryActivity === 'breathing') {
      modeLabel = '🧘 Breathing Sync Mode';
      evidenceText = 'Keep your face centered and follow the breathing cycle instructions on screen.';
    } else if (selectedRecoveryActivity === 'gymnastics') {
      modeLabel = '👀 Eye Gymnastics Mode';
      evidenceText = 'Follow the jumping target orb with your head reticle to complete eye gymnastics.';
    }

    DOM.focusState.textContent = modeLabel;
    DOM.focusState.style.color = 'var(--accent-purple)';
    DOM.focusConfBar.style.width = `${Math.round(recoveryTarget.progress)}%`;
    DOM.focusConfBar.className = 'confidence-fill purple';
    DOM.focusConfVal.textContent = `${Math.round(recoveryTarget.progress)}%`;
    DOM.focusEvidence.textContent = evidenceText;
    
    sendToOLED('focus', {
      line1: 'RECOVERING...',
      line2: `${Math.round(recoveryTarget.progress)}%`,
      scroll: false,
    });
    return;
  }

  DOM.focusConfBar.className = 'confidence-fill cyan';
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
    'Drowsy': 'var(--accent-purple)',
    'Microsleep Detected': 'var(--accent-pink)',
    'Disengaged': 'var(--text-muted)',
    'Not reliably detectable': 'var(--text-muted)',
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

function updateNeuroLensUI(result) {
  if (!result) return;

  // Alert zone styling — with sharper colours per severity
  const zoneColors = {
    'SAFE':       { color: 'var(--accent-green)',  icon: '\u2713',  bg: 'rgba(105, 240, 174, 0.08)' },
    'NEAR':       { color: 'var(--accent-orange)', icon: '\u26a0',  bg: 'rgba(255, 171, 64, 0.08)' },
    'VERY CLOSE': { color: 'var(--accent-red)',    icon: '\u26d4',  bg: 'rgba(255, 82, 82, 0.12)' },
  };

  const zoneStyle = zoneColors[result.alertZone] || zoneColors['SAFE'];
  DOM.nlAlertZone.style.background = zoneStyle.bg;
  DOM.nlAlertZone.style.borderColor = zoneStyle.color;
  DOM.nlAlertIcon.textContent = zoneStyle.icon;
  DOM.nlAlertIcon.style.color = zoneStyle.color;
  DOM.nlAlertLabel.textContent = result.alertLabel;
  DOM.nlAlertLabel.style.color = zoneStyle.color;

  DOM.nlDetectionCount.textContent = result.detectionCount;

  // Closest (most dangerous) object details
  if (result.mostDangerous) {
    const movingText = result.mostDangerous.isMoving ? ' [MOVING]' : '';
    DOM.nlAlertSub.textContent = `${result.mostDangerous.class}${movingText} \u2014 ${result.mostDangerous.distanceM}m ${result.mostDangerous.direction}`;
    DOM.nlThreatCard.style.display = '';
    DOM.nlThreatName.textContent = result.mostDangerous.class.toUpperCase();
    DOM.nlThreatDist.textContent = `${result.mostDangerous.distanceM}m`;
    DOM.nlThreatDir.textContent = result.mostDangerous.direction;
    DOM.nlThreatDanger.textContent = result.mostDangerous.dangerScore;

    // Colour danger score by severity
    const ds = parseFloat(result.mostDangerous.dangerScore);
    DOM.nlThreatDanger.style.color = ds > 0.7 ? 'var(--accent-red)' : ds > 0.4 ? 'var(--accent-orange)' : 'var(--accent-green)';
  } else {
    DOM.nlAlertSub.textContent = 'No obstacles detected';
    DOM.nlThreatCard.style.display = 'none';
  }

  // Detection list (show at most 5, closest first)
  DOM.nlDetectionsList.innerHTML = '';
  for (let i = 0; i < Math.min(result.detections.length, 5); i++) {
    const det = result.detections[i];
    const chip = document.createElement('div');
    chip.className = `nl-detection-chip nl-zone-${det.zone.replace(' ', '-').toLowerCase()}`;
    const isClosest = (i === 0);
    const movingIndicator = det.isMoving ? ' \ud83d\udca8' : '';
    chip.innerHTML = `<span class="nl-det-name">${isClosest ? '\ud83d\udccd ' : ''}${det.class}${movingIndicator}</span>
      <span class="nl-det-zone">${det.zone}</span>
      <span class="nl-det-dist">${det.distanceM}m</span>
      <span class="nl-det-dir">${det.direction}</span>`;
    DOM.nlDetectionsList.appendChild(chip);
  }

  // Panel glow animation by severity
  DOM.obstaclePanel.classList.toggle('obstacle-warning', result.alertZone === 'NEAR');
  DOM.obstaclePanel.classList.toggle('obstacle-danger', result.alertZone === 'VERY CLOSE');

  // OLED: send alert label and obstacle details — respect 3s cooldown to avoid spam
  if (result.mostDangerous && !result.cooldownActive) {
    const objName = result.mostDangerous.class;
    const dist = result.mostDangerous.distanceM;
    const dir = result.mostDangerous.direction;
    sendToOLED('obstacle', {
      line1: result.alertLabel,
      line2: `${objName} - ${dist}m ${dir}`,
      scroll: result.alertLabel.length > 16
    });
    
    if (window.nlAudioEnabled) {
      playNotificationSound(); // Play a beep first
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const msg = new SpeechSynthesisUtterance(result.alertLabel);
        msg.rate = 1.1; // Slightly faster to be responsive
        const voices = window.speechSynthesis.getVoices();
        if (voices.length > 0) {
          msg.voice = voices.find(v => v.lang.startsWith('en')) || voices[0];
        }
        window.speechSynthesis.speak(msg);
      }
    }
  } else if (!result.mostDangerous) {
    sendToOLED('obstacle', {
      line1: 'SAFE PATH',
      line2: 'No obstacles',
      scroll: false
    });
  }
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

// ── Audio Feedback Functions ──
function playNotificationSound() {
  try {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    
    // Note 1 (C5)
    const osc1 = audioCtx.createOscillator();
    const gain1 = audioCtx.createGain();
    osc1.type = 'sine';
    osc1.frequency.setValueAtTime(523.25, audioCtx.currentTime);
    gain1.gain.setValueAtTime(0.12, audioCtx.currentTime);
    gain1.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.45);
    osc1.connect(gain1);
    gain1.connect(audioCtx.destination);
    osc1.start();
    osc1.stop(audioCtx.currentTime + 0.45);

    // Note 2 (G5) at 120ms
    const osc2 = audioCtx.createOscillator();
    const gain2 = audioCtx.createGain();
    osc2.type = 'sine';
    osc2.frequency.setValueAtTime(783.99, audioCtx.currentTime + 0.12);
    gain2.gain.setValueAtTime(0, audioCtx.currentTime);
    gain2.gain.setValueAtTime(0.12, audioCtx.currentTime + 0.12);
    gain2.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.57);
    osc2.connect(gain2);
    gain2.connect(audioCtx.destination);
    osc2.start();
    osc2.stop(audioCtx.currentTime + 0.57);
  } catch (e) {
    console.warn("Failed to play notification sound:", e);
  }
}

function playSuccessSound() {
  try {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const now = audioCtx.currentTime;

    // Chime 1 (E5)
    const osc1 = audioCtx.createOscillator();
    const gain1 = audioCtx.createGain();
    osc1.type = 'sine';
    osc1.frequency.setValueAtTime(659.25, now);
    gain1.gain.setValueAtTime(0.1, now);
    gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.4);
    osc1.connect(gain1);
    gain1.connect(audioCtx.destination);
    osc1.start();
    osc1.stop(now + 0.4);

    // Chime 2 (G5) at 100ms
    const osc2 = audioCtx.createOscillator();
    const gain2 = audioCtx.createGain();
    osc2.type = 'sine';
    osc2.frequency.setValueAtTime(783.99, now + 0.1);
    gain2.gain.setValueAtTime(0, now);
    gain2.gain.setValueAtTime(0.1, now + 0.1);
    gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.5);
    osc2.connect(gain2);
    gain2.connect(audioCtx.destination);
    osc2.start();
    osc2.stop(now + 0.5);

    // Chime 3 (C6) at 200ms
    const osc3 = audioCtx.createOscillator();
    const gain3 = audioCtx.createGain();
    osc3.type = 'sine';
    osc3.frequency.setValueAtTime(1046.50, now + 0.2);
    gain3.gain.setValueAtTime(0, now);
    gain3.gain.setValueAtTime(0.12, now + 0.2);
    gain3.gain.exponentialRampToValueAtTime(0.001, now + 0.75);
    osc3.connect(gain3);
    gain3.connect(audioCtx.destination);
    osc3.start();
    osc3.stop(now + 0.75);
  } catch (e) {
    console.warn("Failed to play success sound:", e);
  }
}

// ── NeuroLens AI Orb Controller ──
const AI_ORB_CONFIG = {
  emotion: {
    status: 'Analyzing expressions...',
    glow: 'rgba(0, 229, 255, 0.55)',
    tip: 'Emotion Mode tracks micro-expressions and filters out head-pose variance for calm transitions.'
  },
  focus: {
    status: 'Tracking attention...',
    glow: 'rgba(185, 173, 223, 0.65)',
    tip: 'Focus Mode uses auto-calibration to detect Deep Focus, Looking Away, or Distracted states.'
  },
  sign: {
    status: 'Interpreting gestures...',
    glow: 'rgba(105, 240, 174, 0.6)',
    tip: 'Sign Language Mode supports real-time hand landmark skeleton rendering and sentence building.'
  },
  neurolens: {
    status: 'Scanning safe path...',
    glow: 'rgba(255, 110, 110, 0.65)',
    tip: 'Obstacle Mode alerts you of close obstacles in the center, left, or right columns.'
  },
  textscanner: {
    status: 'Reading text...',
    glow: 'rgba(0, 204, 180, 0.6)',
    tip: 'Text Scanner lets you scan text to get an instant summaries powered by Tesseract and local AI.'
  },
  calmzone: {
    status: 'Deep relaxation active...',
    glow: 'rgba(105, 240, 174, 0.55)',
    tip: 'Calm Zone is designed to help reduce sensory overload. Try Calm Mode to soften the page contrast.'
  }
};

function initAIOrb() {
  const container = document.getElementById('nlAiOrbContainer');
  const orb = document.getElementById('nlAiOrb');
  const tooltip = document.getElementById('nlAiOrbTooltip');
  const statusEl = document.getElementById('nlAiOrbStatus');
  const panel = document.getElementById('nlAiPanel');
  const panelClose = document.getElementById('nlAiPanelClose');
  const panelMode = document.getElementById('nlAiPanelMode');
  const panelStatus = document.getElementById('nlAiPanelStatus');
  const panelTip = document.getElementById('nlAiPanelTip');

  if (!orb) return;

  // Typing animation function
  let typingTimeout;
  function typeText(element, text) {
    clearTimeout(typingTimeout);
    element.textContent = '';
    let i = 0;
    function nextChar() {
      if (i < text.length) {
        element.textContent += text.charAt(i);
        i++;
        typingTimeout = setTimeout(nextChar, 35);
      }
    }
    nextChar();
  }

  // Update Orb visuals and labels based on current mode
  window.updateAIOrbMode = function(mode) {
    const config = AI_ORB_CONFIG[mode] || { status: 'NeuroLens AI Ready', glow: 'rgba(0, 229, 255, 0.5)', tip: 'Welcome!' };
    
    // Set custom property for CSS glow
    orb.style.setProperty('--orb-glow-color', config.glow);
    
    // Update labels
    typeText(statusEl, config.status);
    
    if (panelMode) panelMode.textContent = mode === 'neurolens' ? 'Obstacles' : mode.charAt(0).toUpperCase() + mode.slice(1);
    if (panelStatus) panelStatus.textContent = config.status;
    if (panelTip) panelTip.textContent = config.tip;
  };

  // Hover animations
  orb.addEventListener('mouseenter', () => {
    orb.style.transform = 'scale(1.15)';
    tooltip.style.opacity = '1';
    tooltip.style.transform = 'translateY(0)';
  });

  orb.addEventListener('mouseleave', () => {
    orb.style.transform = '';
    if (panel.style.display !== 'block') {
      tooltip.style.opacity = '';
      tooltip.style.transform = '';
    }
  });

  // Toggle info panel
  orb.addEventListener('click', (e) => {
    e.stopPropagation();
    const isShowing = panel.style.display === 'block';
    if (isShowing) {
      panel.style.display = 'none';
      panel.style.opacity = '0';
      panel.style.transform = 'scale(0.9) translateY(10px)';
    } else {
      panel.style.display = 'block';
      // Trigger reflow for animations
      panel.offsetHeight;
      panel.style.opacity = '1';
      panel.style.transform = 'scale(1) translateY(0)';
      // Hide tooltip when panel is open
      tooltip.style.opacity = '0';
    }
  });

  // Close panel
  if (panelClose) {
    panelClose.addEventListener('click', (e) => {
      e.stopPropagation();
      panel.style.display = 'none';
      panel.style.opacity = '0';
      panel.style.transform = 'scale(0.9) translateY(10px)';
    });
  }

  // Document click closes panel
  document.addEventListener('click', (e) => {
    if (panel && panel.style.display === 'block' && !container.contains(e.target)) {
      panel.style.display = 'none';
      panel.style.opacity = '0';
      panel.style.transform = 'scale(0.9) translateY(10px)';
    }
  });

  // Init mode
  window.updateAIOrbMode(currentMode);
}

// Call initialization
initAIOrb();


// ── Calm & Neuro Support Zone Controllers ──

// Web Audio Buffers
function createPinkNoiseBuffer(audioContext) {
  const bufferSize = 2 * audioContext.sampleRate;
  const noiseBuffer = audioContext.createBuffer(1, bufferSize, audioContext.sampleRate);
  const output = noiseBuffer.getChannelData(0);
  let b0, b1, b2, b3, b4, b5, b6;
  b0 = b1 = b2 = b3 = b4 = b5 = b6 = 0.0;
  for (let i = 0; i < bufferSize; i++) {
    const white = Math.random() * 2 - 1;
    b0 = 0.99886 * b0 + white * 0.0555179;
    b1 = 0.99332 * b1 + white * 0.0750759;
    b2 = 0.96900 * b2 + white * 0.1538520;
    b3 = 0.86650 * b3 + white * 0.3104856;
    b4 = 0.55000 * b4 + white * 0.5329522;
    b5 = -0.7616 * b5 - white * 0.0168980;
    output[i] = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362;
    output[i] *= 0.11;
    b6 = white * 0.115926;
  }
  return noiseBuffer;
}

let calmAudioCtx = null;
function getCalmAudioCtx() {
  if (!calmAudioCtx) {
    calmAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (calmAudioCtx.state === 'suspended') {
    calmAudioCtx.resume();
  }
  return calmAudioCtx;
}

class FireplaceSynth {
  constructor() {
    this.source = null;
    this.gainNode = null;
    this.filter = null;
    this.active = false;
    this.timeoutId = null;
  }
  start() {
    const ctx = getCalmAudioCtx();
    this.active = true;
    
    // Low rumble (lowpass filtered noise)
    this.gainNode = ctx.createGain();
    this.gainNode.gain.setValueAtTime(0.3, ctx.currentTime);
    
    this.filter = ctx.createBiquadFilter();
    this.filter.type = 'lowpass';
    this.filter.frequency.setValueAtTime(150, ctx.currentTime);
    
    this.source = ctx.createBufferSource();
    this.source.buffer = createPinkNoiseBuffer(ctx);
    this.source.loop = true;
    this.source.connect(this.filter);
    this.filter.connect(this.gainNode);
    this.gainNode.connect(ctx.destination);
    this.source.start(0);
    
    // Random crackle clicks
    const scheduleNextCrackle = () => {
      if (!this.active) return;
      const delay = Math.random() * 400 + 50; // frequent crackles
      this.timeoutId = setTimeout(() => {
        if (!this.active) return;
        this.playCrackle();
        scheduleNextCrackle();
      }, delay);
    };
    scheduleNextCrackle();
  }
  
  playCrackle() {
    try {
      const ctx = getCalmAudioCtx();
      const now = ctx.currentTime;
      
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      
      osc.type = Math.random() > 0.4 ? 'triangle' : 'sine';
      osc.frequency.setValueAtTime(Math.random() * 2000 + 1000, now);
      
      gain.gain.setValueAtTime(0, now);
      gain.gain.linearRampToValueAtTime(Math.random() * 0.04 + 0.01, now + 0.002);
      gain.gain.exponentialRampToValueAtTime(0.001, now + Math.random() * 0.05 + 0.01);
      
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + 0.08);
    } catch(e){}
  }
  
  stop() {
    this.active = false;
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
    if (this.source) {
      try { this.source.stop(); } catch(e){}
      this.source = null;
    }
  }
}

class CricketsSynth {
  constructor() {
    this.active = false;
    this.timeoutId = null;
    this.ambientNoise = null;
    this.ambientGain = null;
  }
  start() {
    const ctx = getCalmAudioCtx();
    this.active = true;
    this.ambientGain = ctx.createGain();
    this.ambientGain.gain.setValueAtTime(0.04, ctx.currentTime);
    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(1200, ctx.currentTime);
    this.ambientNoise = ctx.createBufferSource();
    this.ambientNoise.buffer = createPinkNoiseBuffer(ctx);
    this.ambientNoise.loop = true;
    this.ambientNoise.connect(filter);
    filter.connect(this.ambientGain);
    this.ambientGain.connect(ctx.destination);
    this.ambientNoise.start(0);
    const chirpLoop = () => {
      if (!this.active) return;
      this.playCricketChirp();
      const nextDelay = Math.random() * 800 + 700;
      this.timeoutId = setTimeout(chirpLoop, nextDelay);
    };
    chirpLoop();
  }
  playCricketChirp() {
    try {
      const ctx = getCalmAudioCtx();
      const now = ctx.currentTime;
      const pulseCount = 4;
      const pulseDur = 0.015;
      const pulseGap = 0.015;
      const gainNode = ctx.createGain();
      gainNode.gain.setValueAtTime(0, now);
      gainNode.connect(ctx.destination);
      const filter = ctx.createBiquadFilter();
      filter.type = 'bandpass';
      filter.frequency.setValueAtTime(4200, now);
      filter.Q.setValueAtTime(8, now);
      filter.connect(gainNode);
      for (let i = 0; i < pulseCount; i++) {
        const pStart = now + i * (pulseDur + pulseGap);
        const pEnd = pStart + pulseDur;
        const osc = ctx.createOscillator();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(4200 + Math.random() * 100, pStart);
        osc.connect(filter);
        gainNode.gain.setValueAtTime(0, pStart);
        gainNode.gain.linearRampToValueAtTime(0.03, pStart + 0.003);
        gainNode.gain.exponentialRampToValueAtTime(0.0001, pEnd);
        osc.start(pStart);
        osc.stop(pEnd + 0.01);
      }
    } catch(e){}
  }
  stop() {
    this.active = false;
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
    if (this.ambientNoise) {
      try { this.ambientNoise.stop(); } catch(e){}
      this.ambientNoise = null;
    }
  }
}

function createBrownNoiseBuffer(audioContext) {
  const bufferSize = 2 * audioContext.sampleRate;
  const noiseBuffer = audioContext.createBuffer(1, bufferSize, audioContext.sampleRate);
  const output = noiseBuffer.getChannelData(0);
  let lastOut = 0.0;
  for (let i = 0; i < bufferSize; i++) {
    const white = Math.random() * 2 - 1;
    output[i] = (lastOut + (0.02 * white)) / 1.02;
    lastOut = output[i];
    output[i] *= 3.5;
  }
  return noiseBuffer;
}

class BrownNoiseSynth {
  constructor() {
    this.source = null;
    this.gainNode = null;
  }
  start() {
    const ctx = getCalmAudioCtx();
    this.gainNode = ctx.createGain();
    this.gainNode.gain.setValueAtTime(0.4, ctx.currentTime);
    this.source = ctx.createBufferSource();
    this.source.buffer = createBrownNoiseBuffer(ctx);
    this.source.loop = true;
    this.source.connect(this.gainNode);
    this.gainNode.connect(ctx.destination);
    this.source.start(0);
  }
  stop() {
    if (this.source) {
      try { this.source.stop(); } catch(e){}
      this.source = null;
    }
  }
}

class WindSynth {
  constructor() {
    this.source = null;
    this.gainNode = null;
    this.filter = null;
    this.lfo = null;
  }
  start() {
    const ctx = getCalmAudioCtx();
    this.gainNode = ctx.createGain();
    this.gainNode.gain.setValueAtTime(0.35, ctx.currentTime);
    this.filter = ctx.createBiquadFilter();
    this.filter.type = 'bandpass';
    this.filter.frequency.setValueAtTime(350, ctx.currentTime);
    this.filter.Q.setValueAtTime(1.5, ctx.currentTime);
    this.source = ctx.createBufferSource();
    this.source.buffer = createPinkNoiseBuffer(ctx);
    this.source.loop = true;
    this.lfo = ctx.createOscillator();
    this.lfo.type = 'sine';
    this.lfo.frequency.setValueAtTime(0.07, ctx.currentTime);
    const lfoGain = ctx.createGain();
    lfoGain.gain.setValueAtTime(180, ctx.currentTime);
    this.lfo.connect(lfoGain);
    lfoGain.connect(this.filter.frequency);
    this.source.connect(this.filter);
    this.filter.connect(this.gainNode);
    this.gainNode.connect(ctx.destination);
    this.source.start(0);
    this.lfo.start(0);
  }
  stop() {
    if (this.source) {
      try { this.source.stop(); } catch(e){}
      this.source = null;
    }
    if (this.lfo) {
      try { this.lfo.stop(); } catch(e){}
      this.lfo = null;
    }
  }
}

class PianoSynth {
  constructor() {
    this.active = false;
    this.timeoutId = null;
  }
  start() {
    this.active = true;
    const noteLoop = () => {
      if (!this.active) return;
      this.playPianoNote();
      const delay = Math.random() * 2000 + 1500;
      this.timeoutId = setTimeout(noteLoop, delay);
    };
    noteLoop();
  }
  playPianoNote() {
    try {
      const ctx = getCalmAudioCtx();
      const now = ctx.currentTime;
      const notes = [277.18, 311.13, 349.23, 415.30, 466.16, 554.37, 622.25, 698.46];
      const freq = notes[Math.floor(Math.random() * notes.length)];
      const osc1 = ctx.createOscillator();
      const osc2 = ctx.createOscillator();
      const gainNode = ctx.createGain();
      osc1.type = 'triangle';
      osc1.frequency.setValueAtTime(freq, now);
      osc2.type = 'sine';
      osc2.frequency.setValueAtTime(freq * 2, now);
      const overtoneGain = ctx.createGain();
      overtoneGain.gain.setValueAtTime(0.2, now);
      gainNode.gain.setValueAtTime(0, now);
      gainNode.gain.linearRampToValueAtTime(0.08, now + 0.05);
      gainNode.gain.exponentialRampToValueAtTime(0.0001, now + 3.0);
      const filter = ctx.createBiquadFilter();
      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(800, now);
      osc1.connect(filter);
      osc2.connect(overtoneGain);
      overtoneGain.connect(filter);
      filter.connect(gainNode);
      gainNode.connect(ctx.destination);
      osc1.start(now);
      osc2.start(now);
      osc1.stop(now + 3.1);
      osc2.stop(now + 3.1);
    } catch(e){}
  }
  stop() {
    this.active = false;
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
  }
}

class TrainSynth {
  constructor() {
    this.active = false;
    this.timeoutId = null;
    this.oscRumble1 = null;
    this.oscRumble2 = null;
    this.rumbleGain = null;
  }
  start() {
    const ctx = getCalmAudioCtx();
    this.active = true;
    this.rumbleGain = ctx.createGain();
    this.rumbleGain.gain.setValueAtTime(0.08, ctx.currentTime);
    this.oscRumble1 = ctx.createOscillator();
    this.oscRumble1.type = 'sine';
    this.oscRumble1.frequency.setValueAtTime(55, ctx.currentTime);
    this.oscRumble2 = ctx.createOscillator();
    this.oscRumble2.type = 'sine';
    this.oscRumble2.frequency.setValueAtTime(110, ctx.currentTime);
    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(120, ctx.currentTime);
    this.oscRumble1.connect(filter);
    this.oscRumble2.connect(filter);
    filter.connect(this.rumbleGain);
    this.rumbleGain.connect(ctx.destination);
    this.oscRumble1.start(0);
    this.oscRumble2.start(0);
    const trackLoop = () => {
      if (!this.active) return;
      this.playTrackClack();
      this.timeoutId = setTimeout(trackLoop, 2200);
    };
    trackLoop();
  }
  playTrackClack() {
    try {
      const ctx = getCalmAudioCtx();
      const now = ctx.currentTime;
      const playPop = (time, vol) => {
        const noise = ctx.createBufferSource();
        noise.buffer = createPinkNoiseBuffer(ctx);
        const filter = ctx.createBiquadFilter();
        filter.type = 'bandpass';
        filter.frequency.setValueAtTime(180, time);
        filter.Q.setValueAtTime(3, time);
        const gain = ctx.createGain();
        gain.gain.setValueAtTime(0, time);
        gain.gain.linearRampToValueAtTime(vol, time + 0.005);
        gain.gain.exponentialRampToValueAtTime(0.001, time + 0.06);
        noise.connect(filter);
        filter.connect(gain);
        gain.connect(ctx.destination);
        noise.start(time);
        noise.stop(time + 0.08);
      };
      playPop(now, 0.05);
      playPop(now + 0.15, 0.04);
    } catch(e){}
  }
  stop() {
    this.active = false;
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
    if (this.oscRumble1) {
      try { this.oscRumble1.stop(); } catch(e){}
      this.oscRumble1 = null;
    }
    if (this.oscRumble2) {
      try { this.oscRumble2.stop(); } catch(e){}
      this.oscRumble2 = null;
    }
  }
}

class SpaceSynth {
  constructor() {
    this.osc1 = null;
    this.osc2 = null;
    this.lfo = null;
    this.gainNode = null;
  }
  start() {
    const ctx = getCalmAudioCtx();
    this.gainNode = ctx.createGain();
    this.gainNode.gain.setValueAtTime(0.12, ctx.currentTime);
    this.osc1 = ctx.createOscillator();
    this.osc1.type = 'sawtooth';
    this.osc1.frequency.setValueAtTime(65.41, ctx.currentTime);
    this.osc2 = ctx.createOscillator();
    this.osc2.type = 'sawtooth';
    this.osc2.frequency.setValueAtTime(65.71, ctx.currentTime);
    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(220, ctx.currentTime);
    filter.Q.setValueAtTime(5, ctx.currentTime);
    this.lfo = ctx.createOscillator();
    this.lfo.type = 'sine';
    this.lfo.frequency.setValueAtTime(0.05, ctx.currentTime);
    const lfoGain = ctx.createGain();
    lfoGain.gain.setValueAtTime(100, ctx.currentTime);
    this.lfo.connect(lfoGain);
    lfoGain.connect(filter.frequency);
    this.osc1.connect(filter);
    this.osc2.connect(filter);
    filter.connect(this.gainNode);
    this.gainNode.connect(ctx.destination);
    this.osc1.start(0);
    this.osc2.start(0);
    this.lfo.start(0);
  }
  stop() {
    if (this.osc1) {
      try { this.osc1.stop(); } catch(e){}
      this.osc1 = null;
    }
    if (this.osc2) {
      try { this.osc2.stop(); } catch(e){}
      this.osc2 = null;
    }
    if (this.lfo) {
      try { this.lfo.stop(); } catch(e){}
      this.lfo = null;
    }
  }
}

class RainSynth {
  constructor() {
    this.source = null;
    this.gainNode = null;
    this.filter = null;
    this.modulator = null;
  }
  start() {
    const ctx = getCalmAudioCtx();
    this.gainNode = ctx.createGain();
    this.gainNode.gain.setValueAtTime(0.4, ctx.currentTime);
    this.filter = ctx.createBiquadFilter();
    this.filter.type = 'lowpass';
    this.filter.frequency.setValueAtTime(800, ctx.currentTime);
    this.source = ctx.createBufferSource();
    this.source.buffer = createPinkNoiseBuffer(ctx);
    this.source.loop = true;
    this.modulator = ctx.createOscillator();
    this.modulator.type = 'sine';
    this.modulator.frequency.setValueAtTime(0.08, ctx.currentTime);
    const modGain = ctx.createGain();
    modGain.gain.setValueAtTime(250, ctx.currentTime);
    this.modulator.connect(modGain);
    modGain.connect(this.filter.frequency);
    this.source.connect(this.filter);
    this.filter.connect(this.gainNode);
    this.gainNode.connect(ctx.destination);
    this.source.start(0);
    this.modulator.start(0);
  }
  stop() {
    if (this.source) {
      try { this.source.stop(); } catch(e){}
      this.source = null;
    }
    if (this.modulator) {
      try { this.modulator.stop(); } catch(e){}
      this.modulator = null;
    }
  }
}

class OceanSynth {
  constructor() {
    this.source = null;
    this.gainNode = null;
    this.filter = null;
    this.modulator = null;
  }
  start() {
    const ctx = getCalmAudioCtx();
    this.gainNode = ctx.createGain();
    this.gainNode.gain.setValueAtTime(0, ctx.currentTime);
    this.filter = ctx.createBiquadFilter();
    this.filter.type = 'lowpass';
    this.filter.frequency.setValueAtTime(450, ctx.currentTime);
    this.source = ctx.createBufferSource();
    this.source.buffer = createPinkNoiseBuffer(ctx);
    this.source.loop = true;
    this.modulator = ctx.createOscillator();
    this.modulator.type = 'sine';
    this.modulator.frequency.setValueAtTime(0.15, ctx.currentTime);
    const modGain = ctx.createGain();
    modGain.gain.setValueAtTime(0.18, ctx.currentTime);
    const gainOffset = ctx.createGain();
    gainOffset.gain.setValueAtTime(0.2, ctx.currentTime);
    const filterModGain = ctx.createGain();
    filterModGain.gain.setValueAtTime(200, ctx.currentTime);
    this.modulator.connect(modGain);
    modGain.connect(this.gainNode.gain);
    this.modulator.connect(filterModGain);
    filterModGain.connect(this.filter.frequency);
    this.source.connect(this.filter);
    this.filter.connect(gainOffset);
    gainOffset.connect(ctx.destination);
    this.source.connect(this.filter);
    this.filter.connect(this.gainNode);
    this.gainNode.connect(ctx.destination);
    this.source.start(0);
    this.modulator.start(0);
  }
  stop() {
    if (this.source) {
      try { this.source.stop(); } catch(e){}
      this.source = null;
    }
    if (this.modulator) {
      try { this.modulator.stop(); } catch(e){}
      this.modulator = null;
    }
  }
}

class ForestSynth {
  constructor() {
    this.active = false;
    this.timer = null;
    this.ambientNoise = null;
    this.ambientGain = null;
  }
  start() {
    const ctx = getCalmAudioCtx();
    this.active = true;
    this.ambientGain = ctx.createGain();
    this.ambientGain.gain.setValueAtTime(0.08, ctx.currentTime);
    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(1000, ctx.currentTime);
    this.ambientNoise = ctx.createBufferSource();
    this.ambientNoise.buffer = createPinkNoiseBuffer(ctx);
    this.ambientNoise.loop = true;
    this.ambientNoise.connect(filter);
    filter.connect(this.ambientGain);
    this.ambientGain.connect(ctx.destination);
    this.ambientNoise.start(0);
    const scheduleNextChirp = () => {
      if (!this.active) return;
      const delay = Math.random() * 3000 + 1500;
      this.timer = setTimeout(() => {
        if (!this.active) return;
        this.playBirdChirp();
        scheduleNextChirp();
      }, delay);
    };
    scheduleNextChirp();
  }
  playBirdChirp() {
    try {
      const ctx = getCalmAudioCtx();
      const now = ctx.currentTime;
      const chirpType = Math.floor(Math.random() * 3);
      const gainNode = ctx.createGain();
      gainNode.gain.setValueAtTime(0, now);
      gainNode.connect(ctx.destination);
      if (chirpType === 0) {
        const baseFreq = Math.random() * 600 + 1200;
        const osc1 = ctx.createOscillator();
        osc1.type = 'sine';
        osc1.frequency.setValueAtTime(baseFreq, now);
        osc1.frequency.exponentialRampToValueAtTime(baseFreq * 1.5, now + 0.1);
        osc1.connect(gainNode);
        gainNode.gain.setValueAtTime(0, now);
        gainNode.gain.linearRampToValueAtTime(0.06, now + 0.02);
        gainNode.gain.exponentialRampToValueAtTime(0.001, now + 0.1);
        osc1.start(now);
        osc1.stop(now + 0.12);
        const osc2 = ctx.createOscillator();
        osc2.type = 'sine';
        osc2.frequency.setValueAtTime(baseFreq * 1.1, now + 0.15);
        osc2.frequency.exponentialRampToValueAtTime(baseFreq * 1.6, now + 0.25);
        osc2.connect(gainNode);
        gainNode.gain.setValueAtTime(0, now + 0.15);
        gainNode.gain.linearRampToValueAtTime(0.06, now + 0.17);
        gainNode.gain.exponentialRampToValueAtTime(0.001, now + 0.25);
        osc2.start(now + 0.15);
        osc2.stop(now + 0.27);
      } else if (chirpType === 1) {
        const baseFreq = Math.random() * 400 + 2000;
        const osc = ctx.createOscillator();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(baseFreq, now);
        const vibrato = ctx.createOscillator();
        vibrato.frequency.value = 25;
        const vibGain = ctx.createGain();
        vibGain.gain.value = 150;
        vibrato.connect(vibGain);
        vibGain.connect(osc.frequency);
        osc.connect(gainNode);
        gainNode.gain.setValueAtTime(0, now);
        gainNode.gain.linearRampToValueAtTime(0.04, now + 0.05);
        gainNode.gain.exponentialRampToValueAtTime(0.001, now + 0.4);
        vibrato.start(now);
        osc.start(now);
        vibrato.stop(now + 0.42);
        osc.stop(now + 0.42);
      } else {
        const baseFreq = Math.random() * 500 + 1500;
        const osc = ctx.createOscillator();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(baseFreq, now);
        osc.frequency.exponentialRampToValueAtTime(baseFreq * 0.7, now + 0.2);
        osc.connect(gainNode);
        gainNode.gain.setValueAtTime(0, now);
        gainNode.gain.linearRampToValueAtTime(0.05, now + 0.03);
        gainNode.gain.exponentialRampToValueAtTime(0.001, now + 0.2);
        osc.start(now);
        osc.stop(now + 0.22);
      }
    } catch(e) {
      console.warn("Failed to chirp bird:", e);
    }
  }
  stop() {
    this.active = false;
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    if (this.ambientNoise) {
      try { this.ambientNoise.stop(); } catch(e){}
      this.ambientNoise = null;
    }
  }
}

const synths = {
  rain: new RainSynth(),
  ocean: new OceanSynth(),
  forest: new ForestSynth(),
  fireplace: new FireplaceSynth(),
  crickets: new CricketsSynth(),
  wind: new WindSynth(),
  piano: new PianoSynth(),
  space: new SpaceSynth()
};

const playingStates = {
  rain: false,
  ocean: false,
  forest: false,
  fireplace: false,
  crickets: false,
  wind: false,
  piano: false,
  space: false
};

function handleCalmZoneSwitch(mode) {
  if (mode !== 'calmzone') {
    Object.keys(playingStates).forEach(key => {
      if (playingStates[key]) {
        synths[key].stop();
        playingStates[key] = false;
        const card = document.querySelector(`.cz-sound-card[data-sound="${key}"]`);
        if (card) {
          card.classList.remove('playing');
          const playBtn = card.querySelector('.cz-play-btn');
          if (playBtn) playBtn.textContent = 'Play';
        }
      }
    });
    deactivateAllWidgets();
  } else {
    const activeSubtab = document.querySelector('.cz-subtab.active');
    const activeCzTab = activeSubtab ? activeSubtab.dataset.cztab : 'sounds';
    activateWidget(activeCzTab);
  }
}

// Focus widgets state
let bubbleState = {
  active: false,
  targetX: 0,
  targetY: 0,
  currentX: 0,
  currentY: 0,
  animationId: null
};

let breathState = {
  timer: null,
  step: 0,
  active: false
};

let dotState = {
  active: false,
  cleared: 0,
  total: 5
};

function initBubbleFollow() {
  const area = document.getElementById('czWidgetBubble');
  const tracker = document.getElementById('czBubbleTracker');
  if (!area || !tracker) return;
  const setInitialPosition = () => {
    const rect = area.getBoundingClientRect();
    bubbleState.targetX = rect.width / 2;
    bubbleState.targetY = rect.height / 2;
    bubbleState.currentX = bubbleState.targetX;
    bubbleState.currentY = bubbleState.targetY;
  };
  setInitialPosition();
  area.addEventListener('mousemove', (e) => {
    const areaRect = area.getBoundingClientRect();
    bubbleState.targetX = e.clientX - areaRect.left;
    bubbleState.targetY = e.clientY - areaRect.top;
  });
  area.addEventListener('mouseleave', () => {
    const areaRect = area.getBoundingClientRect();
    bubbleState.targetX = areaRect.width / 2;
    bubbleState.targetY = areaRect.height / 2;
  });
  window.addEventListener('resize', setInitialPosition);
}

function updateBubblePhysics() {
  if (!bubbleState.active) return;
  const tracker = document.getElementById('czBubbleTracker');
  if (tracker) {
    const k = 0.12;
    const dx = bubbleState.targetX - bubbleState.currentX;
    const dy = bubbleState.targetY - bubbleState.currentY;
    bubbleState.currentX += dx * k;
    bubbleState.currentY += dy * k;
    tracker.style.left = `${bubbleState.currentX - 11}px`;
    tracker.style.top = `${bubbleState.currentY - 11}px`;
  }
  bubbleState.animationId = requestAnimationFrame(updateBubblePhysics);
}

function runBreathingGuide() {
  if (!breathState.active) return;
  const circle = document.getElementById('czBreathCircle');
  const label = document.getElementById('czBreathLabel');
  if (!circle || !label) return;
  const cycle = () => {
    if (!breathState.active) return;
    if (breathState.step === 0) {
      label.textContent = 'Inhale...';
      circle.className = 'cz-breath-circle inhale';
      breathState.step = 1;
      breathState.timer = setTimeout(cycle, 4000);
    } else if (breathState.step === 1) {
      label.textContent = 'Hold...';
      circle.className = 'cz-breath-circle inhale hold';
      breathState.step = 2;
      breathState.timer = setTimeout(cycle, 4000);
    } else {
      label.textContent = 'Exhale...';
      circle.className = 'cz-breath-circle';
      breathState.step = 0;
      breathState.timer = setTimeout(cycle, 4000);
    }
  };
  cycle();
}

function startDotTracker() {
  const container = document.getElementById('czDotsContainer');
  const counter = document.getElementById('czDotsCounter');
  if (!container || !counter) return;
  container.innerHTML = '';
  dotState.cleared = 0;
  counter.textContent = `Cleared: 0/${dotState.total}`;
  spawnNextDot();
}

function spawnNextDot() {
  if (!dotState.active) return;
  const container = document.getElementById('czDotsContainer');
  const counter = document.getElementById('czDotsCounter');
  if (!container) return;
  container.innerHTML = '';
  if (dotState.cleared >= dotState.total) {
    counter.textContent = `Excellent! Focus Restored ✨`;
    playSuccessSound();
    setTimeout(() => {
      if (dotState.active && dotState.cleared >= dotState.total) {
        startDotTracker();
      }
    }, 3000);
    return;
  }
  const dot = document.createElement('div');
  dot.className = 'cz-tracker-dot';
  const rect = container.getBoundingClientRect();
  const borderMargin = 20;
  const maxX = (rect.width || 280) - borderMargin * 2;
  const maxY = (rect.height || 150) - borderMargin * 2;
  const randomX = Math.random() * maxX + borderMargin;
  const randomY = Math.random() * maxY + borderMargin;
  dot.style.left = `${randomX}px`;
  dot.style.top = `${randomY}px`;
  dot.addEventListener('click', () => {
    dotState.cleared++;
    counter.textContent = `Cleared: ${dotState.cleared}/${dotState.total}`;
    playSingleQuickChime();
    spawnNextDot();
  });
  container.appendChild(dot);
}

function playSingleQuickChime() {
  try {
    const audioCtx = getCalmAudioCtx();
    const now = audioCtx.currentTime;
    const osc = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();
    osc.type = 'sine';
    const frequencies = [523.25, 587.33, 659.25, 783.99, 880.00, 1046.50];
    const freq = frequencies[dotState.cleared % frequencies.length];
    osc.frequency.setValueAtTime(freq, now);
    gainNode.gain.setValueAtTime(0.08, now);
    gainNode.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
    osc.connect(gainNode);
    gainNode.connect(audioCtx.destination);
    osc.start();
    osc.stop(now + 0.35);
  } catch(e){}
}

function deactivateAllWidgets() {
  bubbleState.active = false;
  if (bubbleState.animationId) {
    cancelAnimationFrame(bubbleState.animationId);
    bubbleState.animationId = null;
  }
  breathState.active = false;
  if (breathState.timer) {
    clearTimeout(breathState.timer);
    breathState.timer = null;
  }
  const circle = document.getElementById('czBreathCircle');
  if (circle) circle.className = 'cz-breath-circle';
  dotState.active = false;
  const container = document.getElementById('czDotsContainer');
  if (container) container.innerHTML = '';
}

function activateWidget(tab) {
  if (tab === 'reset') {
    const activeBtn = document.querySelector('.cz-reset-btn.active');
    const widgetType = activeBtn ? activeBtn.dataset.widget : 'bubble';
    deactivateAllWidgets();
    if (widgetType === 'bubble') {
      bubbleState.active = true;
      updateBubblePhysics();
    } else if (widgetType === 'breath') {
      breathState.active = true;
      breathState.step = 0;
      runBreathingGuide();
    } else if (widgetType === 'dots') {
      dotState.active = true;
      startDotTracker();
    }
  }
}

// Calm messages cycling
const CALM_MESSAGES = [
  { emoji: '🧘', text: "Welcome to the Calm Zone. Take a deep breath and relax." },
  { emoji: '✨', text: "There is no rush. Focus on one small step at a time." },
  { emoji: '🍃', text: "You are in control. Everything is going to be okay." },
  { emoji: '👀', text: "Let your eyes relax for a brief moment." },
  { emoji: '🌊', text: "Soft light, calm mind. You are doing great." },
  { emoji: '🌱', text: "It is okay to pause. Reset whenever you need." },
  { emoji: '🌸', text: "Take a slow, deep breath in... and let it out." }
];

let calmMessageIndex = 0;
function cycleCalmMessage() {
  const emojiEl = document.getElementById('czSupportEmoji');
  const msgEl = document.getElementById('czSupportMsg');
  if (!emojiEl || !msgEl) return;
  emojiEl.style.transition = 'opacity 0.5s ease';
  msgEl.style.transition = 'opacity 0.5s ease';
  emojiEl.style.opacity = '0';
  msgEl.style.opacity = '0';
  setTimeout(() => {
    calmMessageIndex = (calmMessageIndex + 1) % CALM_MESSAGES.length;
    const nextMsg = CALM_MESSAGES[calmMessageIndex];
    emojiEl.textContent = nextMsg.emoji;
    msgEl.textContent = nextMsg.text;
    emojiEl.style.opacity = '1';
    msgEl.style.opacity = '1';
  }, 500);
}

function initSoundscapes() {
  document.querySelectorAll('.cz-sound-card').forEach(card => {
    const soundKey = card.dataset.sound;
    const playBtn = card.querySelector('.cz-play-btn');
    if (!soundKey || !playBtn) return;
    
    playBtn.addEventListener('click', () => {
      Object.keys(playingStates).forEach(key => {
        if (key !== soundKey && playingStates[key]) {
          synths[key].stop();
          playingStates[key] = false;
          const otherCard = document.querySelector(`.cz-sound-card[data-sound="${key}"]`);
          if (otherCard) {
            otherCard.classList.remove('playing');
            const otherBtn = otherCard.querySelector('.cz-play-btn');
            if (otherBtn) otherBtn.textContent = 'Play';
          }
        }
      });
      
      const isPlaying = playingStates[soundKey];
      if (isPlaying) {
        synths[soundKey].stop();
        playingStates[soundKey] = false;
        card.classList.remove('playing');
        playBtn.textContent = 'Play';
      } else {
        try {
          synths[soundKey].start();
          playingStates[soundKey] = true;
          card.classList.add('playing');
          playBtn.textContent = 'Pause';
        } catch(e) {
          console.error("Failed to start synth:", e);
        }
      }
    });
  });
}

function initFocusReset() {
  const controls = document.getElementById('czResetControls');
  if (!controls) return;
  
  controls.addEventListener('click', (e) => {
    const btn = e.target.closest('.cz-reset-btn');
    if (!btn) return;
    
    const widgetType = btn.dataset.widget;
    if (!widgetType) return;
    
    controls.querySelectorAll('.cz-reset-btn').forEach(b => {
      b.classList.toggle('active', b === btn);
    });
    
    document.getElementById('czWidgetBubble').style.display = widgetType === 'bubble' ? '' : 'none';
    document.getElementById('czWidgetBreath').style.display = widgetType === 'breath' ? '' : 'none';
    document.getElementById('czWidgetDots').style.display = widgetType === 'dots' ? '' : 'none';
    
    deactivateAllWidgets();
    if (widgetType === 'bubble') {
      bubbleState.active = true;
      updateBubblePhysics();
    } else if (widgetType === 'breath') {
      breathState.active = true;
      breathState.step = 0;
      runBreathingGuide();
    } else if (widgetType === 'dots') {
      dotState.active = true;
      startDotTracker();
    }
  });
  
  initBubbleFollow();
}

function initCalmZone() {
  setInterval(cycleCalmMessage, 8000);

  const tabsContainer = document.getElementById('czSubtabs');
  if (tabsContainer) {
    tabsContainer.addEventListener('click', (e) => {
      const tabBtn = e.target.closest('.cz-subtab');
      if (!tabBtn) return;
      const targetTab = tabBtn.dataset.cztab;
      if (!targetTab) return;
      tabsContainer.querySelectorAll('.cz-subtab').forEach(btn => {
        btn.classList.toggle('active', btn === tabBtn);
      });
      document.querySelectorAll('.cz-tab-content').forEach(content => {
        const active = content.dataset.cztab === targetTab;
        content.classList.toggle('active', active);
        if (active) {
          activateWidget(targetTab);
        } else {
          if (content.dataset.cztab === 'reset') {
            deactivateAllWidgets();
          }
        }
      });
    });
  }

  const czCalmModeToggle = document.getElementById('czCalmModeToggle');
  if (czCalmModeToggle) {
    const updateCalmToggleBtnText = () => {
      const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
      if (currentTheme === 'calm') {
        czCalmModeToggle.textContent = 'Deactivate';
        czCalmModeToggle.classList.add('active');
      } else {
        czCalmModeToggle.textContent = 'Activate';
        czCalmModeToggle.classList.remove('active');
      }
    };
    czCalmModeToggle.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
      let targetTheme = 'calm';
      if (currentTheme === 'calm') {
        targetTheme = localStorage.getItem('neurolens-prev-theme') || 'light';
        if (targetTheme === 'calm') targetTheme = 'light';
      } else {
        localStorage.setItem('neurolens-prev-theme', currentTheme);
      }
      document.documentElement.setAttribute('data-theme', targetTheme);
      localStorage.setItem('neurolens-theme', targetTheme);
      const themeChips = document.querySelectorAll('.theme-chip');
      themeChips.forEach(c => {
        c.classList.toggle('active', c.dataset.theme === targetTheme);
      });
      updateCalmToggleBtnText();
    });
    const observer = new MutationObserver(() => {
      updateCalmToggleBtnText();
      if (window.updateAIOrbMode && currentMode) {
        window.updateAIOrbMode(currentMode);
      }
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    updateCalmToggleBtnText();
  }

  initSoundscapes();

  const czVideoModal = document.getElementById('czVideoModal');
  const czVideoModalIframe = document.getElementById('czVideoModalIframe');
  const czVideoModalTitle = document.getElementById('czVideoModalTitle');
  const czVideoModalClose = document.getElementById('czVideoModalClose');
  if (czVideoModal && czVideoModalIframe) {
    document.querySelectorAll('.cz-video-card').forEach(card => {
      card.addEventListener('click', () => {
        const videoId = card.dataset.videoId;
        const title = card.dataset.videoTitle || 'Relaxing Video';
        if (!videoId) return;
        czVideoModalTitle.textContent = title;
        czVideoModalIframe.src = `https://www.youtube.com/embed/${videoId}?autoplay=1&enablejsapi=1`;
        czVideoModal.classList.add('active');
      });
    });
    const closeModal = () => {
      czVideoModal.classList.remove('active');
      setTimeout(() => {
        czVideoModalIframe.src = '';
      }, 300);
    };
    if (czVideoModalClose) {
      czVideoModalClose.addEventListener('click', closeModal);
    }
    czVideoModal.addEventListener('click', (e) => {
      if (e.target === czVideoModal) {
        closeModal();
      }
    });
  }

  initFocusReset();
}

window.handleCalmZoneSwitch = handleCalmZoneSwitch;
window.deactivateAllWidgets = deactivateAllWidgets;
window.activateWidget = activateWidget;

initCalmZone();