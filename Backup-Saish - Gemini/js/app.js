/**
 * NeuroSense AI — Main Application
 * Orchestrates webcam capture, MediaPipe models, analysis modules, and UI rendering.
 */

import { FaceLandmarker, HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { EmotionAnalyzer } from './emotion-analyzer.js';
import { FocusAnalyzer } from './focus-analyzer.js';
import { SignLanguageAnalyzer } from './sign-language-analyzer.js';
import { VisualQualityAnalyzer } from './visual-quality.js';
import { NeuroLensAnalyzer } from './neurolens-analyzer.js';

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
};

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

const emotionAnalyzer = new EmotionAnalyzer();
const focusAnalyzer = new FocusAnalyzer();
const signAnalyzer = new SignLanguageAnalyzer();
const qualityAnalyzer = new VisualQualityAnalyzer();
const neurolensAnalyzer = new NeuroLensAnalyzer();

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
    } else {
      panel.style.display = 'none';
    }
  });
  const modeNames = { emotion: 'Emotion Detection', focus: 'Focus Analysis', sign: 'Sign Language', neurolens: 'NeuroLens Detection' };
  if (DOM.statusDot.classList.contains('active')) {
    DOM.statusText.textContent = modeNames[mode] || 'Live Analysis';
  }
  // Lazy-load NeuroLens model when tab is first selected
  if (mode === 'neurolens' && !neurolensAnalyzer.isLoaded && !neurolensLoading) {
    neurolensLoading = true;
    DOM.nlAlertSub.textContent = 'Loading COCO-SSD model...';
    neurolensAnalyzer.loadModel().then(() => {
      neurolensLoading = false;
      DOM.nlAlertSub.textContent = 'Model ready — scanning...';
      if (DOM.obstacleBadge) {
        DOM.obstacleBadge.className = 'panel-badge active';
        DOM.obstacleBadge.textContent = 'LIVE';
      }
    }).catch(err => {
      neurolensLoading = false;
      DOM.nlAlertSub.textContent = `Model load failed: ${err.message}`;
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

// ── Start Button ──
DOM.startBtn.addEventListener('click', async () => {
  DOM.startBtn.disabled = true;
  DOM.startBtn.textContent = 'Starting…';
  DOM.errorMsg.style.display = 'none';

  try {
    await startCamera();
  } catch (err) {
    DOM.errorMsg.textContent = `Camera error: ${err.message}`;
    DOM.errorMsg.style.display = 'block';
    DOM.startBtn.disabled = false;
    DOM.startBtn.textContent = 'Retry';
  }
});

// ── Camera Setup ──
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
    audio: false,
  });

  DOM.webcam.srcObject = stream;
  await new Promise(r => { DOM.webcam.onloadedmetadata = r; });
  await DOM.webcam.play();

  // Set canvas sizes
  const vw = DOM.webcam.videoWidth;
  const vh = DOM.webcam.videoHeight;
  DOM.canvas.width = vw;
  DOM.canvas.height = vh;
  analysisCanvas.width = vw;
  analysisCanvas.height = vh;
  DOM.resolution.textContent = `${vw}×${vh}`;

  // Hide permission screen, show loading
  DOM.permissionScreen.classList.add('hidden');
  DOM.loadingText.innerHTML = 'Loading <span>MediaPipe</span> models…';

  await initModels();

  DOM.loadingOverlay.classList.add('hidden');
  DOM.statusDot.classList.add('active');

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

// ── Model Initialization ──
async function initModels() {
  DOM.loadingText.innerHTML = 'Loading <span>vision runtime</span>…';
  const filesetResolver = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
  );

  DOM.loadingText.innerHTML = 'Loading <span>Face Landmarker</span>…';
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

  DOM.loadingText.innerHTML = 'Loading <span>Hand Landmarker</span>…';
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
    [0,1],[1,2],[2,3],[3,4],       // thumb
    [0,5],[5,6],[6,7],[7,8],       // index
    [0,9],[9,10],[10,11],[11,12],  // middle
    [0,13],[13,14],[14,15],[15,16],// ring
    [0,17],[17,18],[18,19],[19,20],// pinky
    [5,9],[9,13],[13,17],          // palm
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
const FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10];
const LEFT_EYE = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7,33];
const RIGHT_EYE = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382,362];
const LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,61];
const LEFT_EYEBROW = [70,63,105,66,107,55,65,52,53,46,70];
const RIGHT_EYEBROW = [300,293,334,296,336,285,295,282,283,276,300];
