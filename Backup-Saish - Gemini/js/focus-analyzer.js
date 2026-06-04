/**
 * Focus Analyzer — determines cognitive attention state from eye, gaze, and head signals.
 * Uses blink rate, gaze stability, and head drift to classify focus level.
 */

const FOCUS_STATES = ['Deep Focus', 'Focused', 'Partially Focused', 'Distracted', 'Drowsy', 'Disengaged', 'Microsleep Detected'];

export class FocusAnalyzer {
  constructor() {
    this.blinkEvents = [];       // timestamps of blinks
    this.gazeHistory = [];       // recent gaze positions
    this.headPoseHistory = [];   // recent head positions (nose landmark)
    this.maxHistory = 90;        // ~3 seconds at 30fps
    this.blinkWindow = 60000;    // 60s window for blink rate
    this.blinkThreshold = 0.45;  // blendshape value to count as blink (lowered to catch partial blinks)
    this.wasBlinking = false;

    // Rolling-window state smoothing (replaces majority-vote counters)
    this._stateWindow = [];      // last 30 raw state readings
    this._stateWindowSize = 30;

    // Drowsiness detection state
    this._eyeOpennessHistory = [];  // { avgOpen, t } entries
    this._drowsinessThreshold = 0.4;
    this._drowsinessSeconds = 2;

    // Microsleep detection state
    this._eyesClosedSince = null;   // timestamp when eyes first closed
    this._microsleepThreshold = 2000; // 2 seconds in ms

    // Focus streak tracking
    this._focusStreakStart = null;   // timestamp when current focus streak began
    this._focusStreakSeconds = 0;
  }

  /**
   * @param {Array} blendshapes — face blendshapes
   * @param {Array} landmarks — 478 face landmarks (normalized 0-1)
   * @param {number} timestamp — current time in ms
   */
  analyze(blendshapes, landmarks, timestamp) {
    if (!blendshapes || !landmarks || landmarks.length < 468) {
      this._clearStateWindow();
      this._eyesClosedSince = null;
      this._focusStreakStart = null;
      this._focusStreakSeconds = 0;
      return this._noDetection();
    }

    const bs = {};
    for (const b of blendshapes) {
      bs[b.categoryName] = b.score;
    }

    // ── Blink Detection ──
    const blinkL = bs.eyeBlinkLeft || 0;
    const blinkR = bs.eyeBlinkRight || 0;
    const isBlinking = (blinkL + blinkR) / 2 > this.blinkThreshold;

    if (isBlinking && !this.wasBlinking) {
      this.blinkEvents.push(timestamp);
    }
    this.wasBlinking = isBlinking;

    // Clean old blinks
    const cutoff = timestamp - this.blinkWindow;
    this.blinkEvents = this.blinkEvents.filter(t => t > cutoff);

    // Blink rate (per minute)
    const elapsed = Math.min(this.blinkWindow, timestamp) / 60000;
    const blinkRate = elapsed > 0.1 ? this.blinkEvents.length / elapsed : 0;

    // ── Gaze Tracking ──
    // Use eye look blendshapes for gaze direction
    const gazeX = (bs.eyeLookOutLeft || 0) - (bs.eyeLookInLeft || 0);
    const gazeY = (bs.eyeLookUpLeft || 0) - (bs.eyeLookDownLeft || 0);

    this.gazeHistory.push({ x: gazeX, y: gazeY, t: timestamp });
    if (this.gazeHistory.length > this.maxHistory) this.gazeHistory.shift();

    // Gaze variance (stability)
    const gazeVariance = this._computeVariance(this.gazeHistory);

    // ── Head Pose Drift ──
    // Use nose tip landmark (index 1)
    const noseTip = landmarks[1];
    if (noseTip) {
      this.headPoseHistory.push({ x: noseTip.x, y: noseTip.y, t: timestamp });
      if (this.headPoseHistory.length > this.maxHistory) this.headPoseHistory.shift();
    }

    const headDrift = this._computeVariance(this.headPoseHistory);

    // ── Eye Openness ──
    const eyeOpenL = 1 - (bs.eyeBlinkLeft || 0);
    const eyeOpenR = 1 - (bs.eyeBlinkRight || 0);
    const avgEyeOpen = (eyeOpenL + eyeOpenR) / 2;

    // ── Classification ──
    let score = 0; // 0 = Disengaged, 100 = Deep Focus
    const evidenceParts = [];

    // Blink rate scoring
    // Normal: 15-20/min, focused: 3-10/min, drowsy/distracted: 20+/min
    if (blinkRate < 8) {
      score += 30;
      evidenceParts.push(`Low blink rate (${blinkRate.toFixed(0)}/min) suggesting intense concentration`);
    } else if (blinkRate < 18) {
      score += 20;
      evidenceParts.push(`Normal blink rate (${blinkRate.toFixed(0)}/min)`);
    } else if (blinkRate < 28) {
      score += 8;
      evidenceParts.push(`Elevated blink rate (${blinkRate.toFixed(0)}/min) suggesting reduced focus`);
    } else {
      score += 0;
      evidenceParts.push(`High blink rate (${blinkRate.toFixed(0)}/min) indicating distraction or fatigue`);
    }

    // Gaze stability scoring
    if (gazeVariance < 0.002) {
      score += 30;
      evidenceParts.push(`Very stable gaze fixation (var: ${gazeVariance.toFixed(4)})`);
    } else if (gazeVariance < 0.008) {
      score += 22;
      evidenceParts.push(`Stable gaze with minor saccades (var: ${gazeVariance.toFixed(4)})`);
    } else if (gazeVariance < 0.02) {
      score += 10;
      evidenceParts.push(`Moderate gaze shifts detected (var: ${gazeVariance.toFixed(4)})`);
    } else {
      score += 2;
      evidenceParts.push(`Frequent gaze shifts (var: ${gazeVariance.toFixed(4)}) indicating wandering attention`);
    }

    // Head pose stability scoring
    if (headDrift < 0.0005) {
      score += 25;
      evidenceParts.push(`Head position very steady`);
    } else if (headDrift < 0.002) {
      score += 18;
      evidenceParts.push(`Minimal head movement`);
    } else if (headDrift < 0.006) {
      score += 8;
      evidenceParts.push(`Noticeable head movement detected`);
    } else {
      score += 0;
      evidenceParts.push(`Significant head drift observed`);
    }

    // Eye openness scoring
    if (avgEyeOpen > 0.8) {
      score += 15;
    } else if (avgEyeOpen > 0.5) {
      score += 10;
    } else if (avgEyeOpen > 0.2) {
      score += 3;
      evidenceParts.push(`Partially closed eyes detected`);
    } else {
      score += 0;
      evidenceParts.push(`Eyes appear closed`);
    }

    // Map score to state
    let state;
    if (score >= 85) state = 'Deep Focus';
    else if (score >= 65) state = 'Focused';
    else if (score >= 45) state = 'Partially Focused';
    else if (score >= 25) state = 'Distracted';
    else state = 'Disengaged';

    // ── Microsleep Detection (overrides everything) ──
    const eyesClosed = avgEyeOpen < 0.15;
    if (eyesClosed) {
      if (this._eyesClosedSince === null) {
        this._eyesClosedSince = timestamp;
      }
      const closedDuration = (timestamp - this._eyesClosedSince) / 1000;
      if (closedDuration >= this._microsleepThreshold / 1000) {
        state = 'Microsleep Detected';
        evidenceParts.length = 0; // clear previous evidence
        evidenceParts.push(`Eyes closed for ${closedDuration.toFixed(1)}s — possible microsleep`);
      }
    } else {
      this._eyesClosedSince = null;
    }

    // ── Drowsiness Detection (overrides score-based state, but not microsleep) ──
    this._eyeOpennessHistory.push({ avgOpen: avgEyeOpen, t: timestamp });
    const drowsyCutoff = timestamp - (this._drowsinessSeconds * 1000);
    this._eyeOpennessHistory = this._eyeOpennessHistory.filter(e => e.t > drowsyCutoff);

    if (state !== 'Microsleep Detected' && this._eyeOpennessHistory.length > 0) {
      const avgOpenRecent = this._eyeOpennessHistory.reduce((s, e) => s + e.avgOpen, 0)
        / this._eyeOpennessHistory.length;
      const windowSpan = (timestamp - this._eyeOpennessHistory[0].t) / 1000;

      if (avgOpenRecent < this._drowsinessThreshold
          && blinkRate > 25
          && windowSpan >= this._drowsinessSeconds) {
        state = 'Drowsy';
        evidenceParts.push(`Drowsiness detected: avg eye openness ${avgOpenRecent.toFixed(2)}, blink rate ${blinkRate.toFixed(0)}/min`);
      }
    }

    // ── Rolling-window state smoothing ──
    this._stateWindow.push(state);
    if (this._stateWindow.length > this._stateWindowSize) {
      this._stateWindow.shift();
    }
    const smoothedState = this._getMostFrequent(this._stateWindow);

    // ── Focus Streak Tracking ──
    const isFocused = smoothedState === 'Deep Focus' || smoothedState === 'Focused';
    if (isFocused) {
      if (this._focusStreakStart === null) {
        this._focusStreakStart = timestamp;
      }
      this._focusStreakSeconds = (timestamp - this._focusStreakStart) / 1000;
    } else {
      this._focusStreakStart = null;
      this._focusStreakSeconds = 0;
    }

    // Confidence
    let confidence = Math.min(92, Math.max(20, Math.round(score * 0.95)));
    if (this.gazeHistory.length < 15) {
      confidence = Math.min(confidence, 50);
      evidenceParts.push('Insufficient temporal data for high confidence');
    }

    return {
      state: smoothedState,
      confidence: `${confidence}%`,
      evidence: evidenceParts.join('. ') + '.',
      focusStreakSeconds: Math.round(this._focusStreakSeconds),
      _score: score,
      _blinkRate: blinkRate,
      _gazeVariance: gazeVariance,
      _headDrift: headDrift,
    };
  }

  _computeVariance(history) {
    if (history.length < 3) return 0;
    const recent = history.slice(-30);
    const meanX = recent.reduce((s, p) => s + p.x, 0) / recent.length;
    const meanY = recent.reduce((s, p) => s + p.y, 0) / recent.length;
    const variance = recent.reduce((s, p) => {
      return s + (p.x - meanX) ** 2 + (p.y - meanY) ** 2;
    }, 0) / recent.length;
    return variance;
  }

  _noDetection() {
    return {
      state: 'Not reliably detectable',
      confidence: '0%',
      evidence: 'Insufficient facial landmark data for focus analysis.',
      focusStreakSeconds: 0,
      _score: 0,
      _blinkRate: 0,
      _gazeVariance: 0,
      _headDrift: 0,
    };
  }

  /** Clear rolling state window (called when face is lost) */
  _clearStateWindow() {
    this._stateWindow = [];
  }

  /** Return the most frequent value in an array */
  _getMostFrequent(arr) {
    const counts = {};
    for (const v of arr) {
      counts[v] = (counts[v] || 0) + 1;
    }
    let best = arr[arr.length - 1]; // fallback to latest
    let bestCount = 0;
    for (const [val, count] of Object.entries(counts)) {
      if (count > bestCount) {
        bestCount = count;
        best = val;
      }
    }
    return best;
  }
}
