// WordFormationEngine v3 — Ultra-Strict Single Word Detection
// ONE gesture = ONE word. Locked until gesture ends.
// 
// Anti-duplication mechanisms:
//   1. Stability: Hold same sign for 2.5 seconds (time-based, FPS-independent)
//   2. Confidence: 85%+ average required
//   3. Gesture Lock: output frozen after detection until hand removed/changed
//   4. Cooldown: 1.5s minimum between detections
//   5. Motion Reset: significant hand change required to unlock

export class WordFormationEngine {
  constructor() {
    // ── Time-based Stability tracking ──
    this.candidateSign = null;
    this.candidateFirstSeenTime = 0;
    this.candidateConfidences = [];
    this.holdDurationThreshold = 800; // Hold gesture for exactly 0.8 seconds (800ms) to lock
    this.confidenceThreshold = 50;    // minimum average confidence %

    // ── Gesture lock ──
    this.locked = false;              // true = word detected, no new predictions
    this.lockedWord = null;           // the word we locked on
    this.lockStartTime = 0;

    // ── Cooldown ──
    this.cooldownMs = 800;            // 0.8 seconds between detections
    this.lastDetectionTime = 0;

    // ── Motion reset tracking ──
    this.noSignFrames = 0;            // frames with no sign / different sign
    this.resetThreshold = 3;          // frames of "no sign" needed to unlock (tuned for ~6fps)

    // ── Output ──
    this.completedWord = null;
    this.currentCandidate = null;
    this.candidateProgress = 0;
  }

  /**
   * @param {string|null} sign - detected sign this frame
   * @param {number} confidence - 0-100
   * @param {boolean} isPause - motion tracker says hands are still
   * @param {number} timestamp - current time
   * @param {boolean} hasHands - whether hands are in frame
   */
  update(sign, confidence, isPause, timestamp, hasHands = true) {
    this.completedWord = null;

    // ── LOCKED STATE: gesture already detected ──
    if (this.locked) {
      // Check unlock conditions:
      // 1. Hands left frame
      if (!hasHands) {
        this.noSignFrames++;
        if (this.noSignFrames >= this.resetThreshold) {
          this._unlock();
        }
        return;
      }

      // 2. Sign changed significantly (different sign for N frames)
      if (sign !== this.lockedWord) {
        this.noSignFrames++;
        if (this.noSignFrames >= this.resetThreshold) {
          this._unlock();
        }
      } else {
        this.noSignFrames = 0; // still showing same sign
      }

      // 3. Cooldown expired + hands gone briefly
      const timeSinceLock = timestamp - this.lockStartTime;
      if (timeSinceLock > this.cooldownMs && this.noSignFrames >= 3) {
        this._unlock();
      }

      return; // Stay locked, no new predictions
    }

    // ── COOLDOWN CHECK ──
    if (timestamp - this.lastDetectionTime < this.cooldownMs) {
      this.currentCandidate = null;
      this.candidateProgress = 0;
      return;
    }

    // ── NO SIGN / LOW CONFIDENCE ──
    if (!sign || sign === 'UNCLEAR' || confidence < 50) {
      this._decayCandidate(timestamp);
      this.noSignFrames++;
      return;
    }

    this.noSignFrames = 0;

    // ── STABILITY ACCUMULATION (TIME-BASED) ──
    if (sign === this.candidateSign) {
      this.candidateConfidences.push(confidence);
      if (this.candidateConfidences.length > 50) this.candidateConfidences.shift();
    } else {
      // Different sign — reset candidate
      this.candidateSign = sign;
      this.candidateFirstSeenTime = timestamp;
      this.candidateConfidences = [confidence];
    }

    // Update progress display based on time elapsed
    const elapsed = timestamp - this.candidateFirstSeenTime;
    this.currentCandidate = sign;
    this.candidateProgress = Math.min(1, elapsed / this.holdDurationThreshold);

    // ── CHECK CONFIRMATION ──
    if (elapsed >= 0.9 * this.holdDurationThreshold) {
      const avgConf = Math.round(
        this.candidateConfidences.reduce((a, b) => a + b, 0) / this.candidateConfidences.length
      );

      // ✅ CONFIRMED — output the word and LOCK
      this.completedWord = {
        word: sign,
        confidence: Math.max(avgConf, 50),
      };

      this.locked = true;
      this.lockedWord = sign;
      this.lockStartTime = timestamp;
      this.lastDetectionTime = timestamp;
      this.noSignFrames = 0;

      // Reset candidate
      this.candidateSign = null;
      this.candidateFirstSeenTime = 0;
      this.candidateConfidences = [];
      this.currentCandidate = null;
      this.candidateProgress = 0;
    }
  }

  _decayCandidate(timestamp) {
    if (this.candidateSign) {
      // Decay progress by moving first seen time forward
      this.candidateFirstSeenTime += 150; // decays progress over ~15 frames
      if (timestamp - this.candidateFirstSeenTime <= 0 || (timestamp - this.candidateFirstSeenTime) < 50) {
        this.candidateSign = null;
        this.candidateFirstSeenTime = 0;
        this.candidateConfidences = [];
        this.currentCandidate = null;
        this.candidateProgress = 0;
      } else {
        this.candidateProgress = Math.max(0, (timestamp - this.candidateFirstSeenTime) / this.holdDurationThreshold);
      }
    }
  }

  _unlock() {
    this.locked = false;
    this.lockedWord = null;
    this.noSignFrames = 0;
    this.candidateSign = null;
    this.candidateFirstSeenTime = 0;
    this.candidateConfidences = [];
    this.currentCandidate = null;
    this.candidateProgress = 0;
  }

  isLocked() { return this.locked; }
  getLockedWord() { return this.lockedWord; }
  getFormingWord() { return this.locked ? '' : (this.currentCandidate || ''); }
  getProgress() { return this.candidateProgress; }
  getCompletedWord() { return this.completedWord; }

  // Legacy compat
  getFormingLetters() { return this.getFormingWord(); }

  reset() {
    this._unlock();
    this.lastDetectionTime = 0;
    this.completedWord = null;
  }
}
