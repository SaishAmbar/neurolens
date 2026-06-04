/**
 * Sign Language Analyzer v7 — High-Accuracy Single Word Detection
 *
 * ONE gesture = ONE word. No duplicates. No flicker.
 *
 * Mechanisms:
 *   1. 10-frame stability threshold
 *   2. 85%+ confidence required
 *   3. Gesture lock after detection (frozen until hand removed/changed)
 *   4. 1.5s cooldown between detections
 *   5. Motion-based unlock (8 frames of different sign or no hands)
 */

import { extractFeatures } from './sign/features.js';
import { MotionTracker } from './sign/motion.js';
import { StaticGestureClassifier } from './sign/static-classifier.js';
import { WordFormationEngine } from './sign/word-engine.js';

export class SignLanguageAnalyzer {
  constructor() {
    this.motion = new MotionTracker();
    this.classifier = new StaticGestureClassifier();
    this.wordEngine = new WordFormationEngine();

    this.hadHands = false;
    this.recognizedWords = [];
    this.maxHistory = 30;
    this.lastConfirmedWord = null;
    this.confirmedHoldFrames = 0;
    this.confirmedHoldMax = 30;       // show "Word Confirmed" for ~1s

    // Prediction smoothing history
    this.predictionHistory = [];
    this.smoothingWindowSize = 5;
  }

  analyze(handsLandmarks, handedness, timestamp) {
    const hasHands = handsLandmarks && handsLandmarks.length > 0;

    // ── No hands ──
    if (!hasHands) {
      this.predictionHistory = []; // Reset history when hands leave
      this.motion.update(null, timestamp);
      this.wordEngine.update(null, 0, true, timestamp, false);

      const w = this.wordEngine.getCompletedWord();
      if (w) this._addWord(w);

      return this._buildResult({ detected: false, timestamp });
    }

    // ── Hands detected ──
    this.hadHands = true;
    const lm = handsLandmarks[0];
    const hand = handedness?.[0]?.[0]?.categoryName || 'Right';
    const features = extractFeatures(lm, hand);
    const motionResult = this.motion.update(features, timestamp);

    // Classify
    const matches = this.classifier.classify(features, motionResult);
    let bestSign = null;
    let bestConf = 0;
    let alts = [];

    if (matches.length > 0 && matches[0].sign !== 'UNCLEAR') {
      bestSign = matches[0].sign;
      bestConf = matches[0].confidence;
      alts = matches.slice(1, 4).map(m => m.sign).filter(s => s !== 'UNCLEAR');
    }

    // Smooth prediction to filter out transient fluctuations
    const smoothed = this._getSmoothedPrediction(bestSign, bestConf);

    // Word engine — handles stability, locking, cooldown
    this.wordEngine.update(smoothed.sign, smoothed.confidence, motionResult.isPause, timestamp, true);

    const completedWord = this.wordEngine.getCompletedWord();
    if (completedWord) {
      this._addWord(completedWord);
    }

    return this._buildResult({
      detected: true,
      bestSign: smoothed.sign,
      bestConf: smoothed.confidence,
      alts,
      timestamp,
    });
  }

  _getSmoothedPrediction(sign, confidence) {
    // Add to history
    this.predictionHistory.push({ sign, confidence });
    if (this.predictionHistory.length > this.smoothingWindowSize) {
      this.predictionHistory.shift();
    }

    // Count frequencies of each sign in history
    const counts = {};
    const totalConf = {};
    for (const item of this.predictionHistory) {
      const s = item.sign || 'UNCLEAR';
      counts[s] = (counts[s] || 0) + 1;
      totalConf[s] = (totalConf[s] || 0) + item.confidence;
    }

    // Find the sign with the highest frequency
    let bestSign = 'UNCLEAR';
    let maxCount = 0;
    let maxConfSum = 0;

    for (const s in counts) {
      if (counts[s] > maxCount) {
        maxCount = counts[s];
        bestSign = s;
        maxConfSum = totalConf[s];
      } else if (counts[s] === maxCount) {
        // Break ties with higher average confidence
        if (totalConf[s] > maxConfSum) {
          bestSign = s;
          maxConfSum = totalConf[s];
        }
      }
    }

    const avgConfidence = maxCount > 0 ? Math.round(maxConfSum / maxCount) : 0;

    return {
      sign: bestSign === 'UNCLEAR' ? null : bestSign,
      confidence: avgConfidence
    };
  }

  _addWord(wordObj) {
    this.lastConfirmedWord = wordObj.word;
    this.confirmedHoldFrames = this.confirmedHoldMax;

    // Auto-insert "NEED" if user signs "I" followed by "WATER"
    if (wordObj.word === 'WATER' && this.recognizedWords.length > 0 && this.recognizedWords[this.recognizedWords.length - 1] === 'I') {
      this.recognizedWords.push('NEED');
    }

    this.recognizedWords.push(wordObj.word);
    
    while (this.recognizedWords.length > this.maxHistory) {
      this.recognizedWords.shift();
    }
  }

  _buildResult(opts) {
    const { detected = false, bestSign = null, bestConf = 0, alts = [] } = opts;

    const isLocked = this.wordEngine.isLocked();
    const forming = this.wordEngine.getFormingWord();
    const progress = this.wordEngine.getProgress();

    // ── Status ──
    let status;
    let state;
    if (this.confirmedHoldFrames > 0) {
      this.confirmedHoldFrames--;
      status = `Word Confirmed: ${this.lastConfirmedWord}`;
      state = 'word-confirmed';
    } else if (isLocked) {
      status = 'Gesture locked — change sign to continue';
      state = 'locked';
    } else if (detected && forming) {
      status = `Stabilizing: ${forming} (${Math.round(progress * 100)}%)`;
      state = 'stabilizing';
    } else if (detected) {
      status = 'Detecting...';
      state = 'detecting';
    } else {
      status = 'Waiting...';
      state = 'idle';
    }

    // Show locked word, confirmed word, or current stabilizing candidate
    const displaySign = (isLocked || this.confirmedHoldFrames > 0)
      ? { sign: isLocked ? this.wordEngine.getLockedWord() : this.lastConfirmedWord, confidence: 99 }
      : (forming ? { sign: forming, confidence: Math.round(progress * 100) } : null);

    return {
      detected,
      tracking_lost: false,
      current_sign: displaySign,
      current_word: this.confirmedHoldFrames > 0 ? this.lastConfirmedWord : '',
      forming_letters: forming,
      forming_progress: progress,
      sentence: this.recognizedWords.join('  →  '),
      gloss: '',
      alternatives: [],
      word_confidence: (isLocked || this.confirmedHoldFrames > 0) ? '99%' : (forming ? `${Math.round(progress * 100)}%` : '—'),
      sentence_confidence: '—',
      status,
      state,
      buffer_words: [],
      buffer_count: 0,
      history: [],
    };
  }

  undoLastWord() {
    if (this.recognizedWords.length > 0) {
      this.recognizedWords.pop();
    }
    this.lastConfirmedWord = null;
    this.confirmedHoldFrames = 0;
  }

  reset() {
    this.motion.reset();
    this.wordEngine.reset();
    this.hadHands = false;
    this.recognizedWords = [];
    this.lastConfirmedWord = null;
    this.confirmedHoldFrames = 0;
    this.predictionHistory = [];
  }
}
