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
  }

  analyze(handsLandmarks, handedness, timestamp) {
    const hasHands = handsLandmarks && handsLandmarks.length > 0;

    // ── No hands ──
    if (!hasHands) {
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

    // Word engine — handles stability, locking, cooldown
    this.wordEngine.update(bestSign, bestConf, motionResult.isPause, timestamp, true);

    const completedWord = this.wordEngine.getCompletedWord();
    if (completedWord) {
      this._addWord(completedWord);
    }

    return this._buildResult({
      detected: true,
      bestSign,
      bestConf,
      alts,
      timestamp,
    });
  }

  _addWord(wordObj) {
    this.lastConfirmedWord = wordObj.word;
    this.confirmedHoldFrames = this.confirmedHoldMax;
    this.recognizedWords.push(wordObj.word);
    if (this.recognizedWords.length > this.maxHistory) this.recognizedWords.shift();
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

    // Show locked word or current best sign
    const displaySign = isLocked
      ? { sign: this.wordEngine.getLockedWord(), confidence: 99 }
      : (bestSign ? { sign: bestSign, confidence: bestConf } : null);

    return {
      detected,
      tracking_lost: false,
      current_sign: displaySign,
      current_word: this.confirmedHoldFrames > 0 ? this.lastConfirmedWord : '',
      forming_letters: forming,
      forming_progress: progress,
      sentence: this.recognizedWords.join('  →  '),
      gloss: '',
      alternatives: isLocked ? [] : alts,
      word_confidence: bestConf > 0 ? `${bestConf}%` : '—',
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
  }
}
