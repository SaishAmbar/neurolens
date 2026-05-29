/**
 * Emotion Analyzer — maps MediaPipe face blendshapes to emotion classifications.
 * Uses weighted blendshape profiles with temporal smoothing.
 */

const EMOTION_PROFILES = {
  happy: {
    label: 'Happy',
    weights: {
      mouthSmileLeft: 0.22, mouthSmileRight: 0.22,
      cheekSquintLeft: 0.16, cheekSquintRight: 0.16,
      mouthDimpleLeft: 0.06, mouthDimpleRight: 0.06,
      eyeSquintLeft: 0.06, eyeSquintRight: 0.06,
    },
    threshold: 0.18,
  },
  sad: {
    label: 'Sad',
    weights: {
      mouthFrownLeft: 0.20, mouthFrownRight: 0.20,
      browInnerUp: 0.25,
      mouthLowerDownLeft: 0.07, mouthLowerDownRight: 0.07,
      mouthPressLeft: 0.05, mouthPressRight: 0.05,
      mouthPucker: 0.06,
      // Lip corners down is key distinguisher for sad
      mouthStretchLeft: 0.025, mouthStretchRight: 0.025,
    },
    // Negative inhibitors: presence of these REDUCES sad score
    inhibitors: {
      browDownLeft: 0.35, browDownRight: 0.35,
      noseSneerLeft: 0.25, noseSneerRight: 0.25,
      jawForward: 0.20,
    },
    threshold: 0.16,
  },
  angry: {
    label: 'Angry',
    weights: {
      browDownLeft: 0.25, browDownRight: 0.25,
      noseSneerLeft: 0.12, noseSneerRight: 0.12,
      jawForward: 0.10,
      mouthShrugLower: 0.06,
      // Tightened eyes + pressed lips are key for angry
      eyeSquintLeft: 0.05, eyeSquintRight: 0.05,
    },
    // Negative inhibitors: presence of these REDUCES angry score
    inhibitors: {
      browInnerUp: 0.40,
      mouthPucker: 0.20,
      mouthFrownLeft: 0.10, mouthFrownRight: 0.10,
    },
    threshold: 0.18,
  },
  surprised: {
    label: 'Surprised',
    weights: {
      browOuterUpLeft: 0.17, browOuterUpRight: 0.17,
      browInnerUp: 0.14,
      jawOpen: 0.17,
      eyeWideLeft: 0.17, eyeWideRight: 0.17,
    },
    threshold: 0.2,
  },
  fear: {
    label: 'Fear',
    weights: {
      browInnerUp: 0.18,
      eyeWideLeft: 0.18, eyeWideRight: 0.18,
      mouthStretchLeft: 0.11, mouthStretchRight: 0.11,
      jawOpen: 0.12,
      mouthFunnel: 0.06,
      mouthFrownLeft: 0.03, mouthFrownRight: 0.03,
    },
    threshold: 0.16,
  },
  disgust: {
    label: 'Disgust',
    weights: {
      noseSneerLeft: 0.2, noseSneerRight: 0.2,
      mouthUpperUpLeft: 0.15, mouthUpperUpRight: 0.15,
      mouthFrownLeft: 0.08, mouthFrownRight: 0.08,
      cheekPuff: 0.07,
      browDownLeft: 0.035, browDownRight: 0.035,
    },
    threshold: 0.16,
  },
};

export class EmotionAnalyzer {
  constructor() {
    this.history = [];            // rolling window of raw scores
    this.maxHistory = 15;         // ~0.5s at 30fps
    this.microExpressionBuffer = [];
    this.maxMicroBuffer = 5;
    this.prevBlendshapes = null;
    this.smoothedScores = {};
    this.smoothingAlpha = 0.35;   // EMA factor
  }

  /**
   * @param {Array} blendshapes — array of {categoryName, score}
   * @returns {object} analysis result
   */
  analyze(blendshapes) {
    if (!blendshapes || blendshapes.length === 0) {
      return this._noDetection();
    }

    // Convert to map
    const bs = {};
    for (const b of blendshapes) {
      bs[b.categoryName] = b.score;
    }

    // Score each emotion
    const rawScores = {};
    const evidenceParts = [];

    for (const [emoKey, profile] of Object.entries(EMOTION_PROFILES)) {
      let score = 0;
      const parts = [];
      for (const [shapeName, weight] of Object.entries(profile.weights)) {
        const val = bs[shapeName] || 0;
        score += val * weight;
        if (val > 0.15) {
          parts.push(`${shapeName}: ${val.toFixed(2)}`);
        }
      }
      // Apply inhibitors — subtract penalty when conflicting blendshapes are active
      if (profile.inhibitors) {
        for (const [shapeName, penalty] of Object.entries(profile.inhibitors)) {
          const val = bs[shapeName] || 0;
          if (val > 0.1) {
            score -= val * penalty;
          }
        }
      }
      rawScores[emoKey] = Math.max(0, score);
    }

    // Check for contempt (asymmetric smile)
    const smileL = bs.mouthSmileLeft || 0;
    const smileR = bs.mouthSmileRight || 0;
    const smileAsymmetry = Math.abs(smileL - smileR);
    const avgSmile = (smileL + smileR) / 2;
    if (smileAsymmetry > 0.12 && avgSmile > 0.1) {
      rawScores.contempt = smileAsymmetry * 0.6 + avgSmile * 0.2;
    } else {
      rawScores.contempt = 0;
    }

    // Apply EMA smoothing
    for (const key of Object.keys(rawScores)) {
      if (this.smoothedScores[key] === undefined) {
        this.smoothedScores[key] = rawScores[key];
      } else {
        this.smoothedScores[key] =
          this.smoothingAlpha * rawScores[key] +
          (1 - this.smoothingAlpha) * this.smoothedScores[key];
      }
    }

    // Store history
    this.history.push({ ...this.smoothedScores });
    if (this.history.length > this.maxHistory) this.history.shift();

    // Sort emotions by smoothed score
    const sorted = Object.entries(this.smoothedScores)
      .sort((a, b) => b[1] - a[1]);

    const topKey = sorted[0][0];
    const topScore = sorted[0][1];
    const secondKey = sorted[1] ? sorted[1][0] : null;
    const secondScore = sorted[1] ? sorted[1][1] : 0;

    const topProfile = EMOTION_PROFILES[topKey];
    const topThreshold = topProfile ? topProfile.threshold : 0.15;

    // Determine if we have a valid detection
    if (topScore < topThreshold * 0.7) {
      return this._neutral(bs);
    }

    const label = topKey === 'contempt' ? 'Contempt' :
      (EMOTION_PROFILES[topKey]?.label || topKey);

    // Confidence based on score magnitude and separation from runner-up
    const separation = topScore - secondScore;
    let confidence = Math.min(95, Math.round(
      (topScore / 0.5) * 60 + (separation / 0.2) * 30
    ));
    confidence = Math.max(15, confidence);

    // Intensity
    let intensity;
    if (topScore > 0.35) intensity = 'High';
    else if (topScore > 0.2) intensity = 'Medium';
    else intensity = 'Low';

    // Secondary emotion
    let secondary = 'None';
    const secProfile = EMOTION_PROFILES[secondKey];
    const secThreshold = secProfile ? secProfile.threshold : 0.15;
    if (secondScore > secThreshold * 0.8 && secondScore > topScore * 0.4) {
      secondary = secondKey === 'contempt' ? 'Contempt' :
        (EMOTION_PROFILES[secondKey]?.label || secondKey);
    }

    // Evidence generation
    const evidence = this._buildEvidence(bs, topKey, secondKey);

    // Micro-expression detection
    this._detectMicroExpression(rawScores);

    return {
      primary: label,
      secondary,
      intensity,
      confidence: `${confidence}%`,
      evidence,
      _rawScores: this.smoothedScores,
    };
  }

  _neutral(bs) {
    return {
      primary: 'Neutral',
      secondary: 'None',
      intensity: 'Low',
      confidence: '60%',
      evidence: 'Low activation across all emotional blendshape channels. Face appears in resting state.',
      _rawScores: this.smoothedScores,
    };
  }

  _noDetection() {
    return {
      primary: 'Not reliably detectable',
      secondary: 'N/A',
      intensity: 'N/A',
      confidence: '0%',
      evidence: 'No face blendshape data available.',
      _rawScores: {},
    };
  }

  _buildEvidence(bs, primaryKey, secondaryKey) {
    const parts = [];
    const profile = EMOTION_PROFILES[primaryKey];

    if (primaryKey === 'contempt') {
      const smileL = (bs.mouthSmileLeft || 0).toFixed(2);
      const smileR = (bs.mouthSmileRight || 0).toFixed(2);
      parts.push(`Asymmetric mouth corner elevation (L=${smileL}, R=${smileR})`);
    } else if (profile) {
      const topShapes = Object.entries(profile.weights)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 4);

      for (const [name, w] of topShapes) {
        const val = bs[name] || 0;
        if (val > 0.05) {
          parts.push(`${this._formatName(name)}: ${val.toFixed(2)}`);
        }
      }
    }

    if (parts.length === 0) {
      parts.push('Weak overall facial activation detected');
    }

    return parts.join('. ') + '.';
  }

  _formatName(name) {
    return name
      .replace(/([A-Z])/g, ' $1')
      .replace(/^./, s => s.toUpperCase())
      .replace(/Left$/, '(L)')
      .replace(/Right$/, '(R)');
  }

  _detectMicroExpression(rawScores) {
    if (!this.prevBlendshapes) {
      this.prevBlendshapes = rawScores;
      return null;
    }
    // Detect rapid spikes (micro-expressions last 1/25 to 1/5 second)
    for (const key of Object.keys(rawScores)) {
      const delta = Math.abs(rawScores[key] - (this.prevBlendshapes[key] || 0));
      if (delta > 0.15) {
        this.microExpressionBuffer.push({
          emotion: key,
          delta,
          timestamp: performance.now(),
        });
        if (this.microExpressionBuffer.length > this.maxMicroBuffer) {
          this.microExpressionBuffer.shift();
        }
      }
    }
    this.prevBlendshapes = { ...rawScores };
  }

  getMicroExpressions() {
    const now = performance.now();
    // Only report micro-expressions within the last 2 seconds
    return this.microExpressionBuffer.filter(m => now - m.timestamp < 2000);
  }
}
