// MotionTracker v2 — Enhanced sliding window, velocity, trajectory, gesture boundaries
import { dist } from './features.js';

export class MotionTracker {
  constructor() {
    this.buffer = [];
    this.maxBuffer = 24;            // larger window for better pattern detection
    this.lastHandTime = 0;
    this.trackingLostMs = 600;
    this.pauseMs = 800;             // ms of stillness = word boundary
    this.sentencePauseMs = 2500;    // ms with no hands = sentence end
    this.motionThreshold = 0.012;   // tuned for ~6fps (larger jumps between frames)
    this.gestureStartTime = 0;      // when current gesture movement started
    this.wasMoving = false;
  }

  update(features, timestamp) {
    if (!features) {
      return {
        hasMotion: false,
        velocity: { x: 0, y: 0 },
        speed: 0,
        trajectory: 'none',
        isPause: this._timeSince(timestamp) > this.pauseMs,
        isSentenceEnd: this._timeSince(timestamp) > this.sentencePauseMs,
        isTrackingLost: this._timeSince(timestamp) > this.trackingLostMs,
        pattern: 'none',
        gestureDuration: 0,
        positionZone: 'unknown',
      };
    }

    this.lastHandTime = timestamp;
    this.buffer.push({
      centroid: { ...features.centroid },
      wrist: { ...features.wrist },
      timestamp,
      fingers: [...features.fingers],
      extCount: features.extCount,
      palmFacing: features.palmFacing,
    });

    if (this.buffer.length > this.maxBuffer) this.buffer.shift();
    if (this.buffer.length < 2) {
      return this._emptyResult();
    }

    // Smoothed velocity (weighted recent frames more)
    const vel = this._weightedVelocity();
    const speed = Math.sqrt(vel.x ** 2 + vel.y ** 2);
    const hasMotion = speed > this.motionThreshold;

    // Track gesture duration
    if (hasMotion && !this.wasMoving) {
      this.gestureStartTime = timestamp;
    }
    this.wasMoving = hasMotion;
    const gestureDuration = hasMotion ? timestamp - this.gestureStartTime : 0;

    // Pattern analysis with enhanced detection
    const pattern = this._detectPattern();
    const trajectory = this._classifyTrajectory(vel);
    const isPause = this._isStill(6);

    // Position zone (face, chest, waist)
    const cy = features.centroid.y;
    let positionZone = 'waist';
    if (cy < 0.35) positionZone = 'face';
    else if (cy < 0.55) positionZone = 'chest';

    return {
      hasMotion,
      velocity: vel,
      speed,
      trajectory,
      isPause,
      isSentenceEnd: false,
      isTrackingLost: false,
      pattern,
      gestureDuration,
      positionZone,
    };
  }

  _timeSince(now) {
    return this.lastHandTime > 0 ? now - this.lastHandTime : 0;
  }

  // Weighted average: most recent frames count more
  _weightedVelocity() {
    const n = this.buffer.length;
    const count = Math.min(5, n - 1);
    let vx = 0, vy = 0, totalWeight = 0;
    for (let i = n - count; i < n; i++) {
      const dt = this.buffer[i].timestamp - this.buffer[i - 1].timestamp;
      if (dt <= 0) continue;
      const weight = 1 + (i - (n - count)); // newer frames weighted higher
      vx += weight * (this.buffer[i].centroid.x - this.buffer[i - 1].centroid.x) / dt * 1000;
      vy += weight * (this.buffer[i].centroid.y - this.buffer[i - 1].centroid.y) / dt * 1000;
      totalWeight += weight;
    }
    if (totalWeight === 0) return { x: 0, y: 0 };
    return { x: vx / totalWeight, y: vy / totalWeight };
  }

  _isStill(frames) {
    const n = this.buffer.length;
    if (n < frames) return false;
    for (let i = n - frames; i < n - 1; i++) {
      const d = Math.sqrt(
        (this.buffer[i + 1].centroid.x - this.buffer[i].centroid.x) ** 2 +
        (this.buffer[i + 1].centroid.y - this.buffer[i].centroid.y) ** 2
      );
      if (d > this.motionThreshold) return false;
    }
    return true;
  }

  _classifyTrajectory(vel) {
    const ax = Math.abs(vel.x), ay = Math.abs(vel.y);
    if (ax < 0.015 && ay < 0.015) return 'still';
    if (ax > ay * 1.8) return vel.x > 0 ? 'right' : 'left';
    if (ay > ax * 1.8) return vel.y > 0 ? 'down' : 'up';
    return 'diagonal';
  }

  _detectPattern() {
    if (this.buffer.length < 8) return 'none';
    const recent = this.buffer.slice(-12);
    if (recent.length < 6) return 'none';

    // Wave: alternating x direction (3+ reversals)
    let xReversals = 0;
    for (let i = 2; i < recent.length; i++) {
      const dx1 = recent[i - 1].centroid.x - recent[i - 2].centroid.x;
      const dx2 = recent[i].centroid.x - recent[i - 1].centroid.x;
      if (dx1 * dx2 < 0 && Math.abs(dx1) > 0.002 && Math.abs(dx2) > 0.002) xReversals++;
    }
    if (xReversals >= 3) return 'wave';

    // Nod: alternating y direction (3+ reversals)
    let yReversals = 0;
    for (let i = 2; i < recent.length; i++) {
      const dy1 = recent[i - 1].centroid.y - recent[i - 2].centroid.y;
      const dy2 = recent[i].centroid.y - recent[i - 1].centroid.y;
      if (dy1 * dy2 < 0 && Math.abs(dy1) > 0.002 && Math.abs(dy2) > 0.002) yReversals++;
    }
    if (yReversals >= 3) return 'nod';

    const first = recent[0], last = recent[recent.length - 1];

    // Push/pull: z-axis change
    const dz = (last.centroid.z || 0) - (first.centroid.z || 0);
    if (Math.abs(dz) > 0.04) return dz > 0 ? 'push' : 'pull';

    // Lateral sweep
    const dx = last.centroid.x - first.centroid.x;
    if (Math.abs(dx) > 0.12) return dx > 0 ? 'sweep-right' : 'sweep-left';

    // Vertical sweep
    const dy = last.centroid.y - first.centroid.y;
    if (Math.abs(dy) > 0.12) return dy > 0 ? 'sweep-down' : 'sweep-up';

    return 'none';
  }

  _emptyResult() {
    return {
      hasMotion: false, velocity: { x: 0, y: 0 }, speed: 0,
      trajectory: 'none', isPause: false, isSentenceEnd: false,
      isTrackingLost: false, pattern: 'none', gestureDuration: 0,
      positionZone: 'unknown',
    };
  }

  reset() {
    this.buffer = [];
    this.lastHandTime = 0;
    this.gestureStartTime = 0;
    this.wasMoving = false;
  }
}
