/**
 * Visual Quality Analyzer — assesses frame lighting, occlusion, and clarity.
 * Uses canvas pixel analysis and landmark confidence.
 */

export class VisualQualityAnalyzer {
  constructor() {
    this.prevFrame = null;
    this.brightnessHistory = [];
    this.maxHistory = 20;
  }

  /**
   * @param {HTMLCanvasElement} canvas — the video frame rendered to canvas
   * @param {Array|null} faceLandmarks — face landmarks if detected
   * @param {object|null} faceResult — full face detection result for confidence
   * @returns {object} quality assessment
   */
  analyze(canvas, faceLandmarks, faceResult) {
    const quality = {
      lighting: 'Unknown',
      occlusion: 'None',
      clarity: 'Unknown',
    };

    // ── Lighting Analysis ──
    try {
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      const w = canvas.width;
      const h = canvas.height;

      // Sample pixels in a grid (skip pixels for performance)
      const step = 8;
      let totalBrightness = 0;
      let pixelCount = 0;
      let minBrightness = 255;
      let maxBrightness = 0;

      const imageData = ctx.getImageData(0, 0, w, h);
      const data = imageData.data;

      for (let y = 0; y < h; y += step) {
        for (let x = 0; x < w; x += step) {
          const i = (y * w + x) * 4;
          const brightness = (data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114);
          totalBrightness += brightness;
          pixelCount++;
          if (brightness < minBrightness) minBrightness = brightness;
          if (brightness > maxBrightness) maxBrightness = brightness;
        }
      }

      const avgBrightness = totalBrightness / pixelCount;
      const contrast = maxBrightness - minBrightness;

      this.brightnessHistory.push(avgBrightness);
      if (this.brightnessHistory.length > this.maxHistory) this.brightnessHistory.shift();

      // Classify lighting
      if (avgBrightness < 40) {
        quality.lighting = 'Poor (too dark)';
      } else if (avgBrightness < 70) {
        quality.lighting = 'Low';
      } else if (avgBrightness > 220) {
        quality.lighting = 'Poor (overexposed)';
      } else if (avgBrightness > 190) {
        quality.lighting = 'Bright';
      } else if (contrast < 50) {
        quality.lighting = 'Flat (low contrast)';
      } else {
        quality.lighting = 'Adequate';
      }

      // ── Clarity (motion blur estimation) ──
      // Compare current frame luminance variance to detect blur
      if (this.prevFrame && this.prevFrame.length === data.length) {
        let frameDiff = 0;
        let sampleCount = 0;
        for (let i = 0; i < data.length; i += step * 4) {
          const currL = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
          const prevL = this.prevFrame[i] * 0.299 + this.prevFrame[i + 1] * 0.587 + this.prevFrame[i + 2] * 0.114;
          frameDiff += Math.abs(currL - prevL);
          sampleCount++;
        }
        const avgDiff = frameDiff / sampleCount;

        if (avgDiff > 30) {
          quality.clarity = 'Motion blur likely';
        } else if (avgDiff > 15) {
          quality.clarity = 'Moderate movement';
        } else {
          quality.clarity = 'Good';
        }
      } else {
        quality.clarity = 'Good';
      }

      // Store current frame for next comparison
      this.prevFrame = new Uint8ClampedArray(data);
    } catch (e) {
      quality.lighting = 'Unable to assess';
      quality.clarity = 'Unable to assess';
    }

    // ── Occlusion Detection ──
    if (!faceLandmarks || faceLandmarks.length === 0) {
      quality.occlusion = 'Face not detected';
    } else if (faceLandmarks.length < 400) {
      quality.occlusion = 'Partial (limited landmarks)';
    } else {
      // Check if key face regions have valid landmarks
      // Nose tip (1), left eye (33), right eye (263), mouth (13), chin (152)
      const keyIndices = [1, 33, 263, 13, 152, 70, 300];
      let visibleCount = 0;

      for (const idx of keyIndices) {
        const lm = faceLandmarks[idx];
        if (lm && lm.x > 0.05 && lm.x < 0.95 && lm.y > 0.05 && lm.y < 0.95) {
          visibleCount++;
        }
      }

      if (visibleCount >= 6) {
        quality.occlusion = 'None';
      } else if (visibleCount >= 4) {
        quality.occlusion = 'Partial';
      } else {
        quality.occlusion = 'Significant';
      }
    }

    return quality;
  }
}
