/**
 * NeuroLens — Browser-Based Obstacle Detection Analyzer
 * Uses TensorFlow.js COCO-SSD for real-time object detection with
 * distance estimation and threat classification.
 */

export class NeuroLensAnalyzer {
  constructor() {
    this._model = null;
    this._loading = false;
    this._loaded = false;
    this._startTime = Date.now();
    this._frameCount = 0;

    // Distance estimation constants — calibrated for ~78° FOV webcam
    this._focalLength = 550;

    // Known real-world heights in cm (comprehensive for all COCO-SSD classes)
    this._knownHeights = {
      person: 90,       // Upper-body/torso (webcam rarely sees full body)
      car: 150, bicycle: 100, motorcycle: 110,
      airplane: 400, bus: 300, train: 350, truck: 350, boat: 150,
      'traffic light': 100, 'fire hydrant': 50, 'stop sign': 75,
      'parking meter': 120, bench: 85,
      bird: 20, cat: 30, dog: 60, horse: 160, sheep: 70, cow: 140,
      elephant: 300, bear: 150, zebra: 140, giraffe: 500,
      backpack: 50, umbrella: 100, handbag: 35, tie: 50, suitcase: 60,
      frisbee: 3, skis: 170, snowboard: 150, 'sports ball': 22,
      kite: 80, 'baseball bat': 90, 'baseball glove': 25,
      skateboard: 15, surfboard: 200, 'tennis racket': 70,
      bottle: 25, 'wine glass': 20, cup: 12,
      fork: 20, knife: 25, spoon: 18, bowl: 10,
      banana: 20, apple: 8, sandwich: 8, orange: 8,
      broccoli: 15, carrot: 20, 'hot dog': 5, pizza: 5, donut: 8, cake: 15,
      chair: 90, couch: 85, 'potted plant': 40,
      bed: 60, 'dining table': 75, toilet: 40,
      tv: 50, laptop: 25, mouse: 4, remote: 20, keyboard: 5,
      'cell phone': 15, microwave: 30, oven: 60, toaster: 20,
      sink: 25, refrigerator: 170, book: 25,
      clock: 30, vase: 30, scissors: 20, 'teddy bear': 40,
      'hair drier': 25, toothbrush: 18,
    };

    // Zone thresholds in cm
    this._VERY_CLOSE_CM = 150;
    this._NEAR_CM = 300;
    this._VERY_CLOSE_RATIO = 0.50;
    this._NEAR_RATIO = 0.25;

    // Zone colors for canvas rendering
    this._zoneColors = {
      'SAFE':       { main: '#00ff80', glow: 'rgba(0, 255, 128, 0.3)' },
      'NEAR':       { main: '#ffc800', glow: 'rgba(255, 200, 0, 0.3)' },
      'VERY CLOSE': { main: '#ff4040', glow: 'rgba(255, 64, 64, 0.4)' },
    };
  }

  /** Load the COCO-SSD model */
  async loadModel() {
    if (this._loaded || this._loading) return;
    this._loading = true;

    try {
      // Dynamically load TensorFlow.js and COCO-SSD
      if (!window.cocoSsd) {
        await this._loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0/dist/tf.min.js');
        await this._loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js');
      }

      this._model = await window.cocoSsd.load({
        base: 'lite_mobilenet_v2', // Fast model for real-time
      });

      this._loaded = true;
      this._loading = false;
      console.log('[NeuroLens] COCO-SSD model loaded');
    } catch (err) {
      this._loading = false;
      console.error('[NeuroLens] Failed to load model:', err);
      throw err;
    }
  }

  /** Load an external script dynamically */
  _loadScript(src) {
    return new Promise((resolve, reject) => {
      if (document.querySelector(`script[src="${src}"]`)) {
        resolve();
        return;
      }
      const script = document.createElement('script');
      script.src = src;
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  get isLoaded() { return this._loaded; }

  /**
   * Analyze a video frame for obstacles.
   * @param {HTMLVideoElement} video
   * @param {CanvasRenderingContext2D} ctx - overlay canvas context
   * @param {number} canvasWidth
   * @param {number} canvasHeight
   * @returns {Object} analysis result
   */
  async analyze(video, ctx, canvasWidth, canvasHeight) {
    if (!this._loaded || !this._model) {
      return this._emptyResult();
    }

    this._frameCount++;

    // Run detection
    let predictions;
    try {
      predictions = await this._model.detect(video);
    } catch {
      return this._emptyResult();
    }

    // Filter to obstacle classes only and by confidence
    // Accept all COCO-SSD classes, filter only by confidence
    const obstacles = predictions.filter(p => p.score >= 0.35);

    // Estimate distances and classify zones
    const results = obstacles.map(pred => {
      const [x, y, w, h] = pred.bbox;
      const centerX = x + w / 2;
      const centerY = y + h / 2;
      const estimatedCm = this._estimateDistance(pred.class, h, canvasHeight);
      const zone = this._classifyZone(estimatedCm, h, canvasHeight);
      const direction = this._classifyDirection(centerX, canvasWidth);
      const dangerScore = this._calculateDanger(estimatedCm, centerX, canvasWidth, pred.class);

      return {
        class: pred.class,
        confidence: pred.score,
        bbox: { x, y, w, h },
        centerX, centerY,
        estimatedCm,
        zone,
        direction,
        dangerScore,
      };
    });

    // Sort by danger score (most dangerous first)
    results.sort((a, b) => b.dangerScore - a.dangerScore);

    // Draw HUD
    const elapsed = (Date.now() - this._startTime) / 1000;
    this._drawHUD(ctx, canvasWidth, canvasHeight, results, elapsed);

    // Build result object
    const mostDangerous = results[0] || null;
    const alertZone = mostDangerous ? mostDangerous.zone : 'SAFE';

    return {
      detectionCount: results.length,
      alertZone,
      alertLabel: this._getAlertLabel(alertZone),
      mostDangerous: mostDangerous ? {
        class: mostDangerous.class,
        zone: mostDangerous.zone,
        direction: mostDangerous.direction,
        distanceM: (mostDangerous.estimatedCm / 100).toFixed(1),
        dangerScore: mostDangerous.dangerScore.toFixed(2),
      } : null,
      detections: results.map(r => ({
        class: r.class,
        zone: r.zone,
        direction: r.direction,
        distanceM: (r.estimatedCm / 100).toFixed(1),
        confidence: (r.confidence * 100).toFixed(0) + '%',
      })),
    };
  }

  // ══════════════════════════════════════════════════════════════════════
  //  DISTANCE ESTIMATION
  // ══════════════════════════════════════════════════════════════════════

  _estimateDistance(className, bboxHeight, frameHeight) {
    const knownH = this._knownHeights[className];
    if (knownH && bboxHeight > 0) {
      return (knownH * this._focalLength) / bboxHeight;
    }
    // Fallback
    const ratio = bboxHeight / frameHeight;
    return Math.max(50, (1.0 - ratio) * 500);
  }

  _classifyZone(estimatedCm, bboxHeight, frameHeight) {
    const ratio = bboxHeight / frameHeight;
    if (estimatedCm < this._VERY_CLOSE_CM || ratio > this._VERY_CLOSE_RATIO)
      return 'VERY CLOSE';
    if (estimatedCm < this._NEAR_CM || ratio > this._NEAR_RATIO)
      return 'NEAR';
    return 'SAFE';
  }

  _classifyDirection(centerX, frameWidth) {
    const third = frameWidth / 3;
    if (centerX < third) return 'LEFT';
    if (centerX > third * 2) return 'RIGHT';
    return 'CENTER';
  }

  _calculateDanger(estimatedCm, centerX, frameWidth, className) {
    const distFactor = Math.max(0, 1.0 - estimatedCm / 500);
    const centerOffset = Math.abs(centerX - frameWidth / 2) / (frameWidth / 2);
    const centerFactor = 1.0 - centerOffset * 0.5;
    const knownH = this._knownHeights[className] || 50;
    const sizeFactor = Math.min(1.0, knownH / 200);
    return Math.min(1.0, Math.max(0, distFactor * 0.6 + centerFactor * 0.3 + sizeFactor * 0.1));
  }

  _getAlertLabel(zone) {
    const labels = {
      'SAFE': 'PATH CLEAR',
      'NEAR': 'OBSTACLE AHEAD',
      'VERY CLOSE': 'IMMEDIATE DANGER',
    };
    return labels[zone] || 'PATH CLEAR';
  }

  _emptyResult() {
    return {
      detectionCount: 0,
      alertZone: 'SAFE',
      alertLabel: 'PATH CLEAR',
      mostDangerous: null,
      detections: [],
    };
  }

  // ══════════════════════════════════════════════════════════════════════
  //  HUD RENDERING
  // ══════════════════════════════════════════════════════════════════════

  _drawHUD(ctx, w, h, results, t) {
    ctx.save();

    // ── Grid ──
    this._drawGrid(ctx, w, h);

    // ── Scan line ──
    this._drawScanLine(ctx, w, h, t);

    // ── Detection box — CLOSEST OBJECT ONLY ──
    if (results.length > 0) {
      this._drawDetectionBox(ctx, results[0], true, t, w, h);
    }

    // ── Danger flash ──
    const mostDangerous = results[0];
    if (mostDangerous && mostDangerous.zone === 'VERY CLOSE') {
      this._drawDangerFlash(ctx, w, h, t);
    }

    // ── Corner brackets ──
    const alertZone = mostDangerous ? mostDangerous.zone : 'SAFE';
    this._drawCornerBrackets(ctx, w, h, alertZone);

    // ── Header ──
    this._drawHeader(ctx, w, t);

    // ── Warning panel ──
    this._drawWarningPanel(ctx, w, mostDangerous, t);

    // ── Direction indicators ──
    this._drawDirectionIndicators(ctx, w, h, mostDangerous, t);

    // ── Stats footer ──
    this._drawStatsFooter(ctx, w, h, results.length);

    // ── Distance legend ──
    this._drawDistanceLegend(ctx, w, h);

    ctx.restore();
  }

  _drawGrid(ctx, w, h) {
    ctx.save();
    ctx.globalAlpha = 0.08;
    ctx.strokeStyle = '#405050';
    ctx.lineWidth = 0.5;
    const spacing = 40;
    for (let x = 0; x < w; x += spacing) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += spacing) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
    ctx.restore();
  }

  _drawScanLine(ctx, w, h, t) {
    const cycle = (t % 3.0) / 3.0;
    const y = cycle * h;
    ctx.save();

    // Main line
    const grad = ctx.createLinearGradient(0, y, w, y);
    grad.addColorStop(0, 'transparent');
    grad.addColorStop(0.3, 'rgba(200, 255, 0, 0.5)');
    grad.addColorStop(0.7, 'rgba(200, 255, 0, 0.5)');
    grad.addColorStop(1, 'transparent');
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();

    // Trail
    ctx.globalAlpha = 0.1;
    ctx.strokeStyle = 'rgba(200, 255, 0, 0.3)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 6; i++) {
      const ty = y - i * 3;
      if (ty >= 0) {
        ctx.beginPath(); ctx.moveTo(0, ty); ctx.lineTo(w, ty); ctx.stroke();
      }
    }
    ctx.restore();
  }

  _drawDetectionBox(ctx, result, isPrimary, t, canvasW, canvasH) {
    const { bbox, zone, class: className, confidence, estimatedCm } = result;
    const { x, y, w, h } = bbox;
    const colors = this._zoneColors[zone];
    const color = colors.main;
    const thickness = isPrimary ? 3 : 2;

    ctx.save();

    // ── Glow for primary ──
    if (isPrimary) {
      const pulse = 0.5 + 0.5 * Math.sin(t * 4);
      ctx.globalAlpha = 0.3 * pulse;
      ctx.strokeStyle = color;
      ctx.lineWidth = 6;
      ctx.strokeRect(x - 2, y - 2, w + 4, h + 4);
      ctx.globalAlpha = 1;
    }

    // ── Corner brackets ──
    const cornerLen = Math.min(20, w / 4, h / 4);
    ctx.strokeStyle = color;
    ctx.lineWidth = thickness;

    // Top-left
    ctx.beginPath(); ctx.moveTo(x, y + cornerLen); ctx.lineTo(x, y); ctx.lineTo(x + cornerLen, y); ctx.stroke();
    // Top-right
    ctx.beginPath(); ctx.moveTo(x + w - cornerLen, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + cornerLen); ctx.stroke();
    // Bottom-left
    ctx.beginPath(); ctx.moveTo(x, y + h - cornerLen); ctx.lineTo(x, y + h); ctx.lineTo(x + cornerLen, y + h); ctx.stroke();
    // Bottom-right
    ctx.beginPath(); ctx.moveTo(x + w - cornerLen, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - cornerLen); ctx.stroke();

    // ── Dashed connecting lines ──
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 1;
    // Top
    ctx.beginPath(); ctx.moveTo(x + cornerLen, y); ctx.lineTo(x + w - cornerLen, y); ctx.stroke();
    // Bottom
    ctx.beginPath(); ctx.moveTo(x + cornerLen, y + h); ctx.lineTo(x + w - cornerLen, y + h); ctx.stroke();
    // Left
    ctx.beginPath(); ctx.moveTo(x, y + cornerLen); ctx.lineTo(x, y + h - cornerLen); ctx.stroke();
    // Right
    ctx.beginPath(); ctx.moveTo(x + w, y + cornerLen); ctx.lineTo(x + w, y + h - cornerLen); ctx.stroke();
    ctx.setLineDash([]);

    // ── Label ──
    const distM = (estimatedCm / 100).toFixed(1);
    const label = `${className.toUpperCase()} | ${zone} | ${distM}m`;
    ctx.font = '600 11px "Inter", sans-serif';
    const labelMetrics = ctx.measureText(label);
    const labelW = labelMetrics.width + 12;
    const labelH = 20;
    const labelX = x;
    const labelY = Math.max(y - 26, 4);

    // Label background
    ctx.fillStyle = 'rgba(15, 15, 15, 0.85)';
    ctx.fillRect(labelX, labelY, labelW, labelH);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.strokeRect(labelX, labelY, labelW, labelH);

    // Label text
    ctx.fillStyle = color;
    ctx.fillText(label, labelX + 6, labelY + 14);

    // Confidence
    const confLabel = `${(confidence * 100).toFixed(0)}%`;
    ctx.font = '500 10px "Inter", sans-serif';
    ctx.fillStyle = color;
    const confMetrics = ctx.measureText(confLabel);
    ctx.fillText(confLabel, x + w - confMetrics.width - 2, y + h + 14);

    // ── Crosshair on primary ──
    if (isPrimary) {
      const cx = result.centerX;
      const cy = result.centerY;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx - 10, cy); ctx.lineTo(cx + 10, cy); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx, cy - 10); ctx.lineTo(cx, cy + 10); ctx.stroke();
      ctx.beginPath(); ctx.arc(cx, cy, 12, 0, Math.PI * 2); ctx.stroke();
    }

    ctx.restore();
  }

  _drawDangerFlash(ctx, w, h, t) {
    const pulse = 0.15 + 0.15 * Math.sin(t * 6);
    ctx.save();

    // Red border
    ctx.globalAlpha = pulse;
    ctx.strokeStyle = '#ff0000';
    ctx.lineWidth = 8;
    ctx.strokeRect(4, 4, w - 8, h - 8);

    // Vignette
    const grad = ctx.createRadialGradient(w / 2, h / 2, w * 0.3, w / 2, h / 2, w * 0.7);
    grad.addColorStop(0, 'transparent');
    grad.addColorStop(1, `rgba(200, 0, 0, ${pulse * 0.4})`);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    ctx.restore();
  }

  _drawCornerBrackets(ctx, w, h, alertZone) {
    const colors = this._zoneColors[alertZone];
    const color = colors.main;
    const len = 30;
    const m = 8;

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;

    // Top-left
    ctx.beginPath(); ctx.moveTo(m, m + len); ctx.lineTo(m, m); ctx.lineTo(m + len, m); ctx.stroke();
    // Top-right
    ctx.beginPath(); ctx.moveTo(w - m - len, m); ctx.lineTo(w - m, m); ctx.lineTo(w - m, m + len); ctx.stroke();
    // Bottom-left
    ctx.beginPath(); ctx.moveTo(m, h - m - len); ctx.lineTo(m, h - m); ctx.lineTo(m + len, h - m); ctx.stroke();
    // Bottom-right
    ctx.beginPath(); ctx.moveTo(w - m - len, h - m); ctx.lineTo(w - m, h - m); ctx.lineTo(w - m, h - m - len); ctx.stroke();

    ctx.restore();
  }

  _drawHeader(ctx, w, t) {
    ctx.save();

    // Blinking dot
    if (Math.floor(t * 2) % 2 === 0) {
      ctx.fillStyle = '#00ff80';
      ctx.beginPath(); ctx.arc(25, 30, 4, 0, Math.PI * 2); ctx.fill();
    }

    // Branding
    ctx.font = '700 14px "Inter", sans-serif';
    ctx.fillStyle = '#dce0e8';
    ctx.fillText('NEUROLENS', 38, 35);

    ctx.font = '400 10px "Inter", sans-serif';
    ctx.fillStyle = '#ffb400';
    ctx.fillText('v1.0', 130, 35);

    // Timestamp
    const ts = new Date().toLocaleTimeString();
    ctx.font = '400 10px "JetBrains Mono", monospace';
    ctx.fillStyle = '#8a8aaa';
    const tsW = ctx.measureText(ts).width;
    ctx.fillText(ts, w - tsW - 20, 48);

    ctx.restore();
  }

  _drawWarningPanel(ctx, w, mostDangerous, t) {
    const zone = mostDangerous ? mostDangerous.zone : 'SAFE';
    const colors = this._zoneColors[zone];
    const labels = { 'SAFE': 'PATH CLEAR', 'NEAR': 'OBSTACLE AHEAD', 'VERY CLOSE': 'IMMEDIATE DANGER' };
    const icons = { 'SAFE': '✓', 'NEAR': '⚠', 'VERY CLOSE': '⛔' };

    const panelW = 280;
    const panelH = 50;
    const px = (w - panelW) / 2;
    const py = 10;

    ctx.save();

    // Background
    ctx.fillStyle = 'rgba(15, 15, 15, 0.75)';
    ctx.fillRect(px, py, panelW, panelH);

    // Border
    if (zone === 'VERY CLOSE') {
      const pulse = 0.5 + 0.5 * Math.sin(t * 6);
      ctx.globalAlpha = pulse;
    }
    ctx.strokeStyle = colors.main;
    ctx.lineWidth = zone === 'VERY CLOSE' ? 2 : 1;
    ctx.strokeRect(px, py, panelW, panelH);
    ctx.globalAlpha = 1;

    // Alert text
    const alertText = `${icons[zone]} ${labels[zone]}`;
    ctx.font = '700 13px "Inter", sans-serif';
    ctx.fillStyle = colors.main;
    const textW = ctx.measureText(alertText).width;
    ctx.fillText(alertText, px + (panelW - textW) / 2, py + 22);

    // Object info
    if (mostDangerous) {
      const info = `${mostDangerous.class.toUpperCase()} — ${mostDangerous.direction}`;
      ctx.font = '400 10px "Inter", sans-serif';
      ctx.fillStyle = '#dce0e8';
      const infoW = ctx.measureText(info).width;
      ctx.fillText(info, px + (panelW - infoW) / 2, py + 40);
    }

    ctx.restore();
  }

  _drawDirectionIndicators(ctx, w, h, mostDangerous, t) {
    const barH = 30;
    const barY = h - barH - 12;
    const sectionW = w / 3;
    const activeDir = mostDangerous ? mostDangerous.direction : null;
    const dirs = ['LEFT', 'CENTER', 'RIGHT'];
    const arrows = ['<< LEFT', '[ CENTER ]', 'RIGHT >>'];

    ctx.save();

    dirs.forEach((dir, i) => {
      const sx = i * sectionW;
      const isActive = dir === activeDir;

      // Background
      ctx.fillStyle = isActive && mostDangerous
        ? this._zoneColors[mostDangerous.zone].glow
        : 'rgba(15, 15, 15, 0.45)';
      ctx.fillRect(sx + 2, barY, sectionW - 4, barH);

      // Text
      ctx.font = `${isActive ? '700' : '400'} 11px "Inter", sans-serif`;
      ctx.fillStyle = isActive ? '#ffffff' : '#8a8aaa';
      const label = arrows[i];
      const tw = ctx.measureText(label).width;
      ctx.fillText(label, sx + (sectionW - tw) / 2, barY + barH / 2 + 4);

      // Separator
      if (i > 0) {
        ctx.strokeStyle = '#405050';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(sx, barY + 4); ctx.lineTo(sx, barY + barH - 4); ctx.stroke();
      }
    });

    ctx.restore();
  }

  _drawStatsFooter(ctx, w, h, count) {
    ctx.save();
    const y = h - 55;

    ctx.font = '500 11px "Inter", sans-serif';
    ctx.fillStyle = '#dce0e8';
    ctx.fillText(`DETECTIONS: ${count}`, 18, y);

    ctx.font = '400 9px "JetBrains Mono", monospace';
    ctx.fillStyle = '#ffb400';
    ctx.fillText('MODEL: COCO-SSD | MobileNet v2', 18, y + 16);

    ctx.restore();
  }

  _drawDistanceLegend(ctx, w, h) {
    ctx.save();
    const lx = w - 160;
    let ly = h - 80;

    const items = [
      { color: '#00ff80', label: 'SAFE (>3m)' },
      { color: '#ffc800', label: 'NEAR (1.5-3m)' },
      { color: '#ff4040', label: 'V.CLOSE (<1.5m)' },
    ];

    items.forEach(({ color, label }) => {
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(lx, ly, 4, 0, Math.PI * 2); ctx.fill();
      ctx.font = '400 10px "Inter", sans-serif';
      ctx.fillText(label, lx + 10, ly + 4);
      ly += 16;
    });

    ctx.restore();
  }
}
