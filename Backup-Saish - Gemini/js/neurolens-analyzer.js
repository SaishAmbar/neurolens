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
    this._focalLength = 580;  // Synced with Python config (compensates for lens variation)

    // Known real-world heights in cm
    this._knownHeights = {
      person: 170,
      wall: 200,
      // Furniture
      chair: 90,
      couch: 85,
      sofa: 85,
      bench: 50,
      'dining table': 75,
      bed: 60,
      toilet: 45,
      'potted plant': 60,
      // Vehicles
      car: 150,
      bus: 300,
      truck: 250,
      motorcycle: 110,
      bicycle: 95,
      // Bags / luggage
      backpack: 50,
      handbag: 35,
      suitcase: 65,
      // Appliances / electronics
      refrigerator: 180,
      'cupboard/closet': 180,
      tv: 80,
      laptop: 25,
      'cell phone': 15,
      keyboard: 5,
      monitor: 50,
      microwave: 35,
      oven: 90,
      'washing machine': 90,
      // Food / small items (floor tripping hazards)
      bottle: 30,
      cup: 12,
      bowl: 10,
      vase: 35,
      // Tools / hazards
      scissors: 20,
      knife: 25,
      // Animals
      dog: 55,
      cat: 30,
      horse: 160,
      cow: 140,
      elephant: 280,
      // Street / outdoor
      'fire hydrant': 60,
      'stop sign': 150,
      umbrella: 80,
      // Heuristic categories
      'large obstacle': 180,
      'pole/pillar': 210,
      'step down / stairs': 20,
      'floor hazard': 15,
      'animal': 55,
      'low obstacle': 35,
      // Desktop items
      pen: 15,
      spectacles: 5,
      headset: 18,
      mouse: 5,
      remote: 15,
      book: 22,
      toothbrush: 18,
      'hair drier': 22,
      'wine glass': 10,
    };

    // Whitelist: COCO class → display label
    // 'person' is detected but treated as INFO-only (no danger alerts)
    this._navClassMap = {
      'person':        'person',
      'wall':          'wall',
      // Furniture
      'chair':         'chair',
      'couch':         'couch',
      'sofa':          'couch',
      'dining table':  'table',
      'bench':         'bench',
      'bed':           'large obstacle',
      'toilet':        'large obstacle',
      // Vehicles
      'car':           'vehicle',
      'bus':           'vehicle',
      'truck':         'vehicle',
      'motorcycle':    'scooter/bike',
      'bicycle':       'scooter/bike',
      // Bags / luggage (tripping hazards)
      'backpack':      'bag on floor',
      'handbag':       'bag on floor',
      'suitcase':      'bag on floor',
      // Appliances & electronics
      'refrigerator':  'cupboard/closet',
      'tv':            'large obstacle',
      'laptop':        'low obstacle',
      'keyboard':      'floor hazard',
      'microwave':     'low obstacle',
      'oven':          'large obstacle',
      // Small floor hazards
      'bottle':        'floor hazard',
      'cup':           'floor hazard',
      'bowl':          'floor hazard',
      'vase':          'floor hazard',
      'scissors':      'floor hazard',
      'knife':         'floor hazard',
      // Animals
      'dog':           'animal',
      'cat':           'animal',
      'horse':         'animal',
      'cow':           'animal',
      'elephant':      'animal',
      // Outdoor
      'fire hydrant':  'fire hydrant',
      'stop sign':     'stop sign',
      'umbrella':      'umbrella',
      'potted plant':  'potted plant',
      // Clocks often misclassified as fans
      'clock':         'fan/clock',
    };

    // Whitelist: COCO class → display label for Desktop mode
    this._desktopClassMap = {
      'toothbrush':    'pen',
      'scissors':      'scissors/spectacles',
      'knife':         'pen',
      'wine glass':    'glass/cup',
      'cup':           'cup/headset',
      'teddy bear':    'headset/toy',
      'mouse':         'mouse',
      'keyboard':      'keyboard',
      'cell phone':    'cell phone',
      'remote':        'remote',
      'book':          'book',
      'bottle':        'bottle',
      'bowl':          'bowl/spectacles',
      'laptop':        'laptop',
      'vase':          'vase',
      'clock':         'clock',
      'pen':           'pen',
      'spectacles':    'spectacles',
      'headset':       'headset',
      'wall':          'wall',
    };

    // Tracking state for anti-flicker & stabilization
    this._trackedObjects = [];
    this._nextTrackId = 1;
    this._holdFrames = 15; // Hold lost objects for ~500ms (15 frames)
    this._confirmFrames = 3; // Confirm objects after 3 frames of consecutive detections

    // Alert cooldown state
    this._alertCooldowns = {};

    // Zone thresholds in cm (synced with Python config)
    this._VERY_CLOSE_CM = 100;      // < 1.0m
    this._NEAR_CM = 200;            // 1.0m – 2.0m
    this._MAX_DISTANCE_CM = 300;    // Ignore > 3m

    this._VERY_CLOSE_RATIO = 0.60;  // bbox > 60% frame height
    this._NEAR_RATIO = 0.30;        // bbox 30-60%

    // Zone colors for canvas rendering
    this._zoneColors = {
      'SAFE':       { main: '#00ff80', glow: 'rgba(0, 255, 128, 0.08)' },
      'NEAR':       { main: '#ffc800', glow: 'rgba(255, 200, 0, 0.08)' },
      'VERY CLOSE': { main: '#ff4040', glow: 'rgba(255, 64, 64, 0.12)' },
      'INFO':       { main: '#60a0ff', glow: 'rgba(96, 160, 255, 0.06)' },
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
        base: 'mobilenet_v2', // More accurate model for real-time
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
  async analyze(video, ctx, canvasWidth, canvasHeight, isMirrored = false, mode = 'navigation') {
    if (!this._loaded || !this._model) {
      return this._emptyResult();
    }

    this._frameCount++;

    // Run detection with error handling
    let predictions;
    try {
      predictions = await this._model.detect(video);
    } catch (err) {
      console.error('[NeuroLens] Detection error:', err);
      return this._emptyResult();
    }

    // Throttled logging for debugging
    if (this._frameCount % 30 === 0) {
      console.log("[NeuroLens] Raw predictions:", predictions);
    }

    // Filter to whitelisted classes and apply heuristics
    const tempPredictions = [];
    for (const p of predictions) {
      const [x, y, w, h] = p.bbox;
      const areaRatio = (w * h) / (canvasWidth * canvasHeight);

      // Dynamic confidence threshold: lower the score limit to 0.12 if the object covers
      // a significant portion of the screen (>= 12% area), which indicates a close/cropped threat.
      const minScore = areaRatio >= 0.12 ? 0.12 : 0.25;
      if (p.score < minScore) continue;

      const cX = x + w / 2;
      const cY = y + h / 2;

      // Check whitelist
      const activeMap = mode === 'desktop' ? this._desktopClassMap : this._navClassMap;
      let mappedClass = activeMap[p.class];

      // Desktop Heuristic: remote with aspect ratio < 2.0 is likely a mouse
      if (mode === 'desktop' && p.class === 'remote') {
        const aspect = w > h ? w / h : h / w;
        if (aspect < 2.0) {
          mappedClass = 'mouse';
        }
      }

      // Heuristic 1: Large nearby unlabelled blob → 'large obstacle' (only in navigation mode)
      if (!mappedClass && mode === 'navigation') {
        const isLargeObstacle = areaRatio > 0.10;
        if (isLargeObstacle) {
          mappedClass = 'large obstacle';
        }
      }

      // Heuristic 2: Tall and thin → 'pole/pillar' (only in navigation mode)
      if (!mappedClass && mode === 'navigation') {
        const isPillar = (h / canvasHeight > 0.40) && (w / canvasWidth < 0.12) && (h / w > 3.0);
        if (isPillar) {
          mappedClass = 'pole/pillar';
        }
      }

      // Heuristic 3: Wide flat band near bottom of frame → 'step down / stairs' (only in navigation mode)
      if (!mappedClass && mode === 'navigation') {
        const isStep = (cY / canvasHeight > 0.65) && (h / canvasHeight < 0.22) && (w / canvasWidth > 0.30);
        if (isStep) {
          mappedClass = 'step down / stairs';
        }
      }

      // Heuristic 4: Small low item at bottom of frame → 'floor hazard' (only in navigation mode)
      if (!mappedClass && mode === 'navigation') {
        const isFloorHazard = (cY / canvasHeight > 0.75) && (areaRatio < 0.04) && p.score > 0.30;
        if (isFloorHazard) {
          mappedClass = 'floor hazard';
        }
      }

      if (mappedClass) {
        tempPredictions.push({
          class: mappedClass,
          score: p.score,
          bbox: p.bbox,
          isReclassifiedFan: mappedClass === 'fan/clock' && p.class === 'person'
        });
      }
    }

    // Filter out duplicate clock detections that overlap with a larger reclassified pedestal fan
    const predictionsWithHeuristics = [];
    const reclassifiedFans = tempPredictions.filter(tp => tp.isReclassifiedFan);

    for (const tp of tempPredictions) {
      if (tp.class === 'fan/clock' && !tp.isReclassifiedFan) {
        const [qx, qy, qw, qh] = tp.bbox;
        let isDuplicate = false;
        for (const fan of reclassifiedFans) {
          const [fx, fy, fw, fh] = fan.bbox;
          if (fw > qw * 1.2 && fh > qh * 1.5) {
            const horizInside = qx >= fx - 15 && (qx + qw) <= (fx + fw + 15);
            const vertInside = qy >= fy - 15 && (qy + qh) <= (fy + fh + 15);
            if (horizInside && vertInside) {
              isDuplicate = true;
              break;
            }
          }
        }
        if (isDuplicate) continue;
      }
      predictionsWithHeuristics.push(tp);
    }

    // Centroid-based matching and tracking for anti-flicker stabilization & motion detection
    for (const pred of predictionsWithHeuristics) {
      const [x, y, w, h] = pred.bbox;
      const cX = x + w / 2;
      const cY = y + h / 2;

      let bestMatch = null;
      let minDistance = 80; // association threshold in pixels

      for (const tObj of this._trackedObjects) {
        const dx = tObj.centerX - cX;
        const dy = tObj.centerY - cY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < minDistance) {
          minDistance = dist;
          bestMatch = tObj;
        }
      }

      if (bestMatch) {
        // Update matched object
        const dx = cX - bestMatch.centerX;
        const dy = cY - bestMatch.centerY;
        const velocity = Math.sqrt(dx * dx + dy * dy);

        bestMatch.bbox = pred.bbox;
        bestMatch.centerX = cX;
        bestMatch.centerY = cY;
        bestMatch.class = pred.class;
        bestMatch.confidence = pred.score;
        bestMatch.framesDetected++;
        bestMatch.framesLost = 0;
        bestMatch.velocity = velocity;
        bestMatch.isMoving = velocity > 12; // flagged as moving if centroid shifts fast
        bestMatch.matched = true;
      } else {
        // New tracked object candidate
        this._trackedObjects.push({
          id: this._nextTrackId++,
          class: pred.class,
          confidence: pred.score,
          bbox: pred.bbox,
          centerX: cX,
          centerY: cY,
          framesDetected: 1,
          framesLost: 0,
          velocity: 0,
          isMoving: false,
          isConfirmed: false,
          matched: true
        });
      }
    }

    // Stale/Hold confirmation update
    // Desktop mode: confirm immediately (1 frame) so small objects are shown right away
    // Navigation mode: require 3 frames to suppress flicker
    const effectiveConfirmFrames = mode === 'desktop' ? 1 : this._confirmFrames;
    for (const tObj of this._trackedObjects) {
      if (!tObj.matched) {
        tObj.framesLost++;
      } else {
        tObj.matched = false; // Reset matching flag for next frame
      }

      if (tObj.framesDetected >= effectiveConfirmFrames) {
        tObj.isConfirmed = true;
      }
    }

    // Keep confirmed or held objects, filter out stale ones
    this._trackedObjects = this._trackedObjects.filter(tObj => tObj.framesLost < this._holdFrames);

    // Apply distance gate and threat level classification
    const results = [];
    const activeDetections = this._trackedObjects.filter(tObj => tObj.isConfirmed);

    const maxDistanceCm = mode === 'desktop' ? 150 : this._MAX_DISTANCE_CM;

    for (const obj of activeDetections) {
      const [x, y, w, h] = obj.bbox;
      const estimatedCm = this._estimateDistance(obj.class, h, canvasHeight);

      // Distance gate: ignore objects > maxDistanceCm away
      if (estimatedCm > maxDistanceCm) continue;

      // Person = informational only → INFO zone, zero danger
      const isPerson = obj.class === 'person';
      const zone = isPerson ? 'INFO' : this._classifyZoneDynamic(estimatedCm, h, canvasHeight, mode);
      const rawDirection = this._classifyDirection(obj.centerX, canvasWidth);
      const direction = isMirrored ? (rawDirection === 'LEFT' ? 'RIGHT' : (rawDirection === 'RIGHT' ? 'LEFT' : 'CENTER')) : rawDirection;
      const dangerScore = isPerson ? 0 : this._calculateDangerDynamic(estimatedCm, obj.centerX, canvasWidth, obj.class, mode);

      results.push({
        id: obj.id,
        class: obj.class,
        confidence: obj.confidence,
        bbox: obj.bbox,
        centerX: obj.centerX,
        centerY: obj.centerY,
        estimatedCm,
        zone,
        direction,
        dangerScore,
        isPerson,
        isMoving: obj.isMoving
      });
    }

    // Check for flat, featureless wall close up (only in navigation mode)
    const isWallAhead = mode === 'navigation' && this._detectWall(video, canvasWidth, canvasHeight);
    if (isWallAhead) {
      results.push({
        id: 9999,
        class: 'wall',
        confidence: 0.98,
        bbox: [0, 0, canvasWidth, canvasHeight],
        centerX: canvasWidth / 2,
        centerY: canvasHeight / 2,
        estimatedCm: 60, // 0.6m
        zone: 'VERY CLOSE',
        direction: 'CENTER',
        dangerScore: 0.98,
        isMoving: false
      });
    }

    // Sort by danger score (most dangerous first)
    results.sort((a, b) => b.dangerScore - a.dangerScore);

    // Draw HUD
    const elapsed = (Date.now() - this._startTime) / 1000;
    this._drawHUD(ctx, canvasWidth, canvasHeight, results, elapsed, isMirrored, mode);

    // Build final context-aware result object
    // Pick the first non-person result as the primary threat (persons are info-only)
    const mostDangerous = results.find(r => !r.isPerson) || null;
    let alertZone = 'SAFE';
    let alertLabel = 'SAFE PATH';
    let cooldownActive = false;
    // Track the estimated cm for DANGER:STOP threshold
    if (mostDangerous) mostDangerous.estimatedCm = mostDangerous.estimatedCm || 999;

    if (mostDangerous) {
      alertZone = mostDangerous.zone;

      // Map to navigation-assistant / desktop alert labels
      if (mode === 'desktop') {
        if (alertZone === 'VERY CLOSE') {
          alertLabel = `CLOSE: ${mostDangerous.class.toUpperCase()}`;
        } else if (alertZone === 'NEAR') {
          alertLabel = `NEAR: ${mostDangerous.class.toUpperCase()}`;
        } else {
          alertLabel = `DETECTED: ${mostDangerous.class.toUpperCase()}`;
        }
      } else {
        if (mostDangerous.class === 'wall') {
          alertLabel = 'DANGER: WALL CLOSE';
        } else if (mostDangerous.class === 'step down / stairs') {
          alertLabel = 'STEP DOWN';
        } else if (alertZone === 'VERY CLOSE' && mostDangerous.estimatedCm < 50) {
          if (mostDangerous.direction === 'LEFT') {
            alertLabel = 'MOVE RIGHT — STOP';
          } else if (mostDangerous.direction === 'RIGHT') {
            alertLabel = 'MOVE LEFT — STOP';
          } else {
            const goRight = mostDangerous.centerX < canvasWidth / 2;
            alertLabel = goRight ? 'MOVE RIGHT — STOP' : 'MOVE LEFT — STOP';
          }
        } else if (alertZone === 'VERY CLOSE') {
          if (mostDangerous.direction === 'LEFT') {
            alertLabel = 'MOVE RIGHT';
          } else if (mostDangerous.direction === 'RIGHT') {
            alertLabel = 'MOVE LEFT';
          } else {
            const goRight = mostDangerous.centerX < canvasWidth / 2;
            alertLabel = goRight ? 'MOVE RIGHT' : 'MOVE LEFT';
          }
        } else if (mostDangerous.isMoving) {
          const goRight = mostDangerous.centerX < canvasWidth / 2;
          alertLabel = goRight ? 'MOVE RIGHT' : 'MOVE LEFT';
        } else if (mostDangerous.direction === 'LEFT') {
          alertLabel = 'MOVE RIGHT';  // obstacle is LEFT → tell user to go RIGHT
        } else if (mostDangerous.direction === 'RIGHT') {
          alertLabel = 'MOVE LEFT';   // obstacle is RIGHT → tell user to go LEFT
        } else if (mostDangerous.direction === 'CENTER') {
          const goRight = mostDangerous.centerX < canvasWidth / 2;
          alertLabel = goRight ? 'MOVE RIGHT' : 'MOVE LEFT';
        }
      }

      // Check Warning Cooldown (3 seconds)
      const now = Date.now();
      const cooldownKey = `${mostDangerous.class}_${alertZone}`;
      const lastAlertTime = this._alertCooldowns[cooldownKey] || 0;

      if (now - lastAlertTime < 3000) {
        cooldownActive = true;
      } else {
        this._alertCooldowns[cooldownKey] = now;
      }
    }

    return {
      detectionCount: results.length,
      alertZone,
      alertLabel,
      cooldownActive,
      mostDangerous: mostDangerous ? {
        class: mostDangerous.class,
        zone: mostDangerous.zone,
        direction: mostDangerous.direction,
        distanceM: (mostDangerous.estimatedCm / 100).toFixed(1),
        dangerScore: mostDangerous.dangerScore.toFixed(2),
        isMoving: mostDangerous.isMoving
      } : null,
      detections: results.map(r => ({
        class: r.class,
        zone: r.zone,
        direction: r.direction,
        distanceM: (r.estimatedCm / 100).toFixed(1),
        confidence: (r.confidence * 100).toFixed(0) + '%',
        isMoving: r.isMoving
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

  _classifyZoneDynamic(estimatedCm, bboxHeight, frameHeight, mode) {
    const ratio = bboxHeight / frameHeight;
    const veryCloseCm = mode === 'desktop' ? 30 : this._VERY_CLOSE_CM;
    const nearCm = mode === 'desktop' ? 60 : this._NEAR_CM;
    const veryCloseRatio = mode === 'desktop' ? 0.70 : this._VERY_CLOSE_RATIO;
    const nearRatio = mode === 'desktop' ? 0.40 : this._NEAR_RATIO;

    if (estimatedCm < veryCloseCm || ratio > veryCloseRatio)
      return 'VERY CLOSE';
    if (estimatedCm < nearCm || ratio > nearRatio)
      return 'NEAR';
    return 'SAFE';
  }

  _classifyDirection(centerX, frameWidth) {
    const third = frameWidth / 3;
    if (centerX < third) return 'LEFT';
    if (centerX > third * 2) return 'RIGHT';
    return 'CENTER';
  }

  _calculateDangerDynamic(estimatedCm, centerX, frameWidth, className, mode) {
    const maxDist = mode === 'desktop' ? 150 : 500;
    const distFactor = Math.max(0, 1.0 - estimatedCm / maxDist);
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

  _drawHUD(ctx, w, h, results, t, isMirrored = false, mode = 'navigation') {
    ctx.save();

    // ── Grid ──
    this._drawGrid(ctx, w, h);

    // ── Safe Corridor / Navigation Arrow (only in navigation mode) ──
    if (results.length === 0 && mode === 'navigation') {
      this._drawSafeCorridor(ctx, w, h, t);
    } else if (mode === 'desktop') {
      // Draw desktop target crosshair overlay in background
      ctx.save();
      ctx.strokeStyle = 'rgba(0, 229, 255, 0.05)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(w / 2, h / 2, 80, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath(); ctx.arc(w / 2, h / 2, 160, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(w / 2 - 200, h / 2); ctx.lineTo(w / 2 + 200, h / 2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(w / 2, h / 2 - 150); ctx.lineTo(w / 2, h / 2 + 150); ctx.stroke();
      ctx.restore();
    }

    // ── Scan line ──
    this._drawScanLine(ctx, w, h, t);

    // ── Draw ALL confirmed obstacles ──
    for (let i = results.length - 1; i >= 0; i--) {
      const isPrimary = (i === 0);
      this._drawDetectionBox(ctx, results[i], isPrimary, t, w, h, isMirrored);
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
    this._drawWarningPanel(ctx, w, mostDangerous, t, mode);

    // ── Direction indicators (live dynamic safety zones) ──
    this._drawDirectionIndicators(ctx, w, h, results, mostDangerous, t, isMirrored, mode);

    // ── Stats footer ──
    this._drawStatsFooter(ctx, w, h, results.length);

    // ── Distance legend ──
    this._drawDistanceLegend(ctx, w, h, mode);

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

  _drawDetectionBox(ctx, result, isPrimary, t, canvasW, canvasH, isMirrored = false) {
    const { bbox, zone, class: className, confidence, estimatedCm, isMoving } = result;
    // COCO-SSD returns bbox as [x, y, w, h] array — destructure correctly
    const [x, y, w, h] = bbox;
    const colors = this._zoneColors[zone];
    const color = colors.main;
    const thickness = isPrimary ? 3 : 1.5;

    // Get mirrored horizontal coordinate for drawing
    const rx = isMirrored ? (canvasW - x - w) : x;

    ctx.save();

    // ── Glow or translucent fill ──
    if (isPrimary) {
      const pulse = 0.5 + 0.5 * Math.sin(t * 4);
      ctx.globalAlpha = 0.3 * pulse;
      ctx.strokeStyle = color;
      ctx.lineWidth = 6;
      ctx.strokeRect(rx - 2, y - 2, w + 4, h + 4);
      ctx.globalAlpha = 1;
    } else {
      ctx.globalAlpha = 0.15;
      ctx.fillStyle = color;
      ctx.fillRect(rx, y, w, h);
      ctx.globalAlpha = 0.6;
    }

    // ── Corner brackets ──
    const cornerLen = Math.min(15, w / 4, h / 4);
    ctx.strokeStyle = color;
    ctx.lineWidth = thickness;

    // Top-left
    ctx.beginPath(); ctx.moveTo(rx, y + cornerLen); ctx.lineTo(rx, y); ctx.lineTo(rx + cornerLen, y); ctx.stroke();
    // Top-right
    ctx.beginPath(); ctx.moveTo(rx + w - cornerLen, y); ctx.lineTo(rx + w, y); ctx.lineTo(rx + w, y + cornerLen); ctx.stroke();
    // Bottom-left
    ctx.beginPath(); ctx.moveTo(rx, y + h - cornerLen); ctx.lineTo(rx, y + h); ctx.lineTo(rx + cornerLen, y + h); ctx.stroke();
    // Bottom-right
    ctx.beginPath(); ctx.moveTo(rx + w - cornerLen, y + h); ctx.lineTo(rx + w, y + h); ctx.lineTo(rx + w, y + h - cornerLen); ctx.stroke();

    // ── Dashed connecting lines ──
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    // Top
    ctx.beginPath(); ctx.moveTo(rx + cornerLen, y); ctx.lineTo(rx + w - cornerLen, y); ctx.stroke();
    // Bottom
    ctx.beginPath(); ctx.moveTo(rx + cornerLen, y + h); ctx.lineTo(rx + w - cornerLen, y + h); ctx.stroke();
    // Left
    ctx.beginPath(); ctx.moveTo(rx, y + cornerLen); ctx.lineTo(rx, y + h - cornerLen); ctx.stroke();
    // Right
    ctx.beginPath(); ctx.moveTo(rx + w, y + cornerLen); ctx.lineTo(rx + w, y + h - cornerLen); ctx.stroke();
    ctx.setLineDash([]);

    // ── Distance Ruler Ticks (Primary box only) ──
    if (isPrimary) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      const tickW = 5;
      for (let f = 0.25; f <= 0.75; f += 0.25) {
        const ty = y + h * f;
        ctx.beginPath();
        ctx.moveTo(rx, ty);
        ctx.lineTo(rx + tickW, ty);
        ctx.stroke();
      }
    }

    // ── Label ──
    const distM = (estimatedCm / 100).toFixed(1);
    const movingSuffix = isMoving ? ' | MOVING' : '';
    const label = `${className.toUpperCase()} | ${zone}${movingSuffix} | ${distM}m`;
    ctx.font = '600 11px "Inter", sans-serif';
    const labelMetrics = ctx.measureText(label);
    const labelW = labelMetrics.width + 12;
    const labelH = 20;
    const labelX = rx;
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
    ctx.fillText(confLabel, rx + w - confMetrics.width - 2, y + h + 14);

    // ── Crosshair and Motion Vector on Primary ──
    if (isPrimary) {
      const rcx = isMirrored ? (canvasW - result.centerX) : result.centerX;
      const cy = result.centerY;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(rcx - 10, cy); ctx.lineTo(rcx + 10, cy); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(rcx, cy - 10); ctx.lineTo(rcx, cy + 10); ctx.stroke();
      ctx.beginPath(); ctx.arc(rcx, cy, 12, 0, Math.PI * 2); ctx.stroke();

      // Velocity Arrow for moving obstacles
      if (isMoving) {
        ctx.strokeStyle = '#ff3b30'; // Dynamic action red vector
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(rcx, cy);
        const arrowX = isMirrored ? (rcx - 25) : (rcx + 25);
        ctx.lineTo(arrowX, cy - 15);
        ctx.stroke();

        ctx.fillStyle = '#ff3b30';
        ctx.beginPath();
        ctx.arc(arrowX, cy - 15, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
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

  _drawSafeCorridor(ctx, w, h, t) {
    ctx.save();

    const horizonY = h * 0.4;
    const horizonW = w * 0.1;
    const bottomW = w * 0.5;

    const grad = ctx.createLinearGradient(0, horizonY, 0, h);
    grad.addColorStop(0, 'rgba(0, 255, 128, 0.0)');
    grad.addColorStop(0.5, 'rgba(0, 255, 128, 0.05)');
    grad.addColorStop(1, 'rgba(0, 255, 128, 0.15)');

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(w / 2 - horizonW / 2, horizonY);
    ctx.lineTo(w / 2 + horizonW / 2, horizonY);
    ctx.lineTo(w / 2 + bottomW / 2, h - 50);
    ctx.lineTo(w / 2 - bottomW / 2, h - 50);
    ctx.closePath();
    ctx.fill();

    // Corridor borders
    ctx.strokeStyle = 'rgba(0, 255, 128, 0.4)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([8, 6]);

    ctx.beginPath();
    ctx.moveTo(w / 2 - horizonW / 2, horizonY);
    ctx.lineTo(w / 2 - bottomW / 2, h - 50);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(w / 2 + horizonW / 2, horizonY);
    ctx.lineTo(w / 2 + bottomW / 2, h - 50);
    ctx.stroke();

    // Flowing safe navigation arrow
    const pulseY = (h - 80) - ((t * 40) % 60);
    const arrowAlpha = 1.0 - ((t * 40) % 60) / 60;

    ctx.strokeStyle = `rgba(0, 255, 128, ${arrowAlpha * 0.6})`;
    ctx.lineWidth = 3;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(w / 2 - 12, pulseY + 8);
    ctx.lineTo(w / 2, pulseY);
    ctx.lineTo(w / 2 + 12, pulseY + 8);
    ctx.stroke();

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

  _getWarningTextAndIcon(mostDangerous, mode = 'navigation', frameWidth = 640) {
    if (!mostDangerous) {
      return mode === 'desktop' ? { label: 'DESK AREA CLEAR', icon: '✓' } : { label: 'SAFE PATH', icon: '✓' };
    }

    let label = 'OBJECT AHEAD';
    let icon = '⚠';

    if (mode === 'desktop') {
      if (mostDangerous.zone === 'VERY CLOSE') {
        label = `CLOSE: ${mostDangerous.class.toUpperCase()}`;
        icon = '🛑';
      } else {
        label = `DETECTED: ${mostDangerous.class.toUpperCase()}`;
        icon = '🔍';
      }
    } else {
      if (mostDangerous.class === 'wall') {
        label = 'DANGER: WALL CLOSE';
        icon = '🛑';
      } else if (mostDangerous.class === 'step down / stairs') {
        label = 'STEP DOWN';
        icon = '🪜';
      } else if (mostDangerous.zone === 'VERY CLOSE' && mostDangerous.estimatedCm < 50) {
        if (mostDangerous.direction === 'LEFT') {
          label = 'MOVE RIGHT — STOP';
        } else if (mostDangerous.direction === 'RIGHT') {
          label = 'MOVE LEFT — STOP';
        } else {
          const goRight = mostDangerous.centerX < frameWidth / 2;
          label = goRight ? 'MOVE RIGHT — STOP' : 'MOVE LEFT — STOP';
        }
        icon = '🛑';
      } else if (mostDangerous.zone === 'VERY CLOSE') {
        if (mostDangerous.direction === 'LEFT') {
          label = 'MOVE RIGHT';
        } else if (mostDangerous.direction === 'RIGHT') {
          label = 'MOVE LEFT';
        } else {
          const goRight = mostDangerous.centerX < frameWidth / 2;
          label = goRight ? 'MOVE RIGHT' : 'MOVE LEFT';
        }
        icon = '⛔';
      } else if (mostDangerous.direction === 'LEFT') {
        label = 'MOVE RIGHT';
        icon = '▶';
      } else if (mostDangerous.direction === 'RIGHT') {
        label = 'MOVE LEFT';
        icon = '◀';
      } else {
        const goRight = mostDangerous.centerX < frameWidth / 2;
        label = goRight ? 'MOVE RIGHT' : 'MOVE LEFT';
        icon = goRight ? '▶' : '◀';
      }
    }

    return { label, icon };
  }

  _drawWarningPanel(ctx, w, mostDangerous, t, mode = 'navigation') {
    const { label, icon } = this._getWarningTextAndIcon(mostDangerous, mode, w);
    const zone = mostDangerous ? mostDangerous.zone : 'SAFE';
    const colors = this._zoneColors[zone];

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
    const alertText = `${icon} ${label}`;
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

  _drawDirectionIndicators(ctx, w, h, results, mostDangerous, t, isMirrored = false, mode = 'navigation') {
    const barH = 30;
    const barY = h - barH - 12;
    const sectionW = w / 3;
    const dirs = ['LEFT', 'CENTER', 'RIGHT'];
    const arrows = mode === 'desktop' ? ['LEFT DESK', 'CENTER DESK', 'RIGHT DESK'] : ['<< LEFT', '[ CENTER ]', 'RIGHT >>'];

    const closestPerDir = { LEFT: null, CENTER: null, RIGHT: null };
    for (const r of (results || [])) {
      if (!closestPerDir[r.direction] || r.estimatedCm < closestPerDir[r.direction].estimatedCm) {
        closestPerDir[r.direction] = r;
      }
    }

    ctx.save();

    dirs.forEach((dir, i) => {
      // Map screen column index to data direction
      // Screen column index 0 corresponds to data 'LEFT'
      let dataDir = dir;
      const sx = i * sectionW;
      const obstacle = closestPerDir[dataDir];
      const isActive = mostDangerous && dataDir === mostDangerous.direction;

      let bg = 'rgba(15, 15, 15, 0.45)';
      let textColor = '#8a8aaa';
      let border = 'rgba(64, 80, 80, 0.3)';

      if (obstacle) {
        const colors = this._zoneColors[obstacle.zone];
        bg = colors.glow;
        textColor = colors.main;
        border = colors.main;
      }

      ctx.fillStyle = bg;
      ctx.fillRect(sx + 2, barY, sectionW - 4, barH);

      if (obstacle) {
        ctx.strokeStyle = border;
        ctx.lineWidth = isActive ? 2 : 1;
        ctx.strokeRect(sx + 2, barY, sectionW - 4, barH);
      }

      ctx.font = `${isActive ? '700' : '400'} 11px "Inter", sans-serif`;
      ctx.fillStyle = isActive ? '#ffffff' : textColor;
      const label = arrows[i];
      const tw = ctx.measureText(label).width;
      ctx.fillText(label, sx + (sectionW - tw) / 2, barY + barH / 2 + 4);

      if (i > 0) {
        ctx.strokeStyle = '#405050';
        ctx.lineWidth = 0.5;
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
    ctx.fillText('MODEL: COCO-SSD | Navigation Mode', 18, y + 16);

    ctx.restore();
  }

  _drawDistanceLegend(ctx, w, h, mode = 'navigation') {
    ctx.save();
    const lx = w - 160;
    let ly = h - 80;

    const items = mode === 'desktop' ? [
      { color: '#00ff80', label: 'SAFE (>0.6m)' },
      { color: '#ffc800', label: 'NEAR (0.3-0.6m)' },
      { color: '#ff4040', label: 'V.CLOSE (<0.3m)' },
    ] : [
      { color: '#00ff80', label: 'SAFE (>2.0m)' },
      { color: '#ffc800', label: 'NEAR (1.0-2.0m)' },
      { color: '#ff4040', label: 'V.CLOSE (<1.0m)' },
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

  _detectWall(video, width, height) {
    try {
      if (!video || video.readyState < 2) return false;

      // Get offscreen context
      if (!this._offscreenCanvas) {
        this._offscreenCanvas = document.createElement('canvas');
        this._offscreenCanvas.width = 64;
        this._offscreenCanvas.height = 64;
        this._offscreenCtx = this._offscreenCanvas.getContext('2d', { willReadFrequently: true });
      }

      this._offscreenCtx.drawImage(video, 0, 0, 64, 64);
      const imgData = this._offscreenCtx.getImageData(0, 0, 64, 64);
      const data = imgData.data;

      let totalLuminance = 0;
      const pixelCount = 64 * 64;
      const luminance = new Float32Array(pixelCount);

      for (let i = 0; i < pixelCount; i++) {
        const r = data[i * 4];
        const g = data[i * 4 + 1];
        const b = data[i * 4 + 2];
        const lum = 0.299 * r + 0.587 * g + 0.114 * b;
        luminance[i] = lum;
        totalLuminance += lum;
      }

      const avgLuminance = totalLuminance / pixelCount;

      // If it's too dark (pitch black / camera covered), don't trigger wall warning
      if (avgLuminance < 22) return false;

      // Calculate horizontal and vertical absolute gradients
      let totalDiff = 0;
      let count = 0;

      for (let y = 0; y < 64; y++) {
        for (let x = 0; x < 64; x++) {
          const idx = y * 64 + x;
          if (x < 63) {
            totalDiff += Math.abs(luminance[idx] - luminance[idx + 1]);
            count++;
          }
          if (y < 63) {
            totalDiff += Math.abs(luminance[idx] - luminance[idx + 64]);
            count++;
          }
        }
      }

      const avgGrad = totalDiff / count;

      // If the scene is extremely uniform (low texture variation), it's a flat wall close up
      return avgGrad < 2.0;
    } catch (e) {
      console.warn("[NeuroLens] Wall detection error:", e);
      return false;
    }
  }
}
