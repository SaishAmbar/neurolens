"""
NeuroLens — YOLO Object Detector
Handles YOLOv4-tiny model loading, auto-downloading, and detection pipeline.
"""

import os
import urllib.request
import sys
import cv2
import numpy as np
import time

from . import config


class Detection:
    """Represents a single detected object with threat statistics."""

    __slots__ = ("class_name", "class_id", "confidence", "bbox", "center_x", "center_y",
                 "estimated_cm", "zone", "direction", "danger_score", "is_moving", "cooldown_active")

    def __init__(self, class_name, class_id, confidence, bbox, estimated_cm=0.0, zone="SAFE", 
                 direction="CENTER", danger_score=0.0, is_moving=False, cooldown_active=False):
        self.class_name = class_name
        self.class_id = class_id
        self.confidence = confidence
        self.bbox = bbox  # (x, y, w, h)
        self.center_x = bbox[0] + bbox[2] // 2
        self.center_y = bbox[1] + bbox[3] // 2
        self.estimated_cm = estimated_cm
        self.zone = zone
        self.direction = direction
        self.danger_score = danger_score
        self.is_moving = is_moving
        self.cooldown_active = cooldown_active

    def __repr__(self):
        return (
            f"Detection({self.class_name}, conf={self.confidence:.2f}, "
            f"dist={self.estimated_cm:.1f}cm, zone={self.zone}, dir={self.direction}, moving={self.is_moving})"
        )


class Detector:
    """YOLOv4-tiny based object detector using OpenCV DNN."""

    def __init__(self):
        self._ensure_model_files()
        self._load_class_names()
        self._load_network()
        self._frame_count = 0
        
        # Anti-flicker, stabilization & velocity tracking states
        self._tracked_objects = []
        self._next_track_id = 1
        self._alert_cooldowns = {}

    # ── Model Management ──────────────────────────────────────────────────

    def _ensure_model_files(self):
        """Download YOLO model files if not present."""
        os.makedirs(config.MODELS_DIR, exist_ok=True)

        files_to_download = [
            (config.YOLO_CFG_PATH, config.YOLO_CFG_URL, "YOLOv4-tiny config"),
            (config.YOLO_WEIGHTS_PATH, config.YOLO_WEIGHTS_URL, "YOLOv4-tiny weights (~23MB)"),
            (config.COCO_NAMES_PATH, config.COCO_NAMES_URL, "COCO class names"),
        ]

        for filepath, url, description in files_to_download:
            if not os.path.exists(filepath):
                print(f"[NeuroLens] Downloading {description}...")
                print(f"           URL: {url}")
                try:
                    self._download_with_progress(url, filepath)
                    print(f"           ✓ Saved to {filepath}")
                except Exception as e:
                    print(f"[NeuroLens] ERROR: Failed to download {description}: {e}")
                    print(f"           Please manually download from: {url}")
                    print(f"           And place it at: {filepath}")
                    sys.exit(1)

    @staticmethod
    def _download_with_progress(url, filepath):
        """Download a file with progress reporting."""
        response = urllib.request.urlopen(url)
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        block_size = 8192

        with open(filepath, "wb") as f:
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = downloaded * 100 / total_size
                    bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
                    print(f"\r           [{bar}] {pct:.1f}%", end="", flush=True)
            print()  # newline after progress bar

    def _load_class_names(self):
        """Load COCO class names from file."""
        with open(config.COCO_NAMES_PATH, "r") as f:
            self._all_classes = [line.strip() for line in f.readlines()]
        print(f"[NeuroLens] Loaded {len(self._all_classes)} COCO classes")

    def _load_network(self):
        """Load YOLOv4-tiny network and configure backend."""
        print("[NeuroLens] Loading YOLOv4-tiny network...")
        self._net = cv2.dnn.readNetFromDarknet(config.YOLO_CFG_PATH, config.YOLO_WEIGHTS_PATH)

        # Check if CUDA is actually available in this OpenCV build
        cuda_available = False
        try:
            build_info = cv2.getBuildInformation()
            if "CUDA" in build_info and "YES" in build_info.split("CUDA")[1].split("\n")[0]:
                cuda_available = True
        except Exception:
            pass

        if cuda_available:
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("[NeuroLens] Using CUDA backend (GPU)")
        else:
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("[NeuroLens] Using CPU backend")

        # Get output layer names
        layer_names = self._net.getLayerNames()
        out_layers = self._net.getUnconnectedOutLayers()
        self._output_layers = [layer_names[i - 1] for i in out_layers.flatten()]
        print(f"[NeuroLens] ✓ Network ready ({len(self._output_layers)} output layers)")

    # ── Distance, Direction, and Danger Heuristics ─────────────────────────

    def _estimate_distance(self, class_name, bbox_height, frame_height):
        known_h = config.KNOWN_HEIGHTS_CM.get(class_name)
        if known_h and bbox_height > 0:
            return (known_h * config.FOCAL_LENGTH_PX) / bbox_height
        # Fallback based on height coverage
        ratio = bbox_height / frame_height
        return max(50.0, (1.0 - ratio) * 500.0)

    def _classifyZone(self, estimated_cm, bbox_height, frame_height):
        ratio = bbox_height / frame_height
        if estimated_cm < config.DIST_VERY_CLOSE_CM or ratio > config.BBOX_VERY_CLOSE_RATIO:
            return config.ZONE_VERY_CLOSE
        if estimated_cm < config.DIST_NEAR_CM or ratio > config.BBOX_NEAR_RATIO:
            return config.ZONE_NEAR
        return config.ZONE_SAFE

    def _classifyDirection(self, center_x, frame_width):
        third = frame_width / 3
        if center_x < third:
            return "LEFT"
        if center_x > third * 2:
            return "RIGHT"
        return "CENTER"

    def _calculateDanger(self, estimated_cm, center_x, frame_width, class_name):
        dist_factor = max(0.0, 1.0 - estimated_cm / 500.0)
        center_offset = abs(center_x - frame_width / 2.0) / (frame_width / 2.0)
        center_factor = 1.0 - center_offset * 0.5
        known_h = config.KNOWN_HEIGHTS_CM.get(class_name, 50.0)
        size_factor = min(1.0, known_h / 200.0)
        return min(1.0, max(0.0, dist_factor * 0.6 + center_factor * 0.3 + size_factor * 0.1))

    # ── Detection Pipeline ────────────────────────────────────────────────

    def detect(self, frame, mode='navigation'):
        """
        Run object detection on a frame.

        Args:
            frame: BGR image (numpy array)
            mode: 'navigation' or 'desktop'

        Returns:
            List of Detection objects for relevant obstacles
        """
        self._frame_count += 1
        height, width = frame.shape[:2]

        # 1. Create blob from frame
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1 / 255.0,
            size=config.INPUT_SIZE,
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )

        # 2. Forward pass
        self._net.setInput(blob)
        outputs = self._net.forward(self._output_layers)

        # 3. Process detections with allowed classes and wall/step/pillar heuristics
        boxes = []
        confidences = []
        class_ids = []
        raw_clocks = []

        # First pass: collect raw clock bounding boxes for overlap checks (pedestal fan support)
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if class_id == 74 and confidence > 0.08:  # 74 is clock in COCO
                    cx, cy, w_norm, h_norm = detection[0:4]
                    x_c = int(cx * width - (w_norm * width) / 2)
                    y_c = int(cy * height - (h_norm * height) / 2)
                    w_c = int(w_norm * width)
                    h_c = int(h_norm * height)
                    raw_clocks.append([x_c, y_c, w_c, h_c])

        # Second pass: process and filter detections
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                cx, cy, w_norm, h_norm = detection[0:4]
                w = int(w_norm * width)
                h = int(h_norm * height)
                x = int(cx * width - w / 2)
                y = int(cy * height - h / 2)
                area_ratio = (w * h) / (width * height)

                # Lower threshold for large objects (close-up/cropped)
                required_conf = 0.12 if area_ratio > 0.12 else config.CONFIDENCE_THRESHOLD
                if confidence < required_conf:
                    continue

                if mode == 'desktop':
                    is_whitelisted = class_id in config.DESKTOP_CLASSES
                    is_large_obstacle = False
                    is_pillar = False
                    is_step = False
                else:
                    is_whitelisted = class_id in config.OBSTACLE_CLASSES

                    # Heuristic 1: Wall / Large Obstacle
                    # ONLY applies to NON-whitelisted classes so a close person
                    # is never re-labelled as "large obstacle"
                    # Lowered threshold to 0.12 — captures closer obstacles like table fans
                    is_large_obstacle = (
                        not is_whitelisted and
                        ((area_ratio > 0.12) or (class_id in [62, 72, 59]))
                    )

                    # Heuristic 2: Pole / Pillar (tall, thin, vertically aligned)
                    # Heuristic 3: Step Down / Level Change (short, wide, placed in bottom third)
                    is_pillar = (h / height > 0.45) and (w / width < 0.15) and (h / max(1, w) > 2.5)
                    is_step = (cy > 0.70) and (h / height < 0.20) and (w / width > 0.25)

                if is_whitelisted or is_large_obstacle or is_pillar or is_step:
                    boxes.append([x, y, w, h])
                    confidences.append(confidence)

                    if is_pillar and not is_whitelisted:
                        class_ids.append(-2)
                    elif is_step and not is_whitelisted:
                        class_ids.append(-3)
                    elif is_large_obstacle:
                        class_ids.append(-1)
                    else:
                        class_ids.append(class_id)

        # 4. Apply Non-Maximum Suppression
        if len(boxes) == 0:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, 0.12, config.NMS_THRESHOLD  # Use 0.12 to retain low confidence large boxes
        )

        # 5. Map raw predictions
        temp_detections = []
        idx_list = indices.flatten() if hasattr(indices, "flatten") else indices
        
        if len(idx_list) > 0:
            for idx in idx_list:
                cid = class_ids[idx]
                px, py, pw, ph = boxes[idx]

                # Check if it is a person but actually a pedestal fan.
                # Pedestal fans are often misclassified as "person" due to their stand and round head.
                # If there's an overlapping "clock" box (fan head) in the upper 60% of the "person" box, reclassify it.
                if cid == 0:  # 0 is person in COCO
                    has_clock_overlap = False
                    for cx_box, cy_box, cw_box, ch_box in raw_clocks:
                        horiz_overlap = (cx_box + cw_box/2 >= px) and (cx_box + cw_box/2 <= px + pw)
                        vert_overlap = (cy_box + ch_box/2 >= py) and (cy_box + ch_box/2 <= py + ph * 0.6)
                        if horiz_overlap and vert_overlap:
                            has_clock_overlap = True
                            break
                    if has_clock_overlap:
                        cid = 74  # 74 is clock in COCO (maps to fan/clock)

                if cid == -1:
                    cname = "large obstacle"
                elif cid == -2:
                    cname = "pole/pillar"
                elif cid == -3:
                    cname = "step down / stairs"
                else:
                    if mode == 'desktop':
                        cname = config.DESKTOP_CLASSES.get(cid, self._all_classes[cid])
                    else:
                        cname = config.OBSTACLE_CLASSES.get(cid, self._all_classes[cid])

                temp_detections.append({
                    "class_name": cname,
                    "class_id": cid,
                    "confidence": confidences[idx],
                    "box": boxes[idx],
                    "is_reclassified_fan": cid == 74 and class_ids[idx] == 0
                })

        # Filter out duplicate clock detections that overlap with a larger reclassified pedestal fan
        raw_detections = []
        reclassified_fans = [d for d in temp_detections if d["is_reclassified_fan"]]

        for d in temp_detections:
            if d["class_id"] == 74 and not d["is_reclassified_fan"]:
                cx, cy, cw, ch = d["box"]
                is_duplicate = False
                for fan in reclassified_fans:
                    fx, fy, fw, fh = fan["box"]
                    if fw > cw * 1.2 and fh > ch * 1.5:
                        horiz_inside = cx >= fx - 15 and (cx + cw) <= (fx + fw + 15)
                        vert_inside = cy >= fy - 15 and (cy + ch) <= (fy + fh + 15)
                        if horiz_inside and vert_inside:
                            is_duplicate = True
                            break
                if is_duplicate:
                    continue

            raw_detections.append({
                "class_name": d["class_name"],
                "class_id": d["class_id"],
                "confidence": d["confidence"],
                "bbox": tuple(d["box"])
            })

        # Check for close-up flat wall (only in navigation mode)
        if mode == 'navigation' and self._detect_wall(frame):
            raw_detections.append({
                "class_name": "wall",
                "class_id": -1,
                "confidence": 0.98,
                "bbox": (0, 0, width, height)
            })

        # 6. Centroid tracking & stabilization for anti-flicker / velocity checks
        for r_det in raw_detections:
            bbox = r_det["bbox"]
            cX = bbox[0] + bbox[2] // 2
            cY = bbox[1] + bbox[3] // 2

            best_match = None
            min_distance = 80 # pixels association threshold

            for t_obj in self._tracked_objects:
                dx = t_obj["center_x"] - cX
                dy = t_obj["center_y"] - cY
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < min_distance:
                    min_distance = dist
                    best_match = t_obj

            if best_match:
                dx = cX - best_match["center_x"]
                dy = cY - best_match["center_y"]
                velocity = np.sqrt(dx * dx + dy * dy)

                best_match["bbox"] = bbox
                best_match["center_x"] = cX
                best_match["center_y"] = cY
                best_match["class_name"] = r_det["class_name"]
                best_match["class_id"] = r_det["class_id"]
                best_match["confidence"] = r_det["confidence"]
                best_match["frames_detected"] += 1
                best_match["frames_lost"] = 0
                best_match["velocity"] = velocity
                best_match["is_moving"] = velocity > config.MOTION_VELOCITY_THRESHOLD
                best_match["matched"] = True
            else:
                self._tracked_objects.append({
                    "id": self._next_track_id,
                    "class_name": r_det["class_name"],
                    "class_id": r_det["class_id"],
                    "confidence": r_det["confidence"],
                    "bbox": bbox,
                    "center_x": cX,
                    "center_y": cY,
                    "frames_detected": 1,
                    "frames_lost": 0,
                    "velocity": 0.0,
                    "is_moving": False,
                    "is_confirmed": False,
                    "matched": True
                })
                self._next_track_id += 1

        # Process lost frames and confirm active ones
        for t_obj in self._tracked_objects:
            if not t_obj["matched"]:
                t_obj["frames_lost"] += 1
            else:
                t_obj["matched"] = False

            if t_obj["frames_detected"] >= config.DETECTION_CONFIRM_FRAMES:
                t_obj["is_confirmed"] = True

        # Keep tracks within hold frames grace period
        self._tracked_objects = [t for t in self._tracked_objects if t["frames_lost"] < config.DETECTION_HOLD_FRAMES]

        # 7. Apply distance gating, zone classification and warning cooldowns
        detections = []
        confirmed_tracks = [t for t in self._tracked_objects if t["is_confirmed"]]

        for track in confirmed_tracks:
            bbox = track["bbox"]
            h = bbox[3]
            cname = track["class_name"]

            if cname == "wall":
                estimated_cm = 60.0
            else:
                estimated_cm = self._estimate_distance(cname, h, height)

            # Max Distance Gate
            max_distance_cm = 150.0 if mode == 'desktop' else config.MAX_DISTANCE_CM
            if estimated_cm > max_distance_cm:
                continue

            # Classify dynamic zone
            if mode == 'desktop':
                very_close_cm = 30.0
                near_cm = 60.0
                very_close_ratio = 0.70
                near_ratio = 0.40
            else:
                very_close_cm = config.DIST_VERY_CLOSE_CM
                near_cm = config.DIST_NEAR_CM
                very_close_ratio = config.BBOX_VERY_CLOSE_RATIO
                near_ratio = config.BBOX_NEAR_RATIO

            ratio = h / height
            if estimated_cm < very_close_cm or ratio > very_close_ratio:
                zone = config.ZONE_VERY_CLOSE
            elif estimated_cm < near_cm or ratio > near_ratio:
                zone = config.ZONE_NEAR
            else:
                zone = config.ZONE_SAFE

            direction = self._classifyDirection(track["center_x"], width)
            
            # Calculate dynamic danger
            max_dist_danger = 150.0 if mode == 'desktop' else 500.0
            dist_factor = max(0.0, 1.0 - estimated_cm / max_dist_danger)
            center_offset = abs(track["center_x"] - width / 2.0) / (width / 2.0)
            center_factor = 1.0 - center_offset * 0.5
            known_h = config.KNOWN_HEIGHTS_CM.get(cname, 50.0)
            size_factor = min(1.0, known_h / 200.0)
            danger_score = min(1.0, max(0.0, dist_factor * 0.6 + center_factor * 0.3 + size_factor * 0.1))

            # Warning Cooldown timer checking
            now = time.time()
            cooldown_key = f"{cname}_{zone}"
            last_alert = self._alert_cooldowns.get(cooldown_key, 0.0)
            cooldown_active = False
            if now - last_alert < config.WARNING_COOLDOWN_S:
                cooldown_active = True
            else:
                self._alert_cooldowns[cooldown_key] = now

            if mode == 'desktop':
                clean_name = config.DESKTOP_CLASS_MAP.get(cname, cname)
            else:
                clean_name = config.NAVIGATION_CLASS_MAP.get(cname, cname)

            det = Detection(
                class_name=clean_name,
                class_id=track["class_id"],
                confidence=track["confidence"],
                bbox=bbox,
                estimated_cm=estimated_cm,
                zone=zone,
                direction=direction,
                danger_score=danger_score,
                is_moving=track["is_moving"],
                cooldown_active=cooldown_active
            )
            detections.append(det)

        # Sort detections by danger rating
        detections.sort(key=lambda d: d.danger_score, reverse=True)
        return detections

    def _detect_wall(self, frame):
        try:
            if frame is None or frame.size == 0:
                return False

            # Resize frame to a small 64x64 thumbnail
            small = cv2.resize(frame, (64, 64))

            # Convert to grayscale
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            # Calculate average luminance (brightness)
            avg_lum = np.mean(gray)
            if avg_lum < 22:
                return False

            # Calculate absolute gradients in X and Y directions
            grad_x = np.abs(gray[:, :-1].astype(np.int16) - gray[:, 1:].astype(np.int16))
            grad_y = np.abs(gray[:-1, :].astype(np.int16) - gray[1:, :].astype(np.int16))

            avg_grad = (np.mean(grad_x) + np.mean(grad_y)) / 2.0

            # If the scene is extremely uniform (low texture variation), it's a flat wall close up
            return avg_grad < 2.0
        except Exception as e:
            print(f"[NeuroLens] Python wall detection error: {e}")
            return False

    @property
    def frame_count(self):
        return self._frame_count
