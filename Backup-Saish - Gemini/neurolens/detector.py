"""
NeuroLens — YOLO Object Detector
Handles YOLOv4-tiny model loading, auto-downloading, and detection pipeline.
"""

import os
import urllib.request
import sys
import cv2
import numpy as np

from . import config


class Detection:
    """Represents a single detected object."""

    __slots__ = ("class_name", "class_id", "confidence", "bbox", "center_x", "center_y")

    def __init__(self, class_name, class_id, confidence, bbox):
        self.class_name = class_name
        self.class_id = class_id
        self.confidence = confidence
        self.bbox = bbox  # (x, y, w, h)
        self.center_x = bbox[0] + bbox[2] // 2
        self.center_y = bbox[1] + bbox[3] // 2

    def __repr__(self):
        return (
            f"Detection({self.class_name}, conf={self.confidence:.2f}, "
            f"bbox={self.bbox}, center=({self.center_x},{self.center_y}))"
        )


class Detector:
    """YOLOv4-tiny based object detector using OpenCV DNN."""

    def __init__(self):
        self._ensure_model_files()
        self._load_class_names()
        self._load_network()
        self._frame_count = 0

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

    # ── Detection Pipeline ────────────────────────────────────────────────

    def detect(self, frame):
        """
        Run object detection on a frame.

        Args:
            frame: BGR image (numpy array)

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

        # 3. Process detections
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                # Filter by confidence and obstacle relevance
                if confidence < config.CONFIDENCE_THRESHOLD:
                    continue
                if class_id not in config.OBSTACLE_CLASSES:
                    continue

                # Convert YOLO center-format to top-left format
                cx, cy, w, h = detection[0:4]
                x = int(cx * width - (w * width) / 2)
                y = int(cy * height - (h * height) / 2)
                w = int(w * width)
                h = int(h * height)

                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

        # 4. Apply NMS
        if len(boxes) == 0:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, config.CONFIDENCE_THRESHOLD, config.NMS_THRESHOLD
        )

        # 5. Build Detection objects
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                class_name = config.OBSTACLE_CLASSES.get(
                    class_ids[i], self._all_classes[class_ids[i]]
                )
                det = Detection(
                    class_name=class_name,
                    class_id=class_ids[i],
                    confidence=confidences[i],
                    bbox=tuple(boxes[i]),
                )
                detections.append(det)

        return detections

    @property
    def frame_count(self):
        return self._frame_count
