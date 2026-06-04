"""
NeuroLens Configuration
All tunable parameters, color palettes, thresholds, and YOLO model settings.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

YOLO_CFG_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
YOLO_WEIGHTS_URL = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"
COCO_NAMES_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"

YOLO_CFG_PATH = os.path.join(MODELS_DIR, "yolov4-tiny.cfg")
YOLO_WEIGHTS_PATH = os.path.join(MODELS_DIR, "yolov4-tiny.weights")
COCO_NAMES_PATH = os.path.join(MODELS_DIR, "coco.names")

# ─── Detection ────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.25   # Lowered to 0.25 — improves detection of fans and partial obstacles
NMS_THRESHOLD = 0.4
INPUT_SIZE = (416, 416)

# COCO class indices that are relevant obstacles (Navigation-only Whitelist)
OBSTACLE_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorbike",
    5: "bus",
    7: "truck",
    10: "fire hydrant",
    11: "stop sign",
    13: "bench",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
    56: "chair",
    57: "sofa",
    58: "pottedplant",
    59: "bed",
    60: "diningtable",
    72: "refrigerator",
    74: "clock",
}

# COCO class indices that are relevant in Desktop mode (small/office items)
DESKTOP_CLASSES = {
    39: "bottle",
    40: "wine glass",
    41: "cup",
    43: "knife",
    45: "bowl",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

# Mapping of raw classes to navigation-friendly terms
NAVIGATION_CLASS_MAP = {
    "person": "person",
    "wall": "wall",
    "chair": "chair",
    "sofa": "couch",
    "diningtable": "table",
    "bench": "bench",
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "motorbike": "scooter/bike",
    "bicycle": "scooter/bike",
    "backpack": "bag on floor",
    "handbag": "bag on floor",
    "suitcase": "bag on floor",
    "refrigerator": "cupboard/closet",
    "bed": "large obstacle",
    "clock": "fan/clock",
    "umbrella": "umbrella",
    "pottedplant": "potted plant",
    "stop sign": "stop sign",
    "fire hydrant": "fire hydrant",
    "large obstacle": "large obstacle",
    "pole/pillar": "pole/pillar",
    "step down / stairs": "step down / stairs"
}

# Mapping of raw classes to desktop-friendly terms
DESKTOP_CLASS_MAP = {
    "toothbrush": "pen/toothbrush",
    "scissors": "scissors/spectacles",
    "knife": "knife",
    "wine glass": "glass/cup",
    "cup": "cup/headset",
    "teddy bear": "headset/toy",
    "mouse": "mouse",
    "keyboard": "keyboard",
    "cell phone": "cell phone",
    "remote": "remote",
    "book": "book",
    "bottle": "bottle",
    "bowl": "bowl/spectacles",
    "laptop": "laptop",
    "vase": "vase",
    "clock": "clock",
}

# ─── Tracking & Cooldown Tuning ───────────────────────────────────────────────
DETECTION_HOLD_FRAMES = 15      # grace period for anti-flicker (~500ms at 30fps)
DETECTION_CONFIRM_FRAMES = 3   # consecutive detections required to confirm
WARNING_COOLDOWN_S = 3.0       # repeat alert cooldown in seconds
MOTION_VELOCITY_THRESHOLD = 12 # pixels per frame to trigger moving state
MAX_DISTANCE_CM = 300          # ignore obstacles > 3 meters away

# ─── Distance Estimation ─────────────────────────────────────────────────────
# Known real-world heights in centimeters (approximate)
# These are the VISIBLE heights YOLO tends to detect — slightly conservative
KNOWN_HEIGHTS_CM = {
    "person": 170,          # Full standing adult
    "wall": 200,
    "refrigerator": 180,
    "cupboard/closet": 180,
    "chair": 90,            # Full chair with backrest (~90cm)
    "sofa": 85,             # Sofa visible height
    "couch": 85,
    "bench": 50,
    "diningtable": 75,
    "car": 150,
    "bus": 300,
    "truck": 250,
    "motorbike": 110,
    "bicycle": 95,
    "backpack": 50,
    "handbag": 35,
    "suitcase": 65,
    "clock": 60,
    "fan/clock": 60,
    "umbrella": 80,
    "pottedplant": 60,
    "potted plant": 60,
    "stop sign": 150,
    "fire hydrant": 90,
    "large obstacle": 180,
    "pole/pillar": 210,
    "step down / stairs": 20,
    # Small desktop items
    "pen/toothbrush": 18,
    "scissors/spectacles": 18,
    "glass/cup": 10,
    "cup/headset": 10,
    "headset/toy": 18,
    "mouse": 5,
    "keyboard": 5,
    "cell phone": 15,
    "remote": 15,
    "book": 22,
    "bottle": 25,
    "bowl/spectacles": 8,
    "laptop": 25,
    "vase": 30,
    "clock": 15,
}

# Estimated focal length in pixels for a typical 720p webcam with ~78° FOV
FOCAL_LENGTH_PX = 580   # Slightly raised — compensates for real-world lens variation

# Distance zone thresholds (in centimeters)
DIST_VERY_CLOSE_CM = 100   # < 1.0m  → WARNING
DIST_NEAR_CM = 200          # 1.0m – 2.0m → OBJECT AHEAD
# > 2.0m = SAFE

# Fallback: bbox height as fraction of frame height
# Raised VERY_CLOSE ratio so a close person doesn't auto-escalate to VERY CLOSE
# based purely on frame coverage (e.g. face filling frame)
BBOX_VERY_CLOSE_RATIO = 0.60   # bbox covers > 60% of frame height
BBOX_NEAR_RATIO = 0.30         # bbox covers 30-60%
# < 30% = SAFE
# < 30% = SAFE

# ─── Colors (BGR for OpenCV) ─────────────────────────────────────────────────
COLOR_SAFE = (128, 255, 0)       # Neon green
COLOR_NEAR = (0, 200, 255)       # Amber/yellow
COLOR_VERY_CLOSE = (0, 64, 255)  # Neon red

COLOR_HUD_BG = (15, 15, 15)     # Dark background
COLOR_HUD_TEXT = (200, 220, 230) # Light text
COLOR_HUD_ACCENT = (255, 180, 0) # Cyan accent (BGR)
COLOR_HUD_GRID = (40, 50, 50)   # Subtle grid
COLOR_SCANLINE = (200, 255, 0)   # Scan line green

COLOR_DANGER_FLASH = (0, 0, 200) # Red flash overlay

# ─── HUD Layout ──────────────────────────────────────────────────────────────
HUD_FONT_SCALE_TITLE = 0.7
HUD_FONT_SCALE_LABEL = 0.5
HUD_FONT_SCALE_SMALL = 0.4
HUD_CORNER_LENGTH = 30
HUD_CORNER_THICKNESS = 2
HUD_PANEL_ALPHA = 0.7

# ─── Camera ──────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
TARGET_FPS = 30

# ─── Zone Names ──────────────────────────────────────────────────────────────
ZONE_SAFE = "SAFE"
ZONE_NEAR = "NEAR"
ZONE_VERY_CLOSE = "VERY CLOSE"

# Zone display info
ZONE_INFO = {
    ZONE_SAFE: {
        "color": COLOR_SAFE,
        "icon": "✓",
        "label": "PATH CLEAR",
    },
    ZONE_NEAR: {
        "color": COLOR_NEAR,
        "icon": "⚠",
        "label": "OBSTACLE AHEAD",
    },
    ZONE_VERY_CLOSE: {
        "color": COLOR_VERY_CLOSE,
        "icon": "🚨",
        "label": "IMMEDIATE DANGER",
    },
}
