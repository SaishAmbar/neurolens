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
CONFIDENCE_THRESHOLD = 0.35
NMS_THRESHOLD = 0.4
INPUT_SIZE = (416, 416)

# COCO class indices that are relevant obstacles
# Includes all 80 COCO classes for comprehensive detection
OBSTACLE_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorbike",
    4: "aeroplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "sofa",
    58: "pottedplant",
    59: "bed",
    60: "diningtable",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

# ─── Distance Estimation ─────────────────────────────────────────────────────
# Known real-world heights in centimeters (approximate)
KNOWN_HEIGHTS_CM = {
    "person": 90,       # Upper-body/torso height (webcam rarely sees full body)
    "car": 150,
    "bicycle": 100,
    "motorbike": 110,
    "aeroplane": 400,
    "bus": 300,
    "train": 350,
    "truck": 350,
    "boat": 150,
    "traffic light": 100,
    "fire hydrant": 50,
    "stop sign": 75,
    "parking meter": 120,
    "bench": 85,
    "bird": 20,
    "cat": 30,
    "dog": 60,
    "horse": 160,
    "sheep": 70,
    "cow": 140,
    "elephant": 300,
    "bear": 150,
    "zebra": 140,
    "giraffe": 500,
    "backpack": 50,
    "umbrella": 100,
    "handbag": 35,
    "tie": 50,
    "suitcase": 60,
    "frisbee": 3,
    "skis": 170,
    "snowboard": 150,
    "sports ball": 22,
    "kite": 80,
    "baseball bat": 90,
    "baseball glove": 25,
    "skateboard": 15,
    "surfboard": 200,
    "tennis racket": 70,
    "bottle": 25,
    "wine glass": 20,
    "cup": 12,
    "fork": 20,
    "knife": 25,
    "spoon": 18,
    "bowl": 10,
    "banana": 20,
    "apple": 8,
    "sandwich": 8,
    "orange": 8,
    "broccoli": 15,
    "carrot": 20,
    "hot dog": 5,
    "pizza": 5,
    "donut": 8,
    "cake": 15,
    "chair": 90,
    "sofa": 85,
    "pottedplant": 40,
    "bed": 60,
    "diningtable": 75,
    "toilet": 40,
    "tv": 50,
    "laptop": 25,
    "mouse": 4,
    "remote": 20,
    "keyboard": 5,
    "cell phone": 15,
    "microwave": 30,
    "oven": 60,
    "toaster": 20,
    "sink": 25,
    "refrigerator": 170,
    "book": 25,
    "clock": 30,
    "vase": 30,
    "scissors": 20,
    "teddy bear": 40,
    "hair drier": 25,
    "toothbrush": 18,
}

# Estimated focal length in pixels for a typical 720p webcam with ~78° FOV
# focal_length ≈ (image_width / 2) / tan(FOV/2) ≈ (1280/2) / tan(39°) ≈ 790
# Tuned down further to account for partial object visibility at close range
FOCAL_LENGTH_PX = 550

# Distance zone thresholds (in centimeters)
DIST_VERY_CLOSE_CM = 150   # < 1.5m
DIST_NEAR_CM = 300          # 1.5m - 3m
# > 3m = SAFE

# Fallback: bbox height as fraction of frame height
BBOX_VERY_CLOSE_RATIO = 0.50   # bbox covers > 50% of frame height
BBOX_NEAR_RATIO = 0.25         # bbox covers 25-50%
# < 25% = SAFE

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
