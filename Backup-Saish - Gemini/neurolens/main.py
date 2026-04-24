"""
NeuroLens — Main Entry Point
Orchestrates camera capture, object detection, distance estimation,
and HUD overlay rendering in a real-time loop.
"""

import sys
import os
import time
import cv2

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from . import config
from .detector import Detector
from .distance_estimator import DistanceEstimator
from .hud_overlay import HUDOverlay


def print_banner():
    """Print startup banner."""
    print()
    print("=" * 60)
    print("  _   _ _____ _   _ ____   ___  _     _____ _   _ ____")
    print(" | \\ | | ____| | | |  _ \\ / _ \\| |   | ____| \\ | / ___|")
    print(" |  \\| |  _| | | | | |_) | | | | |   |  _| |  \\| \\___ \\")
    print(" | |\\  | |___| |_| |  _ <| |_| | |___| |___| |\\  |___) |")
    print(" |_| \\_|_____|\\___/|_| \\_\\\\___/|_____|_____|_| \\_|____/")
    print("  Real-Time Obstacle Detection System v1.0")
    print("=" * 60)
    print()


def main():
    """Main entry point."""
    print_banner()

    # ── Initialize Modules ────────────────────────────────────────────────
    print("[NeuroLens] Initializing modules...")

    detector = Detector()
    distance_estimator = DistanceEstimator()
    hud = HUDOverlay()

    print("[NeuroLens] ✓ All modules initialized")

    # ── Initialize Camera ─────────────────────────────────────────────────
    print(f"[NeuroLens] Opening camera (index {config.CAMERA_INDEX})...")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        print("[NeuroLens] ERROR: Cannot open camera!")
        print("           Try changing CAMERA_INDEX in config.py")
        sys.exit(1)

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[NeuroLens] ✓ Camera opened at {actual_w}x{actual_h}")

    # Update distance estimator with actual resolution
    distance_estimator.update_frame_size(actual_w, actual_h)

    # ── Main Loop ─────────────────────────────────────────────────────────
    print("[NeuroLens] ✓ Starting detection loop (press 'q' to quit)")
    print()

    fps = 0.0
    frame_count = 0
    fps_start_time = time.time()
    fps_update_interval = 0.5  # update FPS display every 0.5s
    fps_frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[NeuroLens] WARNING: Failed to read frame, retrying...")
                continue

            frame_count += 1
            fps_frame_count += 1

            # ── Detection ─────────────────────────────────────────────────
            detections = detector.detect(frame)

            # ── Distance Estimation ───────────────────────────────────────
            distance_results = distance_estimator.estimate_all(detections)

            # ── HUD Overlay ───────────────────────────────────────────────
            frame = hud.render(frame, distance_results, fps)

            # ── FPS Calculation ───────────────────────────────────────────
            elapsed = time.time() - fps_start_time
            if elapsed >= fps_update_interval:
                fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = time.time()

            # ── Display ───────────────────────────────────────────────────
            cv2.imshow("NeuroLens — Obstacle Detection", frame)

            # ── Input Handling ────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or ESC
                print("\n[NeuroLens] Shutting down...")
                break

    except KeyboardInterrupt:
        print("\n[NeuroLens] Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[NeuroLens] Processed {frame_count} frames total")
        print("[NeuroLens] ✓ Shutdown complete")


if __name__ == "__main__":
    main()
