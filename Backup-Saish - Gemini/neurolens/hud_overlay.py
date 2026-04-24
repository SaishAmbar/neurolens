"""
NeuroLens — HUD Overlay Renderer
Renders a futuristic heads-up display with animated scan lines, tech corners,
color-coded bounding boxes, warning panels, and direction indicators.
"""

import time
import math
import cv2
import numpy as np

from . import config


class HUDOverlay:
    """Renders the futuristic HUD overlay on camera frames."""

    def __init__(self):
        self._start_time = time.time()
        self._frame_count = 0
        self._fps = 0.0
        self._fps_update_time = time.time()
        self._fps_frame_count = 0

    # ══════════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════════════

    def render(self, frame, distance_results, fps=0.0):
        """
        Render the complete HUD overlay onto the frame.

        Args:
            frame: BGR image (will be modified in-place)
            distance_results: List of DistanceResult objects (sorted by danger)
            fps: Current frames per second

        Returns:
            The modified frame
        """
        self._frame_count += 1
        self._fps = fps
        h, w = frame.shape[:2]
        t = time.time() - self._start_time  # elapsed time for animations

        # Determine alert level from most dangerous detection
        most_dangerous = distance_results[0] if distance_results else None
        alert_zone = most_dangerous.zone if most_dangerous else config.ZONE_SAFE

        # ── Layer 1: Background effects ───────────────────────────────────
        self._draw_grid(frame, w, h)
        self._draw_scan_line(frame, w, h, t)

        # ── Layer 2: Detection overlays ───────────────────────────────────
        for i, result in enumerate(distance_results):
            is_primary = (i == 0)
            self._draw_detection_box(frame, result, is_primary, t)

        # ── Layer 3: Danger flash (if VERY CLOSE) ────────────────────────
        if alert_zone == config.ZONE_VERY_CLOSE:
            self._draw_danger_flash(frame, w, h, t)

        # ── Layer 4: HUD frame elements ──────────────────────────────────
        self._draw_corner_brackets(frame, w, h, alert_zone)
        self._draw_header(frame, w, t)
        self._draw_warning_panel(frame, w, most_dangerous, t)
        self._draw_direction_indicators(frame, w, h, most_dangerous, t)
        self._draw_stats_footer(frame, w, h, len(distance_results))
        self._draw_distance_legend(frame, w, h)

        return frame

    # ══════════════════════════════════════════════════════════════════════
    #  BACKGROUND EFFECTS
    # ══════════════════════════════════════════════════════════════════════

    def _draw_grid(self, frame, w, h):
        """Draw subtle grid overlay for depth perception."""
        overlay = frame.copy()
        spacing = 40
        for x in range(0, w, spacing):
            cv2.line(overlay, (x, 0), (x, h), config.COLOR_HUD_GRID, 1)
        for y in range(0, h, spacing):
            cv2.line(overlay, (0, y), (w, y), config.COLOR_HUD_GRID, 1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    def _draw_scan_line(self, frame, w, h, t):
        """Animated horizontal scan line sweeping vertically."""
        # Scan line position cycles over 3 seconds
        cycle = (t % 3.0) / 3.0
        y = int(cycle * h)
        alpha = 0.4

        overlay = frame.copy()
        # Main scan line
        cv2.line(overlay, (0, y), (w, y), config.COLOR_SCANLINE, 2)
        # Gradient trail (5 lines above with decreasing opacity)
        for i in range(1, 8):
            trail_y = y - i * 3
            if trail_y >= 0:
                trail_alpha = alpha * (1.0 - i / 8.0)
                cv2.line(overlay, (0, trail_y), (w, trail_y), config.COLOR_SCANLINE, 1)

        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # ══════════════════════════════════════════════════════════════════════
    #  DETECTION BOXES
    # ══════════════════════════════════════════════════════════════════════

    def _draw_detection_box(self, frame, result, is_primary, t):
        """Draw a color-coded bounding box with tech-style corners."""
        det = result.detection
        x, y, w, h = det.bbox
        color = result.zone_color

        # Clamp coordinates
        x = max(0, x)
        y = max(0, y)

        thickness = 3 if is_primary else 2

        # ── Glow effect for primary threat ────────────────────────────────
        if is_primary:
            # Pulsing glow
            pulse = 0.5 + 0.5 * math.sin(t * 4)
            glow_color = tuple(int(c * pulse) for c in color)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x - 3, y - 3), (x + w + 3, y + h + 3), glow_color, 4)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # ── Corner brackets instead of full rectangle ─────────────────────
        corner_len = min(25, w // 4, h // 4)

        # Top-left
        cv2.line(frame, (x, y), (x + corner_len, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + corner_len), color, thickness)
        # Top-right
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, thickness)
        # Bottom-left
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, thickness)
        # Bottom-right
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)

        # ── Dashed connecting lines ───────────────────────────────────────
        self._draw_dashed_line(frame, (x + corner_len, y), (x + w - corner_len, y), color, 1)
        self._draw_dashed_line(frame, (x + corner_len, y + h), (x + w - corner_len, y + h), color, 1)
        self._draw_dashed_line(frame, (x, y + corner_len), (x, y + h - corner_len), color, 1)
        self._draw_dashed_line(frame, (x + w, y + corner_len), (x + w, y + h - corner_len), color, 1)

        # ── Label tag ─────────────────────────────────────────────────────
        dist_m = result.estimated_cm / 100.0
        label = f"{det.class_name.upper()} | {result.zone} | {dist_m:.1f}m"
        conf_label = f"{det.confidence * 100:.0f}%"

        # Label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_x = x
        label_y = max(y - 10, label_h + 10)

        # Semi-transparent label background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (label_x - 2, label_y - label_h - 6),
            (label_x + label_w + 6, label_y + 4),
            config.COLOR_HUD_BG,
            -1,
        )
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Label border
        cv2.rectangle(
            frame,
            (label_x - 2, label_y - label_h - 6),
            (label_x + label_w + 6, label_y + 4),
            color,
            1,
        )

        # Label text
        cv2.putText(frame, label, (label_x + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Confidence on right side
        (conf_w, conf_h), _ = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(frame, conf_label, (x + w - conf_w, y + h + conf_h + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # ── Center crosshair on primary ───────────────────────────────────
        if is_primary:
            cx, cy = det.center_x, det.center_y
            cross_size = 10
            cv2.line(frame, (cx - cross_size, cy), (cx + cross_size, cy), color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy - cross_size), (cx, cy + cross_size), color, 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), cross_size + 2, color, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_dashed_line(frame, pt1, pt2, color, thickness, dash_length=8):
        """Draw a dashed line between two points."""
        x1, y1 = pt1
        x2, y2 = pt2
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if dist == 0:
            return

        num_dashes = int(dist / dash_length)
        for i in range(0, num_dashes, 2):
            start_ratio = i / num_dashes
            end_ratio = min((i + 1) / num_dashes, 1.0)
            sx = int(x1 + (x2 - x1) * start_ratio)
            sy = int(y1 + (y2 - y1) * start_ratio)
            ex = int(x1 + (x2 - x1) * end_ratio)
            ey = int(y1 + (y2 - y1) * end_ratio)
            cv2.line(frame, (sx, sy), (ex, ey), color, thickness)

    # ══════════════════════════════════════════════════════════════════════
    #  DANGER FLASH
    # ══════════════════════════════════════════════════════════════════════

    def _draw_danger_flash(self, frame, w, h, t):
        """Pulsing red border and vignette when an obstacle is VERY CLOSE."""
        pulse = 0.3 + 0.3 * math.sin(t * 6)  # Fast pulsing

        # Red border
        overlay = frame.copy()
        border_thickness = 8
        cv2.rectangle(overlay, (0, 0), (w, h), config.COLOR_DANGER_FLASH, border_thickness)
        cv2.addWeighted(overlay, pulse, frame, 1 - pulse, 0, frame)

        # Red vignette corners
        vignette = np.zeros_like(frame, dtype=np.uint8)
        cv2.rectangle(vignette, (0, 0), (w, h), config.COLOR_DANGER_FLASH, -1)

        # Create radial gradient mask (dark at center, bright at edges)
        mask = np.zeros((h, w), dtype=np.float32)
        cy, cx = h // 2, w // 2
        max_dist = math.sqrt(cx ** 2 + cy ** 2)
        Y, X = np.ogrid[:h, :w]
        dist_map = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
        mask = np.clip(dist_map / max_dist - 0.5, 0, 0.5) * 2  # 0 at center, 1 at corners

        mask3 = np.stack([mask] * 3, axis=-1)
        vignette_weighted = (vignette * mask3 * pulse * 0.3).astype(np.uint8)
        cv2.add(frame, vignette_weighted, frame)

    # ══════════════════════════════════════════════════════════════════════
    #  HUD FRAME ELEMENTS
    # ══════════════════════════════════════════════════════════════════════

    def _draw_corner_brackets(self, frame, w, h, alert_zone):
        """Draw tech-style corner brackets around the frame border."""
        color = config.ZONE_INFO[alert_zone]["color"]
        length = config.HUD_CORNER_LENGTH
        thick = config.HUD_CORNER_THICKNESS
        margin = 8

        corners = [
            # Top-left
            ((margin, margin), (margin + length, margin), (margin, margin + length)),
            # Top-right
            ((w - margin, margin), (w - margin - length, margin), (w - margin, margin + length)),
            # Bottom-left
            ((margin, h - margin), (margin + length, h - margin), (margin, h - margin - length)),
            # Bottom-right
            ((w - margin, h - margin), (w - margin - length, h - margin), (w - margin, h - margin - length)),
        ]

        for corner, h_end, v_end in corners:
            cv2.line(frame, corner, h_end, color, thick, cv2.LINE_AA)
            cv2.line(frame, corner, v_end, color, thick, cv2.LINE_AA)

    def _draw_header(self, frame, w, t):
        """Draw NEUROLENS branding and timestamp at top."""
        # ── Top-left: Branding ────────────────────────────────────────────
        # Blinking dot
        dot_visible = int(t * 2) % 2 == 0
        if dot_visible:
            cv2.circle(frame, (25, 30), 4, config.COLOR_SAFE, -1, cv2.LINE_AA)

        cv2.putText(frame, "NEUROLENS", (38, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, config.HUD_FONT_SCALE_TITLE,
                    config.COLOR_HUD_TEXT, 1, cv2.LINE_AA)

        cv2.putText(frame, "v1.0", (145, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, config.HUD_FONT_SCALE_SMALL,
                    config.COLOR_HUD_ACCENT, 1, cv2.LINE_AA)

        # ── Top-right: FPS + Timestamp ────────────────────────────────────
        fps_text = f"FPS: {self._fps:.0f}"
        timestamp = time.strftime("%H:%M:%S")

        (fps_w, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, fps_text, (w - fps_w - 20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    config.COLOR_SAFE if self._fps >= 20 else config.COLOR_NEAR, 1, cv2.LINE_AA)

        (ts_w, _), _ = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(frame, timestamp, (w - ts_w - 20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    config.COLOR_HUD_TEXT, 1, cv2.LINE_AA)

    def _draw_warning_panel(self, frame, w, most_dangerous, t):
        """Draw the central warning panel at top-center."""
        if most_dangerous is None:
            zone = config.ZONE_SAFE
            obj_name = ""
            direction = ""
        else:
            zone = most_dangerous.zone
            obj_name = most_dangerous.detection.class_name.upper()
            direction = most_dangerous.direction

        zone_info = config.ZONE_INFO[zone]
        color = zone_info["color"]
        label = zone_info["label"]
        icon = zone_info["icon"]

        # Panel dimensions
        panel_w = 340
        panel_h = 55
        px = (w - panel_w) // 2
        py = 12

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), config.COLOR_HUD_BG, -1)
        cv2.addWeighted(overlay, config.HUD_PANEL_ALPHA, frame, 1 - config.HUD_PANEL_ALPHA, 0, frame)

        # Border (pulsing for danger)
        if zone == config.ZONE_VERY_CLOSE:
            pulse = 0.5 + 0.5 * math.sin(t * 6)
            border_color = tuple(int(c * pulse) for c in color)
            cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), border_color, 2)
        else:
            cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), color, 1)

        # Alert text
        alert_text = f"{icon} {label}"
        (text_w, text_h), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_x = px + (panel_w - text_w) // 2
        cv2.putText(frame, alert_text, (text_x, py + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # Object info line
        if obj_name:
            info_text = f"{obj_name} - {direction}"
            (info_w, _), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            info_x = px + (panel_w - info_w) // 2
            cv2.putText(frame, info_text, (info_x, py + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, config.COLOR_HUD_TEXT, 1, cv2.LINE_AA)

    def _draw_direction_indicators(self, frame, w, h, most_dangerous, t):
        """Draw LEFT | CENTER | RIGHT direction indicators at the bottom."""
        bar_h = 35
        bar_y = h - bar_h - 15
        section_w = w // 3

        directions = ["LEFT", "CENTER", "RIGHT"]
        arrows = ["<< LEFT", "[ CENTER ]", "RIGHT >>"]
        active_dir = most_dangerous.direction if most_dangerous else None

        overlay = frame.copy()

        for i, (dir_name, arrow) in enumerate(zip(directions, arrows)):
            sx = i * section_w
            is_active = (dir_name == active_dir)

            # Background
            if is_active and most_dangerous:
                bg_color = most_dangerous.zone_color
                cv2.rectangle(overlay, (sx + 2, bar_y), (sx + section_w - 2, bar_y + bar_h), bg_color, -1)
            else:
                cv2.rectangle(overlay, (sx + 2, bar_y), (sx + section_w - 2, bar_y + bar_h),
                              config.COLOR_HUD_BG, -1)

        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Draw text and separators
        for i, (dir_name, arrow) in enumerate(zip(directions, arrows)):
            sx = i * section_w
            is_active = (dir_name == active_dir)

            text_color = (255, 255, 255) if is_active else config.COLOR_HUD_TEXT

            # Animated arrow for active direction
            if is_active and most_dangerous:
                offset = int(3 * math.sin(t * 5))
                if dir_name == "LEFT":
                    arrow = "<" * (2 + int(t * 2) % 2) + " LEFT"
                elif dir_name == "RIGHT":
                    arrow = "RIGHT " + ">" * (2 + int(t * 2) % 2)
            else:
                offset = 0

            (tw, th), _ = cv2.getTextSize(arrow, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tx = sx + (section_w - tw) // 2 + offset
            ty = bar_y + (bar_h + th) // 2

            cv2.putText(frame, arrow, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color,
                        2 if is_active else 1, cv2.LINE_AA)

            # Separator lines
            if i > 0:
                cv2.line(frame, (sx, bar_y + 5), (sx, bar_y + bar_h - 5),
                         config.COLOR_HUD_GRID, 1)

    def _draw_stats_footer(self, frame, w, h, detection_count):
        """Draw detection count and model info at bottom-left."""
        y_base = h - 60

        cv2.putText(frame, f"DETECTIONS: {detection_count}", (20, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, config.HUD_FONT_SCALE_LABEL,
                    config.COLOR_HUD_TEXT, 1, cv2.LINE_AA)

        cv2.putText(frame, "MODEL: YOLOv4-tiny | 416x416 | 80 CLASSES", (20, y_base + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, config.HUD_FONT_SCALE_SMALL,
                    config.COLOR_HUD_ACCENT, 1, cv2.LINE_AA)

    def _draw_distance_legend(self, frame, w, h):
        """Draw color-coded distance legend at bottom-right."""
        legend_x = w - 180
        legend_y = h - 80

        items = [
            (config.COLOR_SAFE, "SAFE (>3m)"),
            (config.COLOR_NEAR, "NEAR (1.5-3m)"),
            (config.COLOR_VERY_CLOSE, "V.CLOSE (<1.5m)"),
        ]

        for i, (color, text) in enumerate(items):
            y = legend_y + i * 18
            cv2.circle(frame, (legend_x, y), 5, color, -1, cv2.LINE_AA)
            cv2.putText(frame, text, (legend_x + 12, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, config.HUD_FONT_SCALE_SMALL,
                        color, 1, cv2.LINE_AA)
