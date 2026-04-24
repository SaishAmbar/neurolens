"""
NeuroLens — Distance Estimator
Estimates distance to detected objects using the pinhole camera model
and classifies them into SAFE / NEAR / VERY CLOSE zones.
"""

from . import config


class DistanceResult:
    """Holds distance estimation results for a single detection."""

    __slots__ = (
        "detection", "estimated_cm", "zone", "zone_color",
        "direction", "danger_score",
    )

    def __init__(self, detection, estimated_cm, zone, zone_color, direction, danger_score):
        self.detection = detection
        self.estimated_cm = estimated_cm
        self.zone = zone
        self.zone_color = zone_color
        self.direction = direction
        self.danger_score = danger_score

    def __repr__(self):
        return (
            f"DistanceResult({self.detection.class_name}, "
            f"dist={self.estimated_cm:.0f}cm, zone={self.zone}, "
            f"dir={self.direction}, danger={self.danger_score:.2f})"
        )


class DistanceEstimator:
    """Estimates distance and classifies threat levels for detected objects."""

    def __init__(self, frame_width=config.CAMERA_WIDTH, frame_height=config.CAMERA_HEIGHT):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._focal_length = config.FOCAL_LENGTH_PX

    def update_frame_size(self, width, height):
        """Update the frame dimensions (call when camera resolution changes)."""
        self.frame_width = width
        self.frame_height = height
        # Recalculate focal length based on new width
        # Assuming ~78° FOV: f = (w/2) / tan(39°) ≈ w * 0.43
        # This matches the recalibrated FOCAL_LENGTH_PX in config.py
        self._focal_length = width * 0.43

    def estimate(self, detection):
        """
        Estimate distance for a single detection.

        Uses the pinhole camera model: distance = (real_height × focal_length) / pixel_height
        Falls back to bbox-to-frame ratio if the object type is unknown.

        Args:
            detection: Detection object

        Returns:
            DistanceResult
        """
        x, y, w, h = detection.bbox
        bbox_height = max(h, 1)  # avoid division by zero

        # ── Primary: Pinhole camera model ─────────────────────────────────
        known_height = config.KNOWN_HEIGHTS_CM.get(detection.class_name)

        if known_height is not None and bbox_height > 0:
            estimated_cm = (known_height * self._focal_length) / bbox_height
        else:
            # Fallback: estimate from bbox ratio (assume generic 100cm object)
            ratio = bbox_height / self.frame_height
            estimated_cm = max(50, (1.0 - ratio) * 500)  # rough linear mapping

        # ── Classify zone ─────────────────────────────────────────────────
        zone, zone_color = self._classify_zone(estimated_cm, bbox_height)

        # ── Determine direction ───────────────────────────────────────────
        direction = self._classify_direction(detection.center_x)

        # ── Calculate danger score ────────────────────────────────────────
        danger_score = self._calculate_danger(estimated_cm, detection.center_x, detection.class_name)

        return DistanceResult(
            detection=detection,
            estimated_cm=estimated_cm,
            zone=zone,
            zone_color=zone_color,
            direction=direction,
            danger_score=danger_score,
        )

    def estimate_all(self, detections):
        """
        Estimate distances for all detections and sort by danger score.

        Returns:
            List of DistanceResult, sorted by danger_score descending (most dangerous first)
        """
        results = [self.estimate(d) for d in detections]
        results.sort(key=lambda r: r.danger_score, reverse=True)
        return results

    def _classify_zone(self, estimated_cm, bbox_height):
        """Classify distance zone using both absolute distance and bbox ratio."""
        bbox_ratio = bbox_height / self.frame_height

        # Use the more aggressive (closer) of the two estimates
        if estimated_cm < config.DIST_VERY_CLOSE_CM or bbox_ratio > config.BBOX_VERY_CLOSE_RATIO:
            return config.ZONE_VERY_CLOSE, config.COLOR_VERY_CLOSE
        elif estimated_cm < config.DIST_NEAR_CM or bbox_ratio > config.BBOX_NEAR_RATIO:
            return config.ZONE_NEAR, config.COLOR_NEAR
        else:
            return config.ZONE_SAFE, config.COLOR_SAFE

    def _classify_direction(self, center_x):
        """Classify which zone of the frame the object is in."""
        third = self.frame_width / 3
        if center_x < third:
            return "LEFT"
        elif center_x > third * 2:
            return "RIGHT"
        else:
            return "CENTER"

    def _calculate_danger(self, estimated_cm, center_x, class_name):
        """
        Calculate a composite danger score (0-1).
        Higher = more dangerous.

        Factors:
        - Distance (closer = higher danger)
        - Horizontal position (center = higher danger, objects in your path)
        - Object type (larger objects = slightly higher danger)
        """
        # Distance factor: 1.0 at 0cm, 0.0 at 500cm+
        dist_factor = max(0, 1.0 - (estimated_cm / 500.0))

        # Center factor: 1.0 at center, 0.5 at edges
        center_offset = abs(center_x - self.frame_width / 2) / (self.frame_width / 2)
        center_factor = 1.0 - (center_offset * 0.5)

        # Size factor: larger real-world objects are slightly more dangerous
        known_height = config.KNOWN_HEIGHTS_CM.get(class_name, 50)
        size_factor = min(1.0, known_height / 200.0)

        # Weighted composite
        danger = (dist_factor * 0.6) + (center_factor * 0.3) + (size_factor * 0.1)
        return min(1.0, max(0.0, danger))
