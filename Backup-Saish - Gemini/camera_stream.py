"""
camera_stream.py — Pi Zero 2W + Camera Module 3 (IMX708)

FIXES applied:
  1. LAG     — Drop to 320x240 for zero-lag on Zero 2W; use video pipeline
               with low framerate cap; drop buffer_count to 2
  2. COLORS  — Capture in BGR888 (what picamera2 actually gives OpenCV),
               then encode directly via cv2.imencode so browser sees correct colors
  3. FLIP    — Transform(hflip=True) at ISP level (zero CPU cost) is the
               sole horizontal flip. cv2.flip removed — applying BOTH caused a
               double-inversion that made the stream appear as a mirrored laptop cam.
  4. CORS    — flask_cors CORS(app) covers all routes by default, but the MJPEG
               /stream route also gets explicit after_request headers as a belt-and-
               suspenders fix. The browser's crossOrigin='anonymous' on the <img>
               tag will only work if the server returns Access-Control-Allow-Origin.
               Without this header the canvas gets tainted and drawImage() silently
               fails, making the Pi feed appear blank (or shows the laptop webcam
               frame that was there before switching).

Run on Pi:
  python3 camera_stream.py
"""

import io
import time
import logging
import threading
import cv2
import numpy as np
from flask import Flask, Response, jsonify, send_file, request
from flask_cors import CORS

# ──────────────────────────────────────────────────────────────────────────────
# TUNING — adjust these for your scene
# ──────────────────────────────────────────────────────────────────────────────
STREAM_W      = 320          # width  — 320×240 is lag-free on Zero 2W
STREAM_H      = 240          # height
STREAM_FPS    = 15           # target framerate — 15fps is smooth & sustainable
JPEG_QUALITY  = 60           # 60 is plenty for monitoring; lower = less lag
PORT          = 5001
HOST          = '0.0.0.0'
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('neurolens_cam')

app = Flask(__name__)

# FIX: CORS(app) enables Access-Control-Allow-Origin: * on ALL routes, including
# /stream. This is what allows the browser's <img crossOrigin="anonymous"> to
# read MJPEG frames into a canvas without tainting it.
# Without CORS headers on /stream, drawImage() into the canvas throws a
# SecurityError and the Pi feed never appears — the silent catch block in the
# old app.js was hiding this failure completely.
CORS(app, resources={r"/*": {"origins": "*"}})

# ─── Shared frame state ───────────────────────────────────────────────────────
latest_frame_lock = threading.Lock()
latest_frame      = None          # holds the latest JPEG bytes
camera_ready      = False

# ─── Camera capture thread ───────────────────────────────────────────────────
def camera_capture_loop():
    """
    Runs in a background thread.
    Captures frames from picamera2, fixes colors, flips, encodes to JPEG,
    stores in latest_frame for streaming.
    """
    global latest_frame, camera_ready

    from picamera2 import Picamera2
    from libcamera import Transform

    picam2 = Picamera2()

    # Use "BGR888" format — this is what OpenCV expects natively.
    # Despite the confusing name, picamera2 BGR888 gives correct RGB output
    # when encoded to JPEG via cv2.imencode (which expects BGR input).
    config = picam2.create_video_configuration(
        main={"size": (STREAM_W, STREAM_H), "format": "BGR888"},
        controls={"FrameDurationLimits": (int(1e6 // STREAM_FPS),
                                          int(1e6 // STREAM_FPS))},
        transform=Transform(hflip=True),   # ISP-level mirror — zero CPU cost
        buffer_count=2,                    # minimal buffers = less latency
    )
    picam2.configure(config)
    picam2.start()
    log.info(f"📷 Camera started: {STREAM_W}×{STREAM_H} @ {STREAM_FPS}fps")
    time.sleep(1)  # let AE/AWB settle

    # Autofocus
    try:
        from libcamera import controls
        picam2.set_controls({
            "AfMode":  controls.AfModeEnum.Continuous,
            "AfSpeed": controls.AfSpeedEnum.Fast,
        })
        log.info("✅ PDAF autofocus active")
    except Exception as e:
        log.warning(f"⚠️  AF not available: {e}")

    camera_ready = True
    jpeg_params  = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    while True:
        # capture_array() returns a NumPy array in BGR888 format as configured.
        frame_bgr = picam2.capture_array()

        # FIX: REMOVED cv2.flip(frame_bgr, 1) — the ISP-level Transform(hflip=True)
        # above already mirrors the frame at zero CPU cost before it leaves the sensor.
        # Applying cv2.flip() on top of that was a double-inversion: the result looked
        # correct on the Pi locally but appeared as a mirrored image in the browser,
        # making it indistinguishable from a laterally-inverted laptop webcam feed.
        # Rule: use ONE flip method. Transform(hflip=True) is preferred (ISP-level,
        # no CPU cost). cv2.flip is a fallback if Transform is unavailable.

        # cv2.imencode expects BGR — frame_bgr is already BGR, so encode directly.
        ret, jpeg = cv2.imencode('.jpg', frame_bgr, jpeg_params)
        if ret:
            with latest_frame_lock:
                latest_frame = jpeg.tobytes()


# Start camera thread immediately on import
cam_thread = threading.Thread(target=camera_capture_loop, daemon=True)
cam_thread.start()


# ─── MJPEG stream generator ──────────────────────────────────────────────────
def generate_frames():
    """Yield MJPEG frames to the browser."""
    while True:
        with latest_frame_lock:
            frame = latest_frame
        if frame is not None:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )
        else:
            time.sleep(0.05)  # wait for first frame


# ─── Routes ──────────────────────────────────────────────────────────────────

# FIX: Belt-and-suspenders CORS — explicitly inject headers on every response.
# flask_cors handles it for standard JSON endpoints, but some browsers are
# stricter about CORS on multipart/x-mixed-replace (MJPEG) responses and may
# ignore the flask_cors headers. This after_request hook guarantees the header
# is present regardless of content type.
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response
@app.route('/')
def index():
    return jsonify({
        'service':  'NeuroLens Camera (Pi Zero 2W)',
        'camera':   'Camera Module 3 (IMX708)',
        'stream':   f'http://{HOST}:{PORT}/stream',
        'capture':  f'http://{HOST}:{PORT}/capture',
        'health':   f'http://{HOST}:{PORT}/health',
    })


@app.route('/stream')
def stream():
    """Live MJPEG — correct colors, mirrored, low-lag."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/capture')
def capture():
    """Return the latest frame as a downloadable JPEG."""
    with latest_frame_lock:
        frame = latest_frame
    if frame is None:
        return jsonify({'error': 'Camera not ready yet'}), 503
    buf = io.BytesIO(frame)
    buf.seek(0)
    return send_file(
        buf,
        mimetype='image/jpeg',
        as_attachment=True,
        download_name=f'capture_{int(time.time())}.jpg'
    )


@app.route('/health')
def health():
    return jsonify({
        'status':      'ok' if camera_ready else 'starting',
        'device':      'Raspberry Pi Zero 2W',
        'camera':      'Camera Module 3 (IMX708)',
        'resolution':  f'{STREAM_W}×{STREAM_H}',
        'fps_target':  STREAM_FPS,
        'jpeg_quality': JPEG_QUALITY,
        'color_fix':   'BGR888 + cv2.imencode (correct JPEG colors)',
        # FIX: reflects single-flip approach — ISP hflip only, no cv2.flip
        'flip_fix':    'Transform(hflip=True) at ISP level only (no double-flip)',
        'cors':        'Access-Control-Allow-Origin: * (flask_cors + after_request)',
    })


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    log.info("=" * 55)
    log.info("  NeuroLens — Pi Zero 2W Camera Stream")
    log.info(f"  Resolution : {STREAM_W}×{STREAM_H} @ {STREAM_FPS}fps")
    log.info(f"  JPEG quality: {JPEG_QUALITY}%")
    log.info(f"  Color fix  : BGR888 → cv2.imencode (no channel swap)")
    log.info(f"  Flip fix   : ISP hflip ONLY (cv2.flip removed — was double-inverting)")
    log.info(f"  CORS fix   : Access-Control-Allow-Origin: * on all routes")
    log.info(f"  Stream     : http://[pi-ip]:{PORT}/stream")
    log.info("=" * 55)
    app.run(host=HOST, port=PORT, threaded=True)