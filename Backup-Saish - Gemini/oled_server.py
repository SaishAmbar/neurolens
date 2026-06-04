"""
oled_server.py — SSD1306 OLED display server for NeuroLens
Runs on Raspberry Pi Zero 2W alongside camera_stream.py.

Hardware:
  - SSD1306 128x64 OLED via I2C (address 0x3C, bus 1)
  - Wire: SDA → GPIO2 (Pin 3), SCL → GPIO3 (Pin 5), VCC → 3.3V, GND → GND

Install deps on Pi:
  pip install flask flask-cors luma.oled Pillow

Run:
  python3 oled_server.py

Endpoints:
  POST /oled    — { mode, line1, line2, scroll }
  GET  /health  — connection check
"""

import threading
import time
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('oled_server')

app = Flask(__name__)
CORS(app)

# ─── OLED initialisation ─────────────────────────────────────────────────────
try:
    from luma.core.interface.serial import i2c
    from luma.oled.device import ssd1306
    from luma.core.render import canvas
    from PIL import ImageFont

    serial = i2c(port=1, address=0x3C)
    device = ssd1306(serial, width=128, height=64)
    log.info("✅ SSD1306 OLED initialised (128×64, I2C 0x3C)")
    OLED_AVAILABLE = True
except Exception as e:
    log.warning(f"⚠️  OLED not available: {e}  (running in dry-run mode)")
    device = None
    OLED_AVAILABLE = False

# ─── Fonts ───────────────────────────────────────────────────────────────────
# Use built-in PIL bitmap font as fallback — always available, no file needed.
# For better looks: copy a .ttf to /home/pi/ and point FONT_PATH at it.
FONT_PATH = None   # e.g. '/home/pi/DejaVuSans.ttf'

# Common TTF paths on Raspberry Pi OS — tried in order
_FALLBACK_FONTS = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
    '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    '/usr/share/fonts/truetype/piboto/Piboto-Bold.ttf',
]

def _load_font(size):
    # 1. User-specified font
    if FONT_PATH:
        try:
            return ImageFont.truetype(FONT_PATH, size)
        except Exception:
            log.warning(f'Could not load FONT_PATH={FONT_PATH!r}, falling back.')
    # 2. Common Pi system fonts
    for path in _FALLBACK_FONTS:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    # 3. Pillow 10+ supports load_default(size=)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        pass
    # 4. Ancient Pillow — tiny bitmap, unavoidable
    log.warning('Using tiny default bitmap font — install python3-pil.imagetk or a TTF for readable text.')
    return ImageFont.load_default()

FONT_LARGE  = _load_font(24)   # line1: bigger and bold
FONT_SMALL  = _load_font(18)   # line2: readable for scrolling text

# ─── Scroll state ────────────────────────────────────────────────────────────
_scroll_lock   = threading.Lock()
_scroll_thread = None
_scroll_stop   = threading.Event()

DISPLAY_W = 128   # OLED width in pixels
SCROLL_FPS = 30   # frames per second for scroll animation
SCROLL_PX  = 8    # pixels to shift per frame — doubled for noticeably faster scroll

# ─── Core display function ───────────────────────────────────────────────────
def _stop_scroll():
    """Stop any running scroll animation thread."""
    global _scroll_thread
    _scroll_stop.set()
    if _scroll_thread and _scroll_thread.is_alive():
        _scroll_thread.join(timeout=1.0)
    _scroll_stop.clear()
    _scroll_thread = None


def _draw_static(line1: str, line2: str):
    """Draw two lines of text, no animation."""
    if not OLED_AVAILABLE:
        log.info(f"[DRY-RUN] line1='{line1}'  line2='{line2}'")
        return
    with canvas(device) as draw:
        draw.text((0, 0),  line1, font=FONT_LARGE, fill="white")
        draw.text((0, 36), line2, font=FONT_SMALL,  fill="white")


def _scroll_worker(line1: str, line2: str):
    """
    Renders line1 statically and horizontally scrolls line2.
    Runs in a background thread; checks _scroll_stop to exit cleanly.
    """
    if not OLED_AVAILABLE:
        log.info(f"[DRY-RUN SCROLL] line1='{line1}'  line2='{line2}'")
        return

    # Measure text width using font metrics
    try:
        bbox = FONT_SMALL.getbbox(line2)
        text_w = bbox[2] - bbox[0]
    except AttributeError:
        # Older Pillow
        text_w = FONT_SMALL.getsize(line2)[0]

    # Only scroll if text is wider than display
    if text_w <= DISPLAY_W:
        _draw_static(line1, line2)
        return

    offset = 0
    total_w = text_w + DISPLAY_W   # full scroll distance before looping

    while not _scroll_stop.is_set():
        with canvas(device) as draw:
            draw.text((0, 0), line1, font=FONT_LARGE, fill="white")
            # Scroll: start text at (DISPLAY_W - offset) moving left
            draw.text((DISPLAY_W - offset, 36), line2, font=FONT_SMALL, fill="white")

        offset = (offset + SCROLL_PX) % total_w
        time.sleep(1.0 / SCROLL_FPS)


def _clear_screen():
    if not OLED_AVAILABLE:
        log.info("[DRY-RUN] Screen cleared")
        return
    with canvas(device) as draw:
        draw.rectangle(device.bounding_box, outline="black", fill="black")


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route('/oled', methods=['POST'])
def oled_update():
    data = request.get_json(silent=True) or {}
    mode  = data.get('mode', '')
    line1 = str(data.get('line1', ''))
    line2 = str(data.get('line2', ''))
    scroll = bool(data.get('scroll', False))

    log.info(f"OLED ← mode={mode!r}  scroll={scroll}  | {line1!r} / {line2!r}")

    # Stop any active scroll thread before doing anything new
    _stop_scroll()

    # Empty data → clear screen
    if not line1 and not line2:
        _clear_screen()
        return jsonify({'status': 'cleared'}), 200

    if scroll and line2:
        # Launch scroll animation in background thread
        global _scroll_thread
        _scroll_thread = threading.Thread(
            target=_scroll_worker,
            args=(line1, line2),
            daemon=True,
        )
        _scroll_thread.start()
    else:
        _draw_static(line1, line2)

    return jsonify({'status': 'ok', 'mode': mode}), 200


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'oled': 'connected' if OLED_AVAILABLE else 'dry-run',
        'port': 5002,
        'address': '0x3C',
    })


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    log.info("=" * 55)
    log.info("  NeuroLens OLED Server — SSD1306 128×64 I2C")
    log.info(f"  OLED     : {'OK' if OLED_AVAILABLE else 'DRY-RUN (no hardware)'}")
    log.info("  Endpoints: POST /oled   GET /health")
    log.info("  Port     : 5002")
    log.info("=" * 55)
    app.run(host='0.0.0.0', port=5002, threaded=True)