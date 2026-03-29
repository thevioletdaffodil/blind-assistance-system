"""
Blind Assistance System — Web Demo Edition
==========================================
Run this script, then open  http://<YOUR-LAPTOP-IP>:5000  on any phone or browser.
Both devices must be on the same Wi-Fi network.

Features:
  1.  Scene summary (press S on laptop, or tap button in browser)
  2.  Object counter — live per-class tally
  3.  Danger zone line — red line at ~1 m depth
  4.  Navigation / Indoor mode toggle (press M or tap in browser)
  5.  FPS counter
  6.  Obstacle-free path indicator — coloured zone overlays
  7.  Approaching / receding detection
  8.  Session heatmap — shown in browser after session ends
  9.  Console log with timestamps
  10. Flask web dashboard — live MJPEG stream + stats panel on phone

Install extras:  pip install flask
Controls (laptop window):  ESC = quit  |  S = scene summary  |  M = toggle mode
"""

import cv2
from ultralytics import YOLO
import pyttsx3
import threading
import time
import queue
import numpy as np
from datetime import datetime
from collections import defaultdict, deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64, os, json
from flask import Flask, Response, jsonify, request

# ═══════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════
WEBCAM_URL   = "http://10.21.146.58:8080/video"
MODEL_NAME   = "yolov8s.pt"
WEB_PORT     = 5000

FRAME_WIDTH      = 640
FRAME_HEIGHT     = 480
PROCESS_EVERY_N  = 2

SPEAK_COOLDOWN   = 2.5
OBJECT_COOLDOWN  = 7.0
SMOOTH_WINDOW    = 5
SMOOTH_MIN_HITS  = 3

OUTDOOR_OBJECTS = {"bus", "truck", "train", "airplane", "boat",
                   "traffic light", "fire hydrant", "stop sign", "parking meter"}

NAV_OBJECTS = {
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
    "traffic light", "stop sign", "fire hydrant", "bench",
    "dog", "cat", "backpack", "suitcase", "umbrella",
}
INDOOR_OBJECTS = {
    "person", "chair", "dining table", "laptop", "keyboard", "mouse",
    "tv", "monitor", "cell phone", "bottle", "cup", "book",
    "backpack", "suitcase", "bed", "toilet", "sink", "refrigerator",
    "clock", "vase", "potted plant", "dog", "cat",
}
IGNORE_ALWAYS = {
    "sports ball", "frisbee", "skis", "snowboard", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "wine glass", "fork", "knife", "spoon",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "toaster",
    "hair drier", "scissors", "teddy bear",
}
PRIORITY_HIGH = {"person", "car", "truck", "bus", "motorcycle", "bicycle", "dog", "cat"}

CONF_INDOOR  = 0.40
CONF_OUTDOOR = 0.70

FOCAL_LENGTH_PX  = 800
REAL_HEIGHTS_M   = {
    "person": 1.70, "car": 1.50, "truck": 3.00, "bus": 3.20,
    "motorcycle": 1.10, "bicycle": 1.00, "dog": 0.50, "cat": 0.25,
    "chair": 0.90, "dining table": 0.75, "laptop": 0.25,
    "keyboard": 0.04, "mouse": 0.04, "tv": 0.60, "monitor": 0.45,
    "cell phone": 0.15, "bottle": 0.25, "cup": 0.12,
    "backpack": 0.50, "suitcase": 0.65, "umbrella": 0.90,
    "bench": 0.45, "bed": 0.55, "toilet": 0.40, "sink": 0.25,
    "refrigerator": 1.80, "book": 0.22, "clock": 0.30,
    "vase": 0.30, "potted plant": 0.40, "fire hydrant": 0.60,
    "stop sign": 0.75, "traffic light": 0.80,
}
DEFAULT_HEIGHT_M = 0.50

APPROACH_HISTORY_LEN = 6
APPROACH_THRESHOLD   = 0.4
APPROACH_COOLDOWN    = 5.0


# ═══════════════════════════════════════════════════════
#  SHARED STATE  (written by main loop, read by Flask)
# ═══════════════════════════════════════════════════════
state_lock       = threading.Lock()
shared_frame     = None          # latest annotated JPEG bytes
shared_stats     = {             # stats sent to browser as JSON
    "fps": 0,
    "mode": "indoor",
    "frame": 0,
    "objects": {},               # label -> count this session
    "current": [],               # list of {label, conf, dirn, dist, urgency}
    "free_zones": [],
    "last_announcement": "",
    "elapsed": 0,
    "heatmap_b64": "",           # filled on session end
}
trigger_summary  = threading.Event()   # set when S pressed / button tapped
trigger_mode     = threading.Event()   # set when M pressed / button tapped
session_running  = threading.Event()
session_running.set()


# ═══════════════════════════════════════════════════════
#  FLASK APP
# ═══════════════════════════════════════════════════════
app = Flask(__name__)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Blind Assist</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d0d0d; color: #e8e8e8; font-family: 'Segoe UI', sans-serif;
         display: flex; flex-direction: column; align-items: center; min-height: 100vh; }
  header { width: 100%; background: #111; padding: 10px 16px;
           display: flex; align-items: center; justify-content: space-between;
           border-bottom: 2px solid #333; }
  header h1 { font-size: 1rem; font-weight: 700; letter-spacing: 1px;
              color: #4cf; text-transform: uppercase; }
  #mode-badge { padding: 4px 10px; border-radius: 20px; font-size: 0.72rem;
                font-weight: 700; letter-spacing: 1px; text-transform: uppercase; }
  .mode-indoor { background: #1a3a4a; color: #4cf; }
  .mode-navigation { background: #3a2a10; color: #fc8; }

  #stream-wrap { position: relative; width: 100%; max-width: 640px; }
  #stream { width: 100%; display: block; border-bottom: 2px solid #222; }

  /* zone bar overlaid at top of stream */
  #zone-bar { position: absolute; top: 0; left: 0; width: 100%;
              display: flex; height: 28px; pointer-events: none; }
  .zone { flex: 1; display: flex; align-items: center; justify-content: center;
          font-size: 0.65rem; font-weight: 700; letter-spacing: 1px;
          opacity: 0.82; transition: background 0.3s; }
  .zone.clear   { background: rgba(0,180,80,0.55); color: #fff; }
  .zone.blocked { background: rgba(200,30,30,0.55); color: #fff; }

  #stats { width: 100%; max-width: 640px; padding: 10px 12px; }

  /* announcement banner */
  #announce { width: 100%; background: #1a1a1a; border-left: 4px solid #4cf;
              padding: 8px 12px; margin-bottom: 10px; border-radius: 4px;
              font-size: 0.85rem; min-height: 36px; transition: border-color 0.3s; }
  #announce.danger { border-color: #f44; color: #faa; }

  /* info row */
  .info-row { display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
  .chip { background: #1e1e1e; border-radius: 6px; padding: 5px 10px;
          font-size: 0.75rem; color: #aaa; }
  .chip span { color: #fff; font-weight: 600; }

  /* detections list */
  #det-list { width: 100%; }
  .det-row { display: flex; align-items: center; gap: 8px;
             background: #161616; border-radius: 6px; padding: 7px 10px;
             margin-bottom: 6px; border-left: 4px solid #333;
             transition: border-color 0.3s; }
  .det-row.critical { border-color: #f44; }
  .det-row.close    { border-color: #f90; }
  .det-row.nearby   { border-color: #4c4; }
  .det-row.far      { border-color: #444; }
  .det-label { font-weight: 700; font-size: 0.88rem; flex: 1; }
  .det-meta  { font-size: 0.73rem; color: #888; }
  .det-dist  { font-size: 0.8rem; font-weight: 600; min-width: 52px; text-align: right; }
  .det-dist.critical { color: #f55; }
  .det-dist.close    { color: #fa0; }
  .det-dist.nearby   { color: #4d4; }

  /* session totals */
  #totals-wrap { margin-top: 10px; }
  #totals-wrap h3 { font-size: 0.72rem; color: #666; text-transform: uppercase;
                    letter-spacing: 1px; margin-bottom: 6px; }
  #totals { display: flex; flex-wrap: wrap; gap: 6px; }
  .total-chip { background: #1c2c1c; border-radius: 4px; padding: 3px 8px;
                font-size: 0.72rem; color: #8d8; }

  /* buttons */
  #btn-row { display: flex; gap: 10px; margin-bottom: 12px; }
  button { flex: 1; padding: 11px 0; border-radius: 8px; border: none;
           font-size: 0.88rem; font-weight: 700; cursor: pointer;
           letter-spacing: 0.5px; transition: opacity 0.15s; }
  button:active { opacity: 0.7; }
  #btn-summary { background: #1a4060; color: #4cf; }
  #btn-mode    { background: #403010; color: #fc8; }

  /* heatmap modal */
  #heatmap-wrap { display: none; position: fixed; inset: 0;
                  background: rgba(0,0,0,0.92); z-index: 100;
                  flex-direction: column; align-items: center;
                  justify-content: center; padding: 20px; }
  #heatmap-wrap img { max-width: 100%; border-radius: 8px; }
  #close-hm { margin-top: 14px; padding: 10px 30px; background: #333;
              color: #fff; border: none; border-radius: 8px;
              font-size: 1rem; cursor: pointer; }
  #hm-btn-row { margin-top: 12px; }
  #show-heatmap { display: none; background: #2a1060; color: #c8f; }
</style>
</head>
<body>
<header>
  <h1>&#x1F441; Blind Assist</h1>
  <span id="mode-badge" class="mode-indoor">INDOOR</span>
</header>

<div id="stream-wrap">
  <img id="stream" src="/video_feed" alt="Live feed">
  <div id="zone-bar">
    <div class="zone" id="z-left">LEFT</div>
    <div class="zone" id="z-centre">CENTRE</div>
    <div class="zone" id="z-right">RIGHT</div>
  </div>
</div>

<div id="stats">
  <div id="announce">Connecting…</div>

  <div id="btn-row">
    <button id="btn-summary" onclick="triggerSummary()">&#x1F4E2; Scene Summary</button>
    <button id="btn-mode"    onclick="triggerMode()">&#x1F504; Toggle Mode</button>
    <button id="show-heatmap" onclick="showHeatmap()">&#x1F321; Heatmap</button>
  </div>

  <div class="info-row">
    <div class="chip">FPS <span id="fps">—</span></div>
    <div class="chip">Frame <span id="framenum">—</span></div>
    <div class="chip">Elapsed <span id="elapsed">—</span></div>
  </div>

  <div id="det-list"></div>

  <div id="totals-wrap">
    <h3>Session totals</h3>
    <div id="totals"></div>
  </div>
</div>

<div id="heatmap-wrap">
  <img id="heatmap-img" src="" alt="Heatmap">
  <div id="hm-btn-row">
    <button id="close-hm" onclick="closeHeatmap()">Close</button>
  </div>
</div>

<script>
let lastAnnouncement = "";
let heatmapB64 = "";

function triggerSummary() {
  fetch("/trigger/summary", { method: "POST" });
}
function triggerMode() {
  fetch("/trigger/mode", { method: "POST" });
}
function showHeatmap() {
  if (!heatmapB64) return;
  document.getElementById("heatmap-img").src = "data:image/png;base64," + heatmapB64;
  document.getElementById("heatmap-wrap").style.display = "flex";
}
function closeHeatmap() {
  document.getElementById("heatmap-wrap").style.display = "none";
}

function hhmmss(s) {
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = Math.floor(s%60);
  return [h,m,sec].map(x=>String(x).padStart(2,'0')).join(':');
}

async function poll() {
  try {
    const r = await fetch("/stats");
    const d = await r.json();

    // mode badge
    const badge = document.getElementById("mode-badge");
    badge.textContent = d.mode.toUpperCase();
    badge.className = "mode-" + d.mode;

    // info chips
    document.getElementById("fps").textContent       = d.fps;
    document.getElementById("framenum").textContent  = d.frame;
    document.getElementById("elapsed").textContent   = hhmmss(d.elapsed);

    // announcement
    const ann = document.getElementById("announce");
    if (d.last_announcement && d.last_announcement !== lastAnnouncement) {
      lastAnnouncement = d.last_announcement;
      ann.textContent  = d.last_announcement;
      ann.className    = d.last_announcement.startsWith("Danger") ? "danger" : "";
      ann.style.background = "#252525";
      setTimeout(() => ann.style.background = "#1a1a1a", 800);
    }

    // zone bar
    const zones = ["left", "centre", "right"];
    zones.forEach(z => {
      const el = document.getElementById("z-" + z);
      const free = d.free_zones.includes(z);
      el.className = "zone " + (free ? "clear" : "blocked");
      el.textContent = z.toUpperCase() + (free ? " ✓" : " ✗");
    });

    // detections list
    const list = document.getElementById("det-list");
    list.innerHTML = d.current.map(obj => `
      <div class="det-row ${obj.urgency}">
        <div class="det-label">${obj.label}</div>
        <div class="det-meta">${obj.conf}% · ${obj.dirn}</div>
        <div class="det-dist ${obj.urgency}">${obj.dist || "—"}</div>
      </div>`).join("");

    // session totals
    const totals = document.getElementById("totals");
    totals.innerHTML = Object.entries(d.objects)
      .sort((a,b) => b[1]-a[1])
      .map(([l,c]) => `<span class="total-chip">${l} <b>${c}</b></span>`)
      .join("");

    // heatmap (available after session ends)
    if (d.heatmap_b64 && d.heatmap_b64 !== heatmapB64) {
      heatmapB64 = d.heatmap_b64;
      document.getElementById("show-heatmap").style.display = "block";
    }

  } catch(e) { /* connection blip — ignore */ }
  setTimeout(poll, 500);   // poll every 500 ms
}

poll();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return HTML_PAGE

def mjpeg_generator():
    """Yield MJPEG frames from shared_frame."""
    blank = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.putText(blank, "Waiting for stream...", (120, FRAME_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    _, blank_jpg = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 70])
    blank_bytes  = blank_jpg.tobytes()

    while True:
        with state_lock:
            frame_bytes = shared_frame if shared_frame else blank_bytes
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + frame_bytes + b"\r\n")
        time.sleep(0.04)   # ~25 fps to browser

@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stats")
def stats():
    with state_lock:
        return jsonify(shared_stats)

@app.route("/trigger/summary", methods=["POST"])
def api_summary():
    trigger_summary.set()
    return "", 204

@app.route("/trigger/mode", methods=["POST"])
def api_mode():
    trigger_mode.set()
    return "", 204

def run_flask():
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)          # silence Flask request logs
    app.run(host="0.0.0.0", port=WEB_PORT, threaded=True)

threading.Thread(target=run_flask, daemon=True).start()


# ═══════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════
def estimate_distance(label, box_h_px):
    if box_h_px <= 0: return None
    return round((REAL_HEIGHTS_M.get(label, DEFAULT_HEIGHT_M) * FOCAL_LENGTH_PX) / box_h_px, 1)

def distance_label(m):
    if m is None: return ""
    return f"{int(m*100)} cm" if m < 0.5 else f"{m} m"

def distance_urgency(m):
    if m is None: return "far"
    if m < 1.0:   return "critical"
    if m < 2.5:   return "close"
    if m < 5.0:   return "nearby"
    return "far"

def conf_threshold_for(label):
    return CONF_OUTDOOR if label in OUTDOOR_OBJECTS else CONF_INDOOR

def get_direction(cx, w):
    if cx < w // 3:      return "on your left"
    if cx > 2 * w // 3:  return "on your right"
    return "ahead"

approach_history     = defaultdict(lambda: deque(maxlen=APPROACH_HISTORY_LEN))
approach_last_spoken = {}

def update_approach(label, dist_m, now):
    if dist_m is None: return None
    h = approach_history[label]
    h.append((now, dist_m))
    if len(h) < APPROACH_HISTORY_LEN: return None
    times = np.array([x[0] for x in h]); dists = np.array([x[1] for x in h])
    times -= times[0]
    if times[-1] == 0: return None
    rate = -np.polyfit(times, dists, 1)[0]
    if rate >  APPROACH_THRESHOLD: return "approaching"
    if rate < -APPROACH_THRESHOLD: return "receding"
    return None

ZONE_NAMES = ["left", "centre", "right"]
def get_free_zones(boxes, w):
    occupied, third = set(), w // 3
    for (x1,_,x2,*__) in boxes:
        cx = (x1+x2)//2
        occupied.add("left" if cx < third else "right" if cx > 2*third else "centre")
    return [z for z in ZONE_NAMES if z not in occupied]


# ═══════════════════════════════════════════════════════
#  HEATMAP
# ═══════════════════════════════════════════════════════
heatmap_accum = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)

def add_to_heatmap(x1, y1, x2, y2):
    heatmap_accum[max(0,y1):min(FRAME_HEIGHT,y2),
                  max(0,x1):min(FRAME_WIDTH, x2)] += 1.0

def heatmap_to_b64():
    if heatmap_accum.max() == 0: return ""
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(heatmap_accum, cmap="hot", interpolation="gaussian",
                   origin="upper", aspect="auto")
    plt.colorbar(im, ax=ax, label="Detection frequency")
    ax.set_title("Object Detection Heatmap")
    ax.set_xlabel("X (px)"); ax.set_ylabel("Y (px)")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ═══════════════════════════════════════════════════════
#  SPEECH ENGINE
# ═══════════════════════════════════════════════════════
speech_queue = queue.Queue()

def speech_worker():
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 1.0)
    for v in engine.getProperty('voices'):
        if any(k in v.name.lower() for k in ('zira','david','english','hazel')):
            engine.setProperty('voice', v.id); break
    while True:
        text = speech_queue.get()
        if text is None: break
        try: engine.say(text); engine.runAndWait()
        except: pass

threading.Thread(target=speech_worker, daemon=True).start()

last_speech_time   = 0.0
object_last_spoken = {}

def try_speak(text, urgent=False):
    global last_speech_time
    now = time.time()
    if not urgent and now - last_speech_time < SPEAK_COOLDOWN: return
    if not urgent and not speech_queue.empty(): return
    speech_queue.put(text); last_speech_time = now

def speak_now(text):
    speech_queue.put(text)


# ═══════════════════════════════════════════════════════
#  SMOOTHING
# ═══════════════════════════════════════════════════════
recent_frames = deque(maxlen=SMOOTH_WINDOW)

def stable_detections(current_set):
    recent_frames.append(current_set)
    counts = defaultdict(int)
    for s in recent_frames:
        for d in s: counts[d] += 1
    return {d for d, n in counts.items() if n >= SMOOTH_MIN_HITS}


# ═══════════════════════════════════════════════════════
#  SPEECH BUILDER
# ═══════════════════════════════════════════════════════
def build_speech(detections_meta):
    now = time.time()
    critical, high, low = [], [], []
    for label, dirn, dist_m in detections_meta:
        key = (label, dirn)
        if now - object_last_spoken.get(key, 0) < OBJECT_COOLDOWN: continue
        urgency = distance_urgency(dist_m)
        phrase  = f"{label} {dirn}" + (f", {distance_label(dist_m)}" if dist_m else "")
        bucket  = critical if urgency == "critical" else high if label in PRIORITY_HIGH else low
        bucket.append((key, phrase))
    chosen = (critical[:2]+high[:1]) if critical else (high[:2]+low[:1]) if high else low[:2]
    if not chosen: return None, []
    keys, phrases = zip(*chosen)
    return ("Danger! " if critical else "Warning: ") + " and ".join(phrases), list(keys)

def build_scene_summary(boxes):
    if not boxes: return "Nothing detected in view."
    counts = defaultdict(int)
    for (_,_,_,_,label,*__) in boxes: counts[label] += 1
    parts = [f"{c} {l}" if c>1 else f"a {l}" for l,c in sorted(counts.items(), key=lambda x:-x[1])]
    return "I can see " + (parts[0] + "." if len(parts)==1 else
                           ", ".join(parts[:-1]) + f", and {parts[-1]}.")

# logging
session_start = time.time()
def log_detection(label, conf, dirn, dlabel, dist_m, announced):
    ts      = datetime.now().strftime("%H:%M:%S")
    elapsed = round(time.time()-session_start, 1)
    urg     = distance_urgency(dist_m)
    urg_s   = f"  ⚠ {urg.upper()}" if urg in ("critical","close") else ""
    tag     = ">> ANNOUNCED" if announced else "   detected "
    print(f"[{ts}] (+{elapsed:>7}s) {tag}  {label:<18} {int(conf*100):>3}%  "
          f"{dirn:<16}  [{dlabel}]{urg_s}")


# ═══════════════════════════════════════════════════════
#  MODEL + CAPTURE
# ═══════════════════════════════════════════════════════
print(f"Loading {MODEL_NAME}...")
model = YOLO(MODEL_NAME)
model(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8), verbose=False)
print(f"Model ready.")
print(f"\n{'─'*55}")
print(f"  Web dashboard → http://localhost:{WEB_PORT}")
print(f"  On your phone → http://<YOUR-LAPTOP-IP>:{WEB_PORT}")
print(f"  (find your IP: run  ipconfig  in a new terminal)")
print(f"{'─'*55}\n")
print("Controls: ESC = quit  |  S = scene summary  |  M = toggle mode")

def open_cap(url):
    c = cv2.VideoCapture(url)
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return c

cap             = open_cap(WEBCAM_URL)
frame_count     = 0
last_boxes      = []
mode            = "indoor"
object_counts   = defaultdict(int)
last_free_zones = []
free_zone_t     = 0.0
fps_times       = deque(maxlen=30)
fps_display     = 0.0
last_announce   = ""

try_speak("Blind assistance system started. Indoor mode active.")


# ═══════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════
while True:
    ret, frame = cap.read()
    if not ret:
        print("Connection lost. Reconnecting in 2 s...")
        time.sleep(2); cap.release(); cap = open_cap(WEBCAM_URL); continue

    frame_count += 1
    frame      = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame_area = FRAME_WIDTH * FRAME_HEIGHT

    # ── Keyboard + web triggers ───────────────
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break

    do_summary = (key == ord('s') or key == ord('S') or trigger_summary.is_set())
    do_mode    = (key == ord('m') or key == ord('M') or trigger_mode.is_set())
    trigger_summary.clear(); trigger_mode.clear()

    if do_summary:
        summary = build_scene_summary(last_boxes)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] SCENE SUMMARY: {summary}")
        speak_now(summary)
        last_announce = summary

    if do_mode:
        mode = "navigation" if mode == "indoor" else "indoor"
        msg  = f"Switched to {mode} mode."
        print(f"[{datetime.now().strftime('%H:%M:%S')}] MODE → {mode.upper()}")
        speak_now(msg)

    active_objects = NAV_OBJECTS if mode == "navigation" else INDOOR_OBJECTS

    # ── Inference ─────────────────────────────
    if frame_count % PROCESS_EVERY_N == 0:
        raw         = model(frame, conf=CONF_INDOOR, verbose=False)
        current_set = set()
        current_meta= []
        now         = time.time()

        for r in raw:
            for box in r.boxes:
                cls   = int(box.cls[0])
                label = model.names[cls]
                conf  = float(box.conf[0])
                if label in IGNORE_ALWAYS:          continue
                if label not in active_objects:     continue
                if conf < conf_threshold_for(label):continue
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cx     = (x1+x2)//2; box_h = y2-y1
                dirn   = get_direction(cx, FRAME_WIDTH)
                dist_m = estimate_distance(label, box_h)
                current_set.add((label, dirn))
                current_meta.append((label, dirn, dist_m, x1, y1, x2, y2, conf))

        stable      = stable_detections(current_set)
        stable_meta = [(l,d,dm,x1,y1,x2,y2,c) for (l,d,dm,x1,y1,x2,y2,c)
                       in current_meta if (l,d) in stable]

        # Approach detection
        for (label, dirn, dist_m, *_) in stable_meta:
            mv = update_approach(label, dist_m, now)
            if mv == "approaching":
                ka = f"approach_{label}"
                if now - approach_last_spoken.get(ka, 0) > APPROACH_COOLDOWN:
                    msg = f"Warning: {label} approaching, {distance_label(dist_m)} {dirn}"
                    try_speak(msg, urgent=True)
                    last_announce = msg
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]  ↗ APPROACHING "
                          f"{label}  {distance_label(dist_m)}  {dirn}")
                    approach_last_spoken[ka] = now

        # Main speech
        speech_input   = [(l,d,dm) for (l,d,dm,*_) in stable_meta]
        announcement, spoken_keys = build_speech(speech_input)
        announced_keys = set(spoken_keys) if spoken_keys else set()
        if announcement:
            urgent = any(distance_urgency(
                next((dm for l,d,dm,*_ in stable_meta if (l,d)==k), None)
            )=="critical" or k[0] in PRIORITY_HIGH for k in spoken_keys)
            try_speak(announcement, urgent=urgent)
            last_announce = announcement
            for k in spoken_keys: object_last_spoken[k] = now

        # Logging + counters
        for (label, dirn, dist_m, x1, y1, x2, y2, conf) in stable_meta:
            dlabel    = distance_label(dist_m)
            announced = (label, dirn) in announced_keys
            log_detection(label, conf, dirn, dlabel, dist_m, announced)
            object_counts[label] += 1
            add_to_heatmap(x1, y1, x2, y2)

        # Build draw list
        last_boxes = []
        for (label, dirn, dist_m, x1, y1, x2, y2, conf) in stable_meta:
            urg   = distance_urgency(dist_m)
            color = (0,0,255) if urg=="critical" else (0,80,255) if label in PRIORITY_HIGH else (0,210,100)
            last_boxes.append((x1,y1,x2,y2,label,conf,dirn,dist_m,color))

        # Free zones
        last_free_zones = get_free_zones(last_boxes, FRAME_WIDTH)
        if last_free_zones and now - free_zone_t > 8.0:
            best = last_free_zones[0]
            msg  = "Path clear ahead." if best=="centre" else f"Path clear on your {best}."
            try_speak(msg); free_zone_t = now

    # ── FPS ───────────────────────────────────
    fps_times.append(time.time())
    if len(fps_times) >= 2:
        fps_display = round(len(fps_times)/(fps_times[-1]-fps_times[0]+1e-6), 1)

    # ═══════════════════════════════════════════
    #  DRAWING (laptop window)
    # ═══════════════════════════════════════════
    annotated = frame.copy()
    third     = FRAME_WIDTH // 3

    # Zone overlays
    for zname, zx1, zx2 in [("left",0,third),("centre",third,2*third),("right",2*third,FRAME_WIDTH)]:
        free    = zname in last_free_zones
        overlay = annotated.copy()
        cv2.rectangle(overlay,(zx1,0),(zx2,FRAME_HEIGHT),(0,180,0) if free else (0,0,180),-1)
        cv2.addWeighted(overlay,0.08,annotated,0.92,0,annotated)
        cv2.putText(annotated,"CLEAR" if free else "BLOCKED",(zx1+6,20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,220,0) if free else (0,80,255),1,cv2.LINE_AA)

    # Danger zone line
    danger_y = int(FRAME_HEIGHT*0.65)
    cv2.line(annotated,(0,danger_y),(FRAME_WIDTH,danger_y),(0,0,220),1)
    cv2.putText(annotated,"~1m danger zone",(4,danger_y-4),
                cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,220),1,cv2.LINE_AA)

    # Bounding boxes
    for (x1,y1,x2,y2,label,conf,dirn,dist_m,color) in last_boxes:
        urg   = distance_urgency(dist_m)
        flash = urg=="critical" and int(time.time()*4)%2==0
        cv2.rectangle(annotated,(x1,y1),(x2,y2),color,3 if flash else 2)
        tag  = f"{label} {int(conf*100)}%  {dirn}  {distance_label(dist_m)}"
        (tw,th),_ = cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)
        ty   = max(y1-10,th+4)
        cv2.rectangle(annotated,(x1,ty-th-4),(x1+tw+4,ty),color,-1)
        cv2.putText(annotated,tag,(x1+2,ty-2),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1,cv2.LINE_AA)

    # Object counter panel
    panel_x = FRAME_WIDTH-160
    cv2.rectangle(annotated,(panel_x,0),(FRAME_WIDTH,14+16*(min(len(object_counts),8)+1)),(30,30,30),-1)
    cv2.putText(annotated,"SESSION TOTALS",(panel_x+4,12),cv2.FONT_HERSHEY_SIMPLEX,0.35,(200,200,200),1,cv2.LINE_AA)
    for i,(lbl,cnt) in enumerate(sorted(object_counts.items(),key=lambda x:-x[1])[:8]):
        cv2.putText(annotated,f"{lbl[:14]:<14} {cnt:>3}",(panel_x+4,26+i*16),
                    cv2.FONT_HERSHEY_SIMPLEX,0.33,(180,255,180),1,cv2.LINE_AA)

    # HUD bar
    mode_col = (100,200,255) if mode=="indoor" else (255,200,100)
    hud = f"FPS:{fps_display}  Frame:{frame_count}  Mode:{mode.upper()}  ESC=quit S=summary M=mode"
    cv2.rectangle(annotated,(0,FRAME_HEIGHT-18),(FRAME_WIDTH,FRAME_HEIGHT),(30,30,30),-1)
    cv2.putText(annotated,hud,(6,FRAME_HEIGHT-5),cv2.FONT_HERSHEY_SIMPLEX,0.35,mode_col,1,cv2.LINE_AA)

    cv2.imshow("Blind Assistance System", annotated)

    # ── Push frame + stats to Flask ───────────
    _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
    current_det_list = [
        {"label": label, "conf": int(conf*100), "dirn": dirn,
         "dist": distance_label(dist_m), "urgency": distance_urgency(dist_m)}
        for (x1,y1,x2,y2,label,conf,dirn,dist_m,color) in last_boxes
    ]
    with state_lock:
        shared_frame = jpg.tobytes()
        shared_stats.update({
            "fps"              : fps_display,
            "mode"             : mode,
            "frame"            : frame_count,
            "objects"          : dict(object_counts),
            "current"          : current_det_list,
            "free_zones"       : last_free_zones,
            "last_announcement": last_announce,
            "elapsed"          : round(time.time()-session_start, 1),
        })


# ═══════════════════════════════════════════════════════
#  CLEANUP
# ═══════════════════════════════════════════════════════
speech_queue.put(None)
cap.release()
cv2.destroyAllWindows()

print("\n" + "═"*60)
print("  SESSION ENDED")
print("═"*60)
print(f"  Duration : {round(time.time()-session_start,1)} s  |  Frames: {frame_count}")
for lbl,cnt in sorted(object_counts.items(), key=lambda x:-x[1]):
    print(f"      {lbl:<22} {cnt}×")
print("═"*60)

# Generate heatmap and push to browser
print("Generating heatmap...")
hm_b64 = heatmap_to_b64()
with state_lock:
    shared_stats["heatmap_b64"] = hm_b64

if hm_b64:
    print(f"Heatmap ready — open http://localhost:{WEB_PORT} and tap the Heatmap button.")
    input("Press Enter to shut down the server...")
else:
    print("No heatmap data.")
