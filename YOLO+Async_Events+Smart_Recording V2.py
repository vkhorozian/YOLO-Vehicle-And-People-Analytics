"""
HEAVILY DOCUMENTED VERSION

This application acts as a lightweight AI-powered NVR:

1. Connects to an RTSP camera stream.
2. Runs YOLO object detection/tracking using ByteTrack.
3. Tracks vehicles and people.
4. Counts vehicles crossing virtual barriers.
5. Counts people crossing the center barrier.
6. Detects motion based on tracked object movement.
7. Records video clips before and after motion events.
8. Saves MP4 video and JSON metadata.
9. Displays a live operator view.

The comments in this file are intended to explain WHY
the code exists, not just WHAT each line does.
"""


# -----------------------------------------------------------------
# NOTE FOR FUTURE DEVELOPMENT
#
# Most of the application's logic depends on ByteTrack IDs.
# If an object loses its tracking ID and receives a new ID,
# counting and motion logic may behave differently.
#
# When debugging counting issues, always verify:
#   - Object ID stability
#   - Detection confidence
#   - Tracking persistence
#   - Virtual barrier placement
# -----------------------------------------------------------------
from ultralytics import YOLO
import cv2
import time
import json
import threading
import queue
from collections import deque
from pathlib import Path
import sys
import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda build:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")

# =============================
# CONFIG
# =============================
RTSP_URL = "rtsp://0.0.0.0/stream2" #add rtsp stream for camera here. 
MODEL_PATH = "yolov8m.pt" # choose model based on the power of your machine "yolov8n.pt" "yolov8s.pt" "yolov8m.pt" "yolov8l.pt"
RECONNECT_DELAY_SEC = 5

VEHICLE_CLASSES = [2, 3, 5, 7]
PERSON_CLASS = [0]

WIDTH, HEIGHT = 1280, 720
AI_WIDTH, AI_HEIGHT = 960, 540
SCALE_X = WIDTH / AI_WIDTH
SCALE_Y = HEIGHT / AI_HEIGHT
YOLO_EVERY_N_FRAMES = 1
YOLO_CONFIDENCE = 0.55
MOTION_THRESHOLD = 5
VEHICLE_MOTION_THRESHOLD = 12
SHOW_DEBUG_BOXES = True
DEBUG_STILL_FRAMES = 3

# Three vertical barriers (left → right): A, B, C
LINE_A, LINE_B, LINE_C = int(WIDTH * 0.3), int(WIDTH * 0.5), int(WIDTH * 0.7)
PERSON_COUNT_LINE = LINE_B  # People: single crossing at middle barrier

PRE_BUFFER_SEC = 3
POST_BUFFER_SEC = 3

# =============================
# OUTPUT
# =============================
BASE_DIR = Path(__file__).parent
EVENT_DIR_CARS = BASE_DIR / "Events_Cars"
EVENT_DIR_PEOPLE = BASE_DIR / "Events_People"
EVENT_DIR_CARS.mkdir(exist_ok=True)
EVENT_DIR_PEOPLE.mkdir(exist_ok=True)

# =============================
# INIT
# =============================
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names

cap = cv2.VideoCapture(RTSP_URL)

def read_camera_fps(capture):
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 10
    return int(fps)

FPS_ESTIMATE = read_camera_fps(cap)

cv2.namedWindow("AI NVR", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AI NVR", WIDTH, HEIGHT)

# =============================
# STATE
# =============================
vehicle_count_lr = 0
vehicle_count_rl = 0
person_count = 0
counted_vehicle_lr = set()
counted_vehicle_rl = set()
counted_person_ids = set()

track_prev_cx = {}
track_state_lr = {}
track_state_rl = {}
track_prev_pos = {}
track_still_frames = {}

frame_buffer = deque(maxlen=PRE_BUFFER_SEC * FPS_ESTIMATE)

event_queue = queue.Queue()

events = {
    "cars": {"active": False, "frames": [], "end_time": 0},
    "people": {"active": False, "frames": [], "end_time": 0},
}
event_id = 0
frame_counter = 0
last_results = None

# =============================
# HELPERS
# =============================
def scale_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return (
        x1 * SCALE_X,
        y1 * SCALE_Y,
        x2 * SCALE_X,
        y2 * SCALE_Y,
    )

def reset_frame_buffer():
    global frame_buffer
    frame_buffer = deque(maxlen=PRE_BUFFER_SEC * FPS_ESTIMATE)

def reconnect_camera():
    global cap, FPS_ESTIMATE

    print("\n[WARNING] Stream lost. Reconnecting...")
    cap.release()

    while True:
        time.sleep(RECONNECT_DELAY_SEC)
        cap = cv2.VideoCapture(RTSP_URL)
        if cap.isOpened():
            FPS_ESTIMATE = read_camera_fps(cap)
            reset_frame_buffer()
            print(f"[INFO] Reconnected. FPS: {FPS_ESTIMATE}")
            return

def get_count_snapshot():
    return {
        "vehicle_count_lr": vehicle_count_lr,
        "vehicle_count_rl": vehicle_count_rl,
        "person_count": person_count,
    }

def draw_timestamp(frame):
    timestamp_text = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        frame,
        timestamp_text,
        (20, HEIGHT - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

def draw_debug_boxes(frame, results):
    if not SHOW_DEBUG_BOXES or results is None or results[0].boxes is None:
        return

    for box in results[0].boxes:
        if box.id is None:
            continue

        tid = int(box.id[0])
        cls_id = int(box.cls[0])

        # Hide boxes on parked/stationary vehicles to reduce visual noise
        if (
            cls_id in VEHICLE_CLASSES
            and track_still_frames.get(tid, 0) >= DEBUG_STILL_FRAMES
        ):
            continue

        x1, y1, x2, y2 = scale_xyxy(box.xyxy[0].tolist())
        conf = float(box.conf[0])
        label = CLASS_NAMES.get(cls_id, str(cls_id))

        if cls_id in VEHICLE_CLASSES:
            color = (0, 255, 0)
        elif cls_id in PERSON_CLASS:
            color = (0, 200, 255)
        else:
            color = (200, 200, 200)

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            2,
        )

        cv2.putText(
            frame,
            f"ID:{tid} {label} {conf:.2f}",
            (int(x1), max(20, int(y1) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

# =============================
# PROGRESS BAR
# =============================
def print_progress(remaining, total, label="REC"):
    bar_len = 20
    if total <= 0:
        return

    progress = 1 - (remaining / total)
    filled = int(progress * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    sys.stdout.write(f"\r[{label}] {remaining:.1f}s remaining {bar}")
    sys.stdout.flush()

# =============================
# SAVE EVENT
# =============================
def save_event(frames, event_type, output_dir, counts_snapshot):
    global event_id

    if not frames:
        return

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    folder = output_dir / f"event_{timestamp}_{event_type}"
    folder.mkdir(exist_ok=True)

    video_path = folder / "clip.mp4"
    meta_path = folder / "meta.json"

    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS_ESTIMATE,
        (w, h),
    )

    for f in frames:
        out.write(f)

    out.release()

    meta = {
        "id": event_id,
        "time": timestamp,
        "type": event_type,
        "frames": len(frames),
        "duration_seconds": len(frames) / FPS_ESTIMATE,
        "vehicle_count_lr": counts_snapshot["vehicle_count_lr"],
        "vehicle_count_rl": counts_snapshot["vehicle_count_rl"],
        "person_count": counts_snapshot["person_count"],
        "camera_source": RTSP_URL,
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)

    print(f"\n[SAVED] {folder.name}")
    event_id += 1

# =============================
# ASYNC WRITER
# =============================
def writer():
    while True:
        item = event_queue.get()
        if item is None:
            break

        frames, typ, output_dir, counts_snapshot = item
        save_event(frames, typ, output_dir, counts_snapshot)
        event_queue.task_done()

threading.Thread(target=writer, daemon=True).start()

# =============================
# MOTION-BASED EVENT DETECTION
# =============================
def detect_motion_by_class(results):
    if results[0].boxes is None:
        return False, False

    vehicle_motion = False
    person_motion = False

    for box in results[0].boxes:
        if box.id is None:
            continue

        tid = int(box.id[0])
        cls_id = int(box.cls[0])

        x1, y1, x2, y2 = scale_xyxy(box.xyxy[0].tolist())
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        prev = track_prev_pos.get(tid)

        if prev is not None:
            dx = abs(cx - prev[0])
            dy = abs(cy - prev[1])
            dist = ((dx ** 2) + (dy ** 2)) ** 0.5

            motion_limit = (
                VEHICLE_MOTION_THRESHOLD
                if cls_id in VEHICLE_CLASSES
                else MOTION_THRESHOLD
            )
            if dist > motion_limit:
                if cls_id in VEHICLE_CLASSES:
                    vehicle_motion = True
                elif cls_id in PERSON_CLASS:
                    person_motion = True
                track_still_frames[tid] = 0
            else:
                track_still_frames[tid] = track_still_frames.get(tid, 0) + 1

        track_prev_pos[tid] = (cx, cy)

    return vehicle_motion, person_motion

# =============================
# ZONE COUNTERS (A→B→C and C→B→A)
# =============================
def segment_crosses_right(prev_cx, cx, line_x):
    return prev_cx is not None and prev_cx < cx and prev_cx <= line_x <= cx

def segment_crosses_left(prev_cx, cx, line_x):
    return prev_cx is not None and prev_cx > cx and cx <= line_x <= prev_cx

def update_zone_state_lr(prev_cx, cx, state):
    if prev_cx is None:
        return state
    if state == 0 and segment_crosses_right(prev_cx, cx, LINE_A):
        state = 1
    if state == 1 and segment_crosses_right(prev_cx, cx, LINE_B):
        state = 2
    if state == 2 and segment_crosses_right(prev_cx, cx, LINE_C):
        state = 3
    return state

def update_zone_state_rl(prev_cx, cx, state):
    if prev_cx is None:
        return state
    if state == 0 and segment_crosses_left(prev_cx, cx, LINE_C):
        state = 1
    if state == 1 and segment_crosses_left(prev_cx, cx, LINE_B):
        state = 2
    if state == 2 and segment_crosses_left(prev_cx, cx, LINE_A):
        state = 3
    return state

def crosses_line_either_direction(prev_cx, cx, line_x):
    return (
        segment_crosses_right(prev_cx, cx, line_x)
        or segment_crosses_left(prev_cx, cx, line_x)
    )

def start_event(key):
    ev = events[key]
    label = key.upper()
    print(f"\n[EVENT START - {label}]")
    ev["active"] = True
    ev["frames"] = list(frame_buffer)

def finish_event(key, now, output_dir):
    ev = events[key]
    label = key.upper()
    print(f"\n[EVENT END - {label}]")
    event_queue.put((
        ev["frames"].copy(),
        "motion",
        output_dir,
        get_count_snapshot(),
    ))
    ev["active"] = False
    ev["frames"] = []

def update_event_triggers(car_motion, person_motion, now):
    if car_motion and not events["cars"]["active"]:
        start_event("cars")
        events["cars"]["end_time"] = now + POST_BUFFER_SEC
    elif events["cars"]["active"] and car_motion:
        events["cars"]["end_time"] = now + POST_BUFFER_SEC

    if person_motion and not events["people"]["active"]:
        start_event("people")
        events["people"]["end_time"] = now + POST_BUFFER_SEC
    elif events["people"]["active"] and person_motion:
        events["people"]["end_time"] = now + POST_BUFFER_SEC

def record_active_events(display_frame, now):
    for key, output_dir in (
        ("cars", EVENT_DIR_CARS),
        ("people", EVENT_DIR_PEOPLE),
    ):
        ev = events[key]
        if not ev["active"]:
            continue

        ev["frames"].append(display_frame.copy())

        remaining = max(0, ev["end_time"] - now)
        print_progress(remaining, POST_BUFFER_SEC, key.upper())

        if now >= ev["end_time"]:
            finish_event(key, now, output_dir)

# =============================
# MAIN LOOP
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        reconnect_camera()
        continue

    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    frame_counter += 1
    results = None

    if frame_counter % YOLO_EVERY_N_FRAMES == 0:
        ai_frame = cv2.resize(frame, (AI_WIDTH, AI_HEIGHT))

        start = time.time()
        results = model.track(
            ai_frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES + PERSON_CLASS,
            conf=YOLO_CONFIDENCE,
            verbose=False,
        )
        elapsed = time.time() - start
        print(f"YOLO FPS: {1 / elapsed:.1f}")
        last_results = results

#        for box in results[0].boxes:
#            print(
#                "Class:", int(box.cls[0]),
#                "Conf:", round(float(box.conf[0]), 2)
#            )

    # =============================
    # COUNTERS — vehicles: 3-zone | people: single line at B
    # Only update on YOLO frames; track state must persist between them.
    # =============================
    active_ids = set()
    if results is not None and results[0].boxes is not None:
        for box in results[0].boxes:
            if box.id is None:
                continue

            tid = int(box.id[0])
            active_ids.add(tid)

            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES and cls_id not in PERSON_CLASS:
                continue

            x1, y1, x2, y2 = scale_xyxy(box.xyxy[0].tolist())
            cx = (x1 + x2) / 2

            prev_cx = track_prev_cx.get(tid)
            track_prev_cx[tid] = cx

            if cls_id in VEHICLE_CLASSES:
                state_lr = update_zone_state_lr(
                    prev_cx, cx, track_state_lr.get(tid, 0)
                )
                state_rl = update_zone_state_rl(
                    prev_cx, cx, track_state_rl.get(tid, 0)
                )
                track_state_lr[tid] = state_lr
                track_state_rl[tid] = state_rl

                if state_lr == 3 and tid not in counted_vehicle_lr:
                    counted_vehicle_lr.add(tid)
                    vehicle_count_lr += 1
                if state_rl == 3 and tid not in counted_vehicle_rl:
                    counted_vehicle_rl.add(tid)
                    vehicle_count_rl += 1
            elif cls_id in PERSON_CLASS:
                if (
                    tid not in counted_person_ids
                    and crosses_line_either_direction(prev_cx, cx, PERSON_COUNT_LINE)
                ):
                    counted_person_ids.add(tid)
                    person_count += 1

        stale_ids = (
            set(track_prev_cx)
            | set(track_state_lr)
            | set(track_state_rl)
            | set(track_prev_pos)
            | set(track_still_frames)
        )
        stale_ids -= active_ids
        for tid in stale_ids:
            track_prev_cx.pop(tid, None)
            track_state_lr.pop(tid, None)
            track_state_rl.pop(tid, None)
            track_prev_pos.pop(tid, None)
            track_still_frames.pop(tid, None)

    # =============================
    # EVENT TRIGGERS (motion from YOLO frames only)
    # =============================
    if results is not None:
        car_motion, person_motion = detect_motion_by_class(results)
    else:
        car_motion, person_motion = False, False

    now = time.time()
    update_event_triggers(car_motion, person_motion, now)

    # =============================
    # UI + OVERLAYS (applied before buffer + recording)
    # =============================
    draw_debug_boxes(frame, last_results)

    cv2.line(frame, (LINE_A, 0), (LINE_A, HEIGHT), (0, 255, 0), 2)
    cv2.line(frame, (LINE_B, 0), (LINE_B, HEIGHT), (255, 200, 0), 3)
    cv2.line(frame, (LINE_C, 0), (LINE_C, HEIGHT), (0, 255, 0), 2)

    cv2.putText(frame, "A", (LINE_A - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "B", (LINE_B - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    cv2.putText(frame, "C", (LINE_C - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    total_vehicles = vehicle_count_lr + vehicle_count_rl
    cv2.putText(frame, f"Vehicles L-R: {vehicle_count_lr}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Vehicles R-L: {vehicle_count_rl}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Total: {total_vehicles}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"People: {person_count}", (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

    draw_timestamp(frame)

    frame_buffer.append(frame.copy())
    record_active_events(frame, now)

    cv2.imshow("AI NVR", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =============================
# CLEANUP
# =============================
for key, ev in events.items():
    if ev["active"] and ev["frames"]:
        output_dir = EVENT_DIR_CARS if key == "cars" else EVENT_DIR_PEOPLE
        event_queue.put((
            ev["frames"].copy(),
            "motion",
            output_dir,
            get_count_snapshot(),
        ))

cap.release()
cv2.destroyAllWindows()
event_queue.join()
event_queue.put(None)
