import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights" / "best.pt"
THRESHOLDS = {"smoke": 0.30, "fire": 0.35}
COLOURS = {"smoke": (160, 160, 160), "fire": (0, 0, 255)}
WINDOW = 10
MIN_HITS = 5

st.set_page_config(page_title="Fire and Smoke Detector", layout="wide")


@st.cache_resource
def load_model():
    return YOLO(str(WEIGHTS)), ThreadPoolExecutor(max_workers=1)


model, worker = load_model()


def detect(frame):
    r = worker.submit(model.predict, frame, conf=0.10, verbose=False).result()[0]
    found = []
    for (x1, y1, x2, y2), score, c in zip(r.boxes.xyxy.int().tolist(), r.boxes.conf.tolist(), r.boxes.cls.int().tolist()):
        name = model.names[c]
        if score < THRESHOLDS[name]:
            continue
        found.append((name, score))
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOURS[name], 3)
        cv2.putText(frame, f"{name} {score:.2f}", (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOURS[name], 2)
    return found, r.speed["inference"]


st.title("Fire and Smoke Detector")
st.caption("YOLOv8n fine-tuned on the D-Fire dataset. On video and live camera, the alert turns on "
           f"when smoke or fire is seen in at least {MIN_HITS} of the last {WINDOW} frames.")

mode = st.segmented_control("Mode", ["Photo", "Video", "Live camera"], default="Photo", required=True, label_visibility="collapsed")

if mode == "Photo":
    upload = st.file_uploader("Photo", type=["jpg", "jpeg", "png"])
    if upload is not None:
        data = upload.read()
    else:
        samples = sorted((Path(__file__).parent / "samples").glob("*.jpg"))
        names = [p.stem for p in samples]
        start = names.index(st.query_params["sample"]) if st.query_params.get("sample") in names else 0
        sample = st.selectbox("Or try a D-Fire test photo", samples, index=start, format_func=lambda p: p.stem.replace("_", " "))
        data = sample.read_bytes()

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        st.error("That file could not be read as an image.")
        st.stop()
    found, ms = detect(image)

    left, right = st.columns([3, 2])
    left.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch", alt="Photo with detection boxes")
    with right:
        if found:
            st.error("ALERT: " + " + ".join(sorted({n for n, _ in found})).upper())
        else:
            st.success("No fire or smoke found")
        st.subheader("Detections")
        if found:
            for name, score in sorted(found, key=lambda f: -f[1]):
                st.progress(score, text=f"{name}  {score:.2f}")
        else:
            st.write("None above the thresholds.")
        st.caption(f"Thresholds: smoke {THRESHOLDS['smoke']:.2f}, fire {THRESHOLDS['fire']:.2f}. Inference: {ms:.1f} ms.")
    st.stop()

if mode == "Video":
    upload = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"])
    if upload is None or not st.button("Analyse video", type="primary"):
        st.stop()
    temp = tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix, delete=False)
    temp.write(upload.getvalue())
    temp.close()
    source = temp.name
else:
    camera = st.text_input("Camera", "0", help="0 = built-in webcam, 1 = next camera, or an rtsp:// or http:// address")
    if not st.toggle("Start camera"):
        st.stop()
    source = int(camera) if camera.isdigit() else camera

left, right = st.columns([3, 2])
picture = left.empty()
status = right.empty()
counter = right.empty()
progress = right.empty()
log = right.empty()

cap = cv2.VideoCapture(source)
try:
    if not cap.isOpened():
        st.error("Could not open that source. Check that no other app is using the camera.")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    history = deque(maxlen=WINDOW)
    events = []
    seen = set()
    alert_on = False
    n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        found, ms = detect(frame)
        hits = {name for name, _ in found}
        history.append(bool(hits))

        now_on = sum(history) >= MIN_HITS
        if now_on and not alert_on:
            seen = set()
            events.append([n / fps, n / fps, seen])
        if now_on:
            seen |= hits
            events[-1][1] = n / fps
        alert_on = now_on

        # Sending every frame to the browser is slow, so a video shows one in three.
        if mode == "Live camera" or n % 3 == 0:
            picture.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch", alt="Current frame with detection boxes")
            if alert_on:
                status.error("ALERT: " + " + ".join(sorted(seen)).upper())
            else:
                status.success("All clear")
            counter.caption(f"Hits {sum(history)} of the last {len(history)} frames. Inference: {ms:.1f} ms.")
            if mode == "Video" and total > 0:
                progress.progress(min(n / total, 1.0), text=f"Frame {n} of {total}")
        n += 1
finally:
    cap.release()
    if mode == "Video":
        Path(source).unlink(missing_ok=True)

progress.empty()
if mode == "Video":
    if events:
        lines = [f"- {start:.1f}s to {end:.1f}s: {' + '.join(sorted(names)).upper()}" for start, end, names in events]
        log.markdown(f"**{len(events)} alert(s) in {n} frames**\n\n" + "\n".join(lines))
    else:
        log.markdown(f"**No alerts in {n} frames**")
else:
    st.info("The stream ended.")
