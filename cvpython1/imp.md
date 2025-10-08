# virtualenv बनाओ (optional)

python -m venv venv
venv\Scripts\activate # Windows

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu # अगर GPU नहीं है; GPU वाले users अपने CUDA wheel install करें
pip install opencv-python numpy pandas pyttsx3 pillow
pip install yolov5 # या pip install ultralytics (नीचे code yolov5 hub से करता है)

# Note: अगर 'yolov5' package नहीं चलता तो हम torch.hub का इस्तेमाल करते हैं (code में handle है)

camera_counter_report.py

"""
camera_counter_report.py
Run: python camera_counter_report.py
This script:

- opens webcam (0) or a video file (set VIDEO_SOURCE)
- uses YOLOv5 (torch.hub) for object detection (person, car, etc.)
- attempts gender detection on detected faces if gender model files present
- every REPORT_INTERVAL seconds speaks & prints a summary and saves to CSV
  """

import os
import time
import csv
import argparse
from collections import Counter, deque
import numpy as np
import cv2
import torch
import pyttsx3
import pandas as pd
from datetime import datetime
from PIL import Image

# ------------- User options -------------

VIDEO_SOURCE = 0 # 0 for webcam, or "path/to/video.mp4"
REPORT_INTERVAL = 10 # seconds between spoken reports
OUTPUT_CSV = "camera_reports.csv"
CONF_THRESH = 0.4 # detection confidence threshold
USE_GENDER = True # try gender detection if models present

# paths for gender model (optional)

GENDER_PROTO = "models/deploy_gender.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"
GENDER_LABELS = ["Male", "Female"]

# ----------------------------------------

# Initialize text-to-speech (pyttsx3 works offline)

tts = pyttsx3.init()
tts.setProperty("rate", 150)

# helper: speak text

def speak(text):
print("[VOICE] ", text)
try:
tts.say(text)
tts.runAndWait()
except Exception as e:
print("TTS error:", e)

# load YOLOv5 (try torch.hub, fallback to ultralytics if installed)

def load_yolov5_model():
try: # this will download the model if not present (internet needed first time)
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
print("Loaded YOLOv5s via torch.hub")
return model
except Exception as e:
print("torch.hub load failed:", e) # try ultralytics package (YOLOv8) if installed
try:
from ultralytics import YOLO
model = YOLO("yolov8n.pt") # may download
print("Loaded YOLOv8n via ultralytics")
return model
except Exception as e2:
raise RuntimeError("Failed to load YOLO model. Install internet and try again.") from e2

# gender detector (optional)

class GenderDetector:
def **init**(self, proto_path, model_path):
self.ready = False
if os.path.exists(proto_path) and os.path.exists(model_path):
self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
self.ready = True
print("Gender model loaded.")
else:
print("Gender model files not found. Gender detection disabled.")
self.ready = False

    def predict(self, face_img):
        """
        face_img: BGR image (numpy) of face crop
        returns: "Male" or "Female" or None
        """
        if not self.ready:
            return None
        try:
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227,227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            self.net.setInput(blob)
            preds = self.net.forward()
            idx = preds[0].argmax()
            return GENDER_LABELS[idx]
        except Exception as e:
            print("Gender predict error:", e)
            return None

# face detector for cropping faces (using OpenCV's DNN face detector or Haar cascade fallback)

class FaceCropper:
def **init**(self): # try OpenCV's DNN face detector model (if available in opencv installation)
self.face_net = None
try: # default opencv comes with a deep learning face detector? handle gracefully # We'll fallback to Haarcascade if dnn not available
self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
except Exception:
self.face_cascade = None

    def crop(self, frame, bbox):
        """
        frame: BGR image
        bbox: [x1,y1,x2,y2] bounding box of person/object
        returns: face_img or None
        """
        x1,y1,x2,y2 = bbox
        h,w = frame.shape[:2]
        # expand a bit around person box
        pad = 10
        sx = max(0, x1-pad); sy = max(0, y1-pad)
        ex = min(w, x2+pad); ey = min(h, y2+pad)
        roi = frame[sy:ey, sx:ex]
        if roi.size == 0:
            return None
        # detect faces inside roi
        if self.face_cascade:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                # take first face
                (fx,fy,fw,fh) = faces[0]
                face_img = roi[fy:fy+fh, fx:fx+fw]
                return face_img
        return None

def main(): # load detection model
model = load_yolov5_model()

    # determine class names (for yolov5 via torch.hub)
    try:
        names = model.names
    except:
        # ultralytics YOLO object has 'model.names' maybe
        names = model.model.names if hasattr(model, "model") else None

    print("Classes available:", list(names.items())[:10] if isinstance(names, dict) else names[:10])

    # load optional gender detector
    gender_detector = GenderDetector(GENDER_PROTO, GENDER_MODEL) if USE_GENDER else None
    face_cropper = FaceCropper()

    # Open video capture
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Error opening video source:", VIDEO_SOURCE)
        return

    last_report_time = time.time()
    accum_counts = Counter()
    # keep last few frames' detections for smoothing
    recent_counts = deque(maxlen=5)

    # prepare CSV
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "persons", "cars", "other_objects", "females_detected"])

    frame_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame received — exiting.")
                break
            frame_id += 1
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # YOLO detection
            # For torch.hub yolov5, model(img) returns results
            results = model(img)  # works for yolov5 and ultralytics
            # parse results
            detections = []
            try:
                # yolov5: results.xyxy[0] (tensor) ; ultralytics: results[0].boxes
                if hasattr(results, "xyxy"):  # older
                    dets = results.xyxy[0].cpu().numpy()
                    # each det: x1,y1,x2,y2,conf,class
                    for d in dets:
                        x1,y1,x2,y2,conf,cls = d
                        if conf < CONF_THRESH:
                            continue
                        cls = int(cls)
                        label = names[cls] if isinstance(names, dict) else names[cls]
                        detections.append((label, float(conf), (int(x1),int(y1),int(x2),int(y2))))
                else:
                    # ultralytics YOLOv8 result object
                    res0 = results[0]
                    boxes = res0.boxes
                    for box in boxes:
                        conf = float(box.conf[0])
                        if conf < CONF_THRESH: continue
                        cls = int(box.cls[0])
                        label = names[cls] if isinstance(names, dict) else names[cls]
                        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                        detections.append((label, conf, (x1,y1,x2,y2)))
            except Exception as e:
                print("Parse results error:", e)
                continue

            # count categories we care about
            counter = Counter()
            female_count = 0
            for label, conf, bbox in detections:
                # normalize label names (some models call 'car' 'truck' etc.)
                if label.lower() in ["person"]:
                    counter["person"] += 1
                    # try gender detection
                    if gender_detector and gender_detector.ready:
                        face = face_cropper.crop(frame, bbox)
                        if face is not None:
                            g = gender_detector.predict(face)
                            if g and g.lower() == "female":
                                female_count += 1
                elif label.lower() in ["car", "truck", "bus", "motorbike", "motorcycle"]:
                    counter["car"] += 1
                else:
                    counter["other"] += 1

                # draw for visualization
                x1,y1,x2,y2 = bbox
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(10,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # update accumulators
            accum_counts.update(counter)
            recent_counts.append(counter)

            # show counts overlay
            txt = f"Persons:{counter.get('person',0)} Cars:{counter.get('car',0)} Others:{counter.get('other',0)} Females:{female_count}"
            cv2.putText(frame, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            cv2.imshow("Camera Counter", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            # periodic report
            now = time.time()
            if now - last_report_time >= REPORT_INTERVAL:
                # aggregate over recent_counts
                agg = Counter()
                for c in recent_counts:
                    agg.update(c)
                persons = agg.get("person", 0)
                cars = agg.get("car", 0)
                others = agg.get("other", 0)
                females = female_count  # last frame female_count; you can aggregate per-frame similarly

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                report_text = f"Report at {timestamp}: {persons} persons, {cars} vehicles, {others} other objects. Detected approximately {females} females."
                # speak & print
                speak(report_text)
                print(report_text)

                # save to CSV
                with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, persons, cars, others, females])

                last_report_time = now

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Done. CSV saved to", OUTPUT_CSV)

if **name** == "**main**":
main()

pip install --upgrade pip
pip install numpy opencv-python torch torchvision torchaudio pyttsx3 pandas pillow
