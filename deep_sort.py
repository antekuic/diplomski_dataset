import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import argparse
import os
import numpy as np

# Definirajte mapiranje klase
class_names = [
    "Car Ferry", "Cruise Ship", "Speedboat", "Sailing Boat", "Catamaran", 
    "Dinghy", "Yacht", "Mega Yacht", "Tender", "Small Sailing Boat",
    "Passanger Catamaran", "Small boat", "Jet Ski", "Pirate Boat",
    "Small Cruise Ship", "Submarine"
]

# Postavljanje argument parsera
parser = argparse.ArgumentParser(description="YOLOv8 i DEEP SORT praćenje")
parser.add_argument('--model', type=str, required=True, help="Putanja do YOLOv8 modela")
parser.add_argument('--video', type=str, required=True, help="Putanja do video datoteke")
parser.add_argument('--output', type=str, default='output_video.mp4', help="Putanja do izlaznog video zapisa")
args = parser.parse_args()

# Provjera da li video datoteka postoji
if not os.path.exists(args.video):
    print(f"Video datoteka '{args.video}' ne postoji.")
    exit()

# Uvezi YOLO model
model = YOLO(args.model)

# Inicijaliziraj DeepSort tracker
tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)

# Otvori video stream ili video datoteku
cap = cv2.VideoCapture(args.video)

# Provjera da li je video uspješno otvoren
if not cap.isOpened():
    print(f"Neuspješno otvaranje video datoteke '{args.video}'.")
    exit()

# Postavljanje video writera za MP4 format
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Za MP4 format
out = cv2.VideoWriter(args.output, fourcc, 30.0, (frame_width, frame_height))

# Spremi klasu za svaki track
track_class_map = {}

def bbox_iou(box1, box2):
    """Izračunaj IoU između dva bounding boxa."""
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    xi1 = max(x1, x1b)
    yi1 = max(y1, y1b)
    xi2 = min(x2, x2b)
    yi2 = min(y2, y2b)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2b - x1b) * (y2b - y1b)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

while True:
    ret, frame = cap.read()
    if not ret:
        print("Završeno čitanje okvira iz videa ili greška pri čitanju.")
        break

    if frame is None or frame.size == 0:
        print("Učitani okvir je prazan ili neispravan")
        continue
    
    # Detekcija objekata pomoću YOLOv8
    results = model(frame)
    
    # Ekstrahiraj bounding boxove, konfidence i klase
    detections = []
    yolo_detections = []
    
    for r in results[0].boxes.data.cpu().numpy():
        if r.size == 6:
            x1, y1, x2, y2, conf, cls = r
        elif r.size == 5:
            x1, y1, x2, y2, conf = r
            cls = -1  # Ako nema klase, postavite na -1
        else:
            print("Neispravan format detekcija.")
            continue
        
        detections.append(([x1, y1, x2-x1, y2-y1], conf, int(cls) if cls != -1 else -1))
        yolo_detections.append([x1, y1, x2, y2])

    # Praćenje pomoću DEEP SORT-a
    tracks = tracker.update_tracks(detections, frame=frame)

    # Poveži klase s praćenim objektima na temelju preklapanja bounding boxova
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        track_bbox = track.to_ltrb()

        if track_id not in track_class_map:
            best_iou = 0
            best_class = -1
            for i, yolo_bbox in enumerate(yolo_detections):
                iou = bbox_iou(track_bbox, yolo_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_class = detections[i][2]
            track_class_map[track_id] = class_names[best_class] if best_class != -1 else "Unknown"
        
        class_name = track_class_map[track_id]
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Nacrtaj bounding box i naziv klase
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Spremi frame u video
    if frame.size > 0:
        out.write(frame)
    else:
        print("Frame je neispravan za pisanje.")

# Oslobađanje resursa
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video spremljen na '{args.output}'")