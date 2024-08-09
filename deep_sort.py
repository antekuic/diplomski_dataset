import cv2
import argparse
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Postavljanje argument parsera
parser = argparse.ArgumentParser(description="YOLOv8 i DEEP SORT praćenje")
parser.add_argument('--model', type=str, required=True, help="Putanja do YOLOv8 modela")
parser.add_argument('--video', type=str, required=True, help="Putanja do video datoteke")
parser.add_argument('--output', type=str, default='output_video.avi', help="Putanja do izlaznog video zapisa")
args = parser.parse_args()

# Uvezi YOLO model
model = YOLO(args.model)

# Inicijaliziraj DeepSort tracker
tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)

# Otvori video stream ili video datoteku
cap = cv2.VideoCapture(args.video)

# Postavljanje video writera
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(args.output, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detekcija objekata pomoću YOLOv8
    results = model(frame)
    
    # Ekstrahiraj bounding boxove, konfidence i klase
    detections = []
    for r in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = r
        detections.append(([x1, y1, x2-x1, y2-y1], conf, int(cls)))

    # Praćenje pomoću DEEP SORT-a
    tracks = tracker.update_tracks(detections, frame=frame)
    
    # Nacrtaj rezultate na frame
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Spremi frame u video
    out.write(frame)

# Oslobađanje resursa
cap.release()
out.release()
cv2.destroyAllWindows()