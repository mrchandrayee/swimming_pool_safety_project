import argparse
import csv
import json
from pathlib import Path
import time
import cv2
from ultralytics import YOLO
from .tracker import CentroidTracker
from .zones import polygon_pixels, classify_point, foot_point

def build_parser():
    p = argparse.ArgumentParser(description="Swimming pool person tracking and safety-zone alerts")
    p.add_argument("--source", required=True, help="Video file or webcam index, e.g. 0")
    p.add_argument("--config", default="config.json")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--output", default="runs/pool_safety")
    p.add_argument("--show", action="store_true")
    p.add_argument("--device", default=None, help="Optional Ultralytics device: cpu, 0, etc.")
    return p

def run(args):
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(outdir/"annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    log_path = outdir/"events.csv"
    model = YOLO(args.model)
    tracker = CentroidTracker(max_distance=max(50, int(width*0.12)),
                              max_lost=float(cfg.get("lost_after_seconds",2.0)))
    polygon = polygon_pixels(cfg["pool_polygon"], width, height)
    near_distance = float(cfg.get("near_distance_px",65))
    water_limit = float(cfg.get("water_alert_seconds",8.0))
    conf = float(cfg.get("confidence",0.35))
    imgsz = int(cfg.get("imgsz",640))
    show_trails = bool(cfg.get("show_trails",True))
    frame_no = 0
    last_event = {}

    with log_path.open("w", newline="", encoding="utf-8") as f:
        csvlog = csv.writer(f)
        csvlog.writerow(["time_seconds","track_id","event","zone","duration_seconds"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            now = frame_no / fps
            pred = model.predict(frame, conf=conf, imgsz=imgsz, classes=[0],
                                 device=args.device, verbose=False)[0]
            boxes = []
            if pred.boxes is not None:
                for b in pred.boxes:
                    boxes.append(tuple(map(int,b.xyxy[0].tolist())))
            tracks, lost_tracks = tracker.update(boxes, now)

            cv2.polylines(frame,[polygon],True,(255,170,0),3)
            overlay = frame.copy()
            cv2.fillPoly(overlay,[polygon],(255,120,0))
            frame = cv2.addWeighted(overlay,0.12,frame,0.88,0)
            cv2.putText(frame,"SWIMMING POOL",(15,30),cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,(0,255,0),2)

            for tr in tracks:
                pt = foot_point(tr.box)
                zone = classify_point(pt,polygon,near_distance)
                tr.zone = zone
                if zone == "IN WATER":
                    if tr.water_since is None:
                        tr.water_since = now
                    tr.was_in_water = True
                else:
                    tr.water_since = None
                duration = now-tr.water_since if tr.water_since is not None else 0.0
                alert = zone == "IN WATER" and duration >= water_limit
                color = (0,0,255) if alert else ((0,190,255) if zone=="NEAR" else (0,220,0))
                x1,y1,x2,y2 = tr.box
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                label = f"ID {tr.track_id} | {zone}"
                if alert:
                    label += f" | ALERT {duration:.1f}s"
                cv2.putText(frame,label,(x1,max(18,y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.48,color,2)
                cv2.circle(frame,pt,4,color,-1)
                if show_trails and len(tr.trail)>1:
                    import numpy as np
                    cv2.polylines(frame,[np.array(tr.trail,dtype="int32")],
                                  False,color,2)
                if alert:
                    key = (tr.track_id,"WATER_DURATION")
                    if now-last_event.get(key,-999) >= 2.0:
                        csvlog.writerow([round(now,2),tr.track_id,"WATER_DURATION",
                                         zone,round(duration,2)])
                        last_event[key] = now

            for tr in lost_tracks:
                if tr.was_in_water:
                    cv2.putText(frame,f"ALERT: ID {tr.track_id} LOST",
                        (12,60),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)
                    csvlog.writerow([round(now,2),tr.track_id,"LOST_AFTER_WATER",
                                     "LOST",""])
            writer.write(frame)
            if args.show:
                cv2.imshow("Swimming Pool Safety Monitoring",frame)
                if cv2.waitKey(1)&0xFF == ord("q"):
                    break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Annotated video: {outdir/'annotated.mp4'}")
    print(f"Event log: {log_path}")
