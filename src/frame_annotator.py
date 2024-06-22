import ultralytics
import numpy as np
import json
import os
from PIL import Image
import cv2
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np

if __name__ == "__main__":
    HOME = os.getcwd()
    print(HOME)
    label_annotator = sv.LabelAnnotator(color=sv.Color(r=39, g=85, b=156),text_padding=5)
    tracker = sv.ByteTrack()
    player_model = YOLO("./yolo_weights/best.pt")
    frame=Image.open('./raw_images/img_raw_1.png')
    results = player_model(frame, imgsz=1280,conf=0.8)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections).with_nms(threshold=0.1)
    labels=[]
    for j, detection in enumerate(detections):
            label=f"#{detection[4]}"
            labels.append(label)

    annotated_frame = label_annotator.annotate(
    scene=frame.copy(), detections=detections, labels=labels)
    annotated_frame.save("./images/img_ann_1.png")
    print(detections)