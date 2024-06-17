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
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

model = YOLO("./yolo_weights/best.pt")

class Detections:
    def __init__(self, xyxy, mask, confidence, class_id, tracker_id, data):
        self.xyxy = xyxy
        self.mask = mask
        self.confidence = confidence
        self.class_id = class_id
        self.tracker_id = tracker_id
        self.data = data

def extract_colors(image, xyxy):
    colors = []
    for box in xyxy:
        x1, y1, x2, y2 = map(int, box)
        cropped_image = image[y1:y2, x1:x2]
        mean_color = cv2.mean(cropped_image)[:3]  # Ottieni il colore medio (BGR)
        colors.append(mean_color)
    return np.array(colors)

def create_clusters(detections, image):
    colors = extract_colors(image, detections.xyxy)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(colors)
    labels = kmeans.labels_
    team_labels = np.where(labels == 0, 'TA', 'TB')
    detections.data['team'] = team_labels
    distances_to_centers = kmeans.transform(colors) 
    max_distances = np.max(distances_to_centers, axis=1)

    # Trova l'istanza con la massima distanza sia dal primo che dal secondo cluster
    max_distance_index = np.argmax(max_distances)
    detections.data['team'][max_distance_index] = 'ref'
    detections.data['cluster_centers'] = kmeans.cluster_centers_

    return detections

def assign_teams(detections, cluster_centers, image, threshold=30):
    colors = extract_colors(image, detections.xyxy)
    nbrs = NearestNeighbors(n_neighbors=1).fit(cluster_centers)
    distances, indices = nbrs.kneighbors(colors)
    team_labels = np.where(indices.flatten() == 0, 'TA', 'TB')
    detections.data['team'] = team_labels
    outlier_indices = np.where(np.max(distances, axis=1) > threshold)[0]
    detections.data['team'][outlier_indices] = 'ref'

    return detections


class Detections2:
    def __init__(self, xyxy, mask, confidence, class_id, tracker_id, data):
        self.xyxy = xyxy
        self.mask = mask
        self.confidence = confidence
        self.class_id = class_id
        self.tracker_id = tracker_id
        self.data = data


# Funzione per unire due oggetti Detections
def merge_detections(det1, det2):
    merged_xyxy = np.vstack((det1.xyxy, det2.xyxy))
    merged_confidence = np.hstack((det1.confidence, det2.confidence))
    merged_class_id = np.hstack((det1.class_id, det2.class_id))

    if det1.tracker_id is not None and det2.tracker_id is not None:
        merged_tracker_id = np.hstack((det1.tracker_id, det2.tracker_id))
    else:
        merged_tracker_id = det1.tracker_id if det1.tracker_id is not None else det2.tracker_id

    # Creiamo un nuovo dizionario data unendo solo le chiavi presenti in entrambi gli oggetti
    merged_data = {}
    common_keys = set(det1.data.keys()).intersection(set(det2.data.keys()))
    for key in common_keys:
        merged_data[key] = det1.data[key]  # Puoi cambiare questa logica in base alle tue necessità

    return Detections2(
        xyxy=merged_xyxy,
        mask=None,  # Assuming mask is None for both, otherwise handle appropriately
        confidence=merged_confidence,
        class_id=merged_class_id,
        tracker_id=merged_tracker_id,
        data=merged_data
    )

def process_frame(frame: np.ndarray, _) -> np.ndarray:
    global i,detections_with_clusters,cluster_centers,det2
    results = model(frame, imgsz=1280,conf=0.3)[0]

    detections = sv.Detections.from_ultralytics(results)
    #print(detections)
    detections = tracker.update_with_detections(detections).with_nms(threshold=0.1)
    if i==0:
      det2=detections

    if i<10:
      det2=merge_detections(det2,detections)
      detections_with_clusters = create_clusters(det2,frame.copy())
      cluster_centers = detections_with_clusters.data['cluster_centers']
      print(cluster_centers)
      detections=assign_teams(detections, cluster_centers,frame.copy())
      print(detections_with_clusters)
    else:
        detections=assign_teams(detections, cluster_centers,frame.copy())

    #print(detections)
    print(i)
    # Crea le etichette usando i valori estratti
    labels = [
    f"#{tracker_id} {team}" if class_id == 1 and (team == "TA" or team == "TB") else f"#{tracker_id}"
    for tracker_id, class_id, team in zip(detections.tracker_id, detections.class_id, detections.data['team'])
    ]

    i=i+1
    print(i)
    annotated_frame = ellipse_annotator.annotate(
    scene=frame.copy(),
    detections=detections,
    )
    

    label_annotator = sv.LabelAnnotator(color=sv.Color.BLACK)

    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels)

    #detections = detections[detections.class_id == 1]
    #annotated_frame = trace_annotator.annotate(
    #       scene=annotated_frame,
    #      detections=detections)
    return annotated_frame



if __name__ == "__main__":
    HOME = os.getcwd()
    print(HOME)
    
    VIDEO_PATH="./videos/challenge-1179_1.mp4"
    file_name_with_extension = os.path.basename(VIDEO_PATH)
    videoname = os.path.splitext(file_name_with_extension)[0]
    box_annotator = sv.BoundingBoxAnnotator(thickness=1)
    ellipse_annotator = sv.EllipseAnnotator(color=sv.Color.YELLOW)
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()
    tracker = sv.ByteTrack()
    det2=None
    print(videoname)
    i=0
    sv.process_video(source_path=VIDEO_PATH, target_path=f"result.mp4", callback=process_frame)