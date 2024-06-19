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

coordinate = {
    0: (0, 0),
    1: (0, 13.84),
    2: (11, 13.84),
    3: (0, 24.84),
    4: (5.5, 24.84),
    5: (0, 43.16),
    6: (5.5, 43.16),
    7: (0, 54.16),
    8: (11, 54.16),
    9: (0, 68),
    10: (52.5, 0),
    11: (52.5, 34),
    12: (52.5, 68),
    13: (105, 0),
    14: (105, 13.84),
    15: (94, 13.84),
    16: (105, 24.84),
    17: (99.5, 24.84),
    18: (105, 43.16),
    19: (99.5, 43.16),
    20: (105, 54.16),
    21: (94, 54.16),
    22: (105, 68),
    23: (11, 24.84),
    24: (11, 43.16),
    25: (52.5, 24.85),
    26: (52.5, 43.15),
    27: (94, 24.84),
    28: (94, 43.16)
}

def pitch_estimator(field_detections):
    id_array=field_detections.class_id
    num_keypoint=len(id_array)
    points_array = field_detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
    if num_keypoint<4:
        return [],[]

    first_4_ids = id_array[:4]
    first_4_points = points_array[:4]

    # Array indice + posizione
    result_array = np.column_stack((first_4_ids, first_4_points))

    
    class_ids = result_array[:4, 0].astype(int)
    source_coordinates = np.array([point for point in first_4_points[:, 1]])
    target_coordinates = np.array([coordinate[class_id] for class_id in class_ids])
    print(result_array)
    print(target_coordinates)
    return np.array(first_4_points),np.array(target_coordinates)

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)




def filter_detections(detections):
    # Trova gli indici delle prime detections per ogni class_id
    _, unique_indices = np.unique(detections.class_id, return_index=True)

    # Filtra le detections basate sugli indici trovati
    filtered_xyxy = detections.xyxy[unique_indices]
    filtered_class_id = detections.class_id[unique_indices]
    filtered_confidence = detections.confidence[unique_indices]

    # Se ci sono tracker_id e data, filtrarli in base agli stessi indici
    filtered_tracker_id = None
    filtered_data = None
    if detections.tracker_id is not None:
        filtered_tracker_id = detections.tracker_id[unique_indices]
    if detections.data is not None:
        filtered_data = {key: detections.data[key][unique_indices] for key in detections.data}

    # Creare un nuovo oggetto Detections con le detections filtrate
    filtered_detections = sv.Detections(filtered_xyxy,None, filtered_confidence,filtered_class_id,
                                     tracker_id=filtered_tracker_id, data=filtered_data)
    return filtered_detections



def process_frame(frame: np.ndarray, _) -> np.ndarray:

    ellipse_annotator = sv.EllipseAnnotator(color=sv.Color.YELLOW,thickness=1,start_angle=0,end_angle=200)
    circle_annotator = sv.CircleAnnotator(color=sv.Color.YELLOW,thickness=1)

    results = player_model(frame, imgsz=1280,conf=0.1)[0]
    results2=field_model(frame,imgsz=1280,conf=0.01)[0]

    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections).with_nms(threshold=0.1)


    detections2= sv.Detections.from_ultralytics(results2)
    detections2=filter_detections(detections2)
    
    source,target=pitch_estimator(detections2)
    
    view_transformer = ViewTransformer(source=source, target=target)

    points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
        )
    points = view_transformer.transform_points(points=points).astype(int)
    #print(points)

    labels=[]
    #Label con le coordinate reali
    for j, detection in enumerate(detections):
        label=f"{detection[4]} xy:({points[j]})"
        labels.append(label)

    annotated_frame = circle_annotator.annotate(
    scene=frame.copy(),
    detections=detections2,
    )

    annotated_frame = ellipse_annotator.annotate(
    scene=annotated_frame,
    detections=detections,
    )


    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame

if __name__ == "__main__":
    HOME = os.getcwd()
    print(HOME)
    player_model = YOLO("./yolo_weights/best.pt")
    field_model = YOLO("./yolo_weights/keypoint_field.pt")
    VIDEO_PATH="./videos/challenge-1179_1.mp4"
    file_name_with_extension = os.path.basename(VIDEO_PATH)
    videoname = os.path.splitext(file_name_with_extension)[0]
    tracker = sv.ByteTrack()
    label_annotator = sv.LabelAnnotator(color=sv.Color.BLUE,text_padding=5)
    ellipse_annotator = sv.EllipseAnnotator(color=sv.Color.BLUE,thickness=1,start_angle=0,end_angle=200)
    sv.process_video(source_path=VIDEO_PATH, target_path=f"result.mp4", callback=process_frame)