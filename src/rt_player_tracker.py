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
from redis import Redis
from keyframe_queue_interface import *

#Coordinate in metri del Campo 
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

def cv2_to_pil(image_cv):
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    return image_pil

#Mapping tra le coordinate dell'immagine e quelle del campo
def pitch_estimator(field_detections):
    name_array = field_detections.data['class_name']
    id_array = [int(x.strip("'")) for x in name_array]
    print(id_array)
    num_keypoint=len(id_array)
    points_array = field_detections.get_anchors_coordinates(
                anchor=sv.Position.CENTER
            )
    print("points array:",points_array)
    print("classi",id_array)
    if num_keypoint<4:
        return [],[]
    first_4_ids = id_array[:4]
    first_4_points = points_array[:4]
    # Array indice + posizione
    result_array = np.column_stack((first_4_ids, first_4_points)) 
    class_ids = result_array[:4, 0].astype(int)
    source_coordinates = np.array([point for point in first_4_points[:, 1]])
    target_coordinates = np.array([coordinate[class_id] for class_id in class_ids])
    print("Result array",first_4_points)
    print("Target array",target_coordinates)
    return np.array(first_4_points),np.array(target_coordinates)


#Classe per la trasformazione delle coordinate
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





#Filtra i keypoint scartando i possibili duplicati e ordinando per confidence crescente
def filter_detections(detections):
    # Trova gli indici delle prime detections per ogni class_id
    _, unique_indices = np.unique(detections.class_id, return_index=True)
    # Filtra le detections basate sugli indici trovati
    filtered_xyxy = detections.xyxy[unique_indices]
    filtered_class_id = detections.class_id[unique_indices]
    filtered_confidence = detections.confidence[unique_indices]

    sorted_indices = np.argsort(filtered_confidence)[::-1]
    filtered_xyxy = filtered_xyxy[sorted_indices]
    filtered_class_id = filtered_class_id[sorted_indices]
    filtered_confidence = filtered_confidence[sorted_indices]

    filtered_tracker_id = None
    filtered_data = None
    if detections.tracker_id is not None:
        filtered_tracker_id = detections.tracker_id[unique_indices][sorted_indices]
    if detections.data is not None:
        filtered_data = {key: detections.data[key][unique_indices][sorted_indices] for key in detections.data}

    filtered_detections = sv.Detections(filtered_xyxy, None, filtered_confidence, filtered_class_id,
                                        tracker_id=filtered_tracker_id, data=filtered_data)
    print("filtered detections:", filtered_detections)
    return filtered_detections





def process_frame(frame: np.ndarray, _) -> np.ndarray:
    global source, target, kf_queue, seconds, minutes, frame_counter, videoname
    annotated_frame = frame.copy()
    frame_counter = frame_counter + 1
    
    try:
        if frame_counter % 60 == 0 and frame_counter != 0:
            seconds = seconds + 1
            if seconds % 60 == 0:
                minutes = minutes + 1
                seconds = 0

        if frame_counter % 6 == 0 and frame_counter != 0:
            ellipse_annotator = sv.EllipseAnnotator(color=sv.Color.YELLOW, thickness=1, start_angle=0, end_angle=200)
            circle_annotator = sv.CircleAnnotator(color=sv.Color.YELLOW, thickness=1)
            label_annotator = sv.LabelAnnotator(color=sv.Color(r=39, g=85, b=156), text_padding=5)

            # Modelli di rilevamento
            results = player_model(frame, imgsz=1280, conf=0.1)[0]
            results2 = field_model(frame, imgsz=1280, conf=0.4)[0]

            # Detections
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections).with_nms(threshold=0.1)

            detections2 = sv.Detections.from_ultralytics(results2)
            detections2 = filter_detections(detections2)

            new_source, new_target = pitch_estimator(detections2)
            if len(new_source) > 0:
                source = new_source
                target = new_target

            annotated_frame = frame.copy()

            # Trasformazione delle coordinate
            if len(source) > 0:
                view_transformer = ViewTransformer(source=source, target=target)
                points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                points = view_transformer.transform_points(points=points).astype(int)
                labels = []
                
                for j, detection in enumerate(detections):
                    label = f"{detection[4]}"
                    labels.append(label)
            else:
                labels = []
                for j, detection in enumerate(detections):
                    label = f"{detection[4]}"
                    labels.append(label)

            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            if frame_counter % 60 == 0 and frame_counter != 0:
                tracker_data = []
                for j, detection in enumerate(detections):
                    x, y = points[j][0], points[j][1]
                    x1, y1 = detection[0][0], detection[0][1]
                    x2, y2 = detection[0][2], detection[0][3]
                    tracker_data.append((int(detection[4]), int(x), int(y), int(x1), int(y1), int(x2), int(y2)))

                pil_img = cv2_to_pil(annotated_frame)
                keyframe = Keyframe(pil_img, videoname, minutes, seconds, tracker_data)
                code_len = kf_queue.push_keyframe(keyframe)
                print("Lunghezza coda:", code_len)

            print("Frame", frame_counter)
            print("Minuti", minutes)
            print("Secondi", seconds)

            # Annotazione dei keypoint e delle ellissi
            annotated_frame = circle_annotator.annotate(scene=annotated_frame, detections=detections2)
            annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=detections)

    except Exception as e:
        print(f"Errore durante l'elaborazione del frame: {e}")

    return annotated_frame


if __name__ == "__main__":
    HOME = os.getcwd()
    redis_conn = Redis(
        host=os.getenv("REDIS_HOST"),
        port=os.getenv("REDIS_PORT"),
        password=os.getenv("REDIS_PASSWORD"),
        decode_responses=True
    )

    kf_queue = KeyframeQueue(redis_conn)

    seconds=0
    minutes=0
    frame_counter=0

    print(HOME)
    player_model = YOLO("./yolo_weights/best.pt")
    field_model = YOLO("./yolo_weights/best_keypoint.pt")
    VIDEO_PATH="./videos/soccernet_2.mp4"
    file_name_with_extension = os.path.basename(VIDEO_PATH)
    videoname = os.path.splitext(file_name_with_extension)[0]
    tracker = sv.ByteTrack()
    source=[]
    target=[]
    sv.process_video(source_path=VIDEO_PATH, target_path=f"result.mp4", callback=process_frame)