import os
import gc
import re
import json
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
import torch
import umap.umap_ as umap
import supervision as sv
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from ultralytics import YOLO
from transformers import AutoModel, AutoTokenizer, AutoProcessor, SiglipVisionModel
from cv2 import dnn_superres
from more_itertools import chunked
import seaborn as sns
from keyframe_queue_interface import *
from db_control_interface import *

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# Crea una pipeline per la generazione del testo utilizzando il modello LLM
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    # The quantization line
    model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit": True}
)

# Token di terminazione utilizzati per la generazione del testo
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


# Funzione per la generazione del testo
def generate_text(messages,temp=0.0):
    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Applica la pipeline al prompt
    outputs = pipeline(
        prompt,
        max_new_tokens=350,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temp,
        top_p=0.9,
    )

    # Restituisce il testo generato, escludendo il prompt
    generated_text = outputs[0]["generated_text"][len(prompt):]
    return generated_text.strip()  # Rimuove eventuali spazi bianchi in eccesso


lista_predicati = "interaction,is,has"



def genera_tripla(stringa):
    messages = [
    {"role": "system", "content": "Answer questions"},
    {"role": "user", "content": f""" From this text you have to extract informations about players identified by the #number (for example player possession of the ball or interaxctions between players)
    and write them like triplets: (entity1,predicate,entity2)
{stringa}
The predicate must be one of these: is,has,interaction
example:
(Tracker#4,interaction,Tracker#5)
"""},
    ]
    generated_response = generate_text(messages,0.3)
    return generated_response

entity_list="""Tracker,Player,BallTracker"""

def categorizza_tripla(stringa):
    messages = [
    {"role": "system", "content": "Answer questions"},
    {"role": "user", "content": f""" For each numbered triple  1. (entity1, relation, entity2) , categorize each entity by adding the category in square brackets next to the name (both for the first and the second entity) according to one of the proposed categories below (each entity must necessarily be associated with one of these, if not possible ignore the triplet):
{entity_list}

Example of the output that you must strictly provide considering the brackets:
1.(entity1[category_x], relation, entity2[category_y])
2.(entity1[category_z], relation, entity2[category_x])
3.(entity1[category_j], relation, entity2[category_k])
4.(entity2[category_z], relation, entity5[category_x])

Do not provide more details than I have requested.
These are the lines you need to adapt:
{stringa}
The predicate must be one of this list:
{lista_predicati}

"""},
    ]
    generated_response = generate_text(messages,0.4)
    return generated_response

model_numbers = YOLO("/content/Soccer_Tactical_Analysis/src/yolo_weights/shirt_numbers.pt")

sr = dnn_superres.DnnSuperResImpl_create()
path = "/content/Soccer_Tactical_Analysis/src/EDSR_x4.pb"
sr.readModel(path)

sr.setModel("edsr", 2)
def pil_to_cv2(image_pil):
    image_cv = np.array(image_pil)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    return image_cv

def cv2_to_pil(image_cv):
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    return image_pil

def apply_sharpening(image_cv):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(image_cv, -2, kernel)
    return sharpened

def remove_green_color(image_cv):
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    image_cv[mask > 0] = gray_bgr[mask > 0]
    return image_cv

def remove_green(image_pil):
    image_cv = pil_to_cv2(image_pil)
    result_cv = remove_green_color(image_cv)
    result_pil = cv2_to_pil(result_cv)
    return result_pil

def calculate_average_color(image_cv):
    average_color_per_row = np.average(image_cv, axis=0)
    average_color = np.average(average_color_per_row, axis=0)
    return average_color

def label_color(average_color, reference_colors):
    min_distance = float('inf')
    closest_color_name = None
    for color_name, color_value in reference_colors.items():
        distance = np.linalg.norm(average_color - np.array(color_value))
        if distance < min_distance:
            min_distance = distance
            closest_color_name = color_name
    return closest_color_name

reference_colors = {
    'rosso': [255, 0, 0],
    'verde': [0, 255, 0],
    'blu': [0, 0, 255],
    'giallo': [255, 255, 0],
    'ciano': [0, 255, 255],
    'viola': [255, 0, 255],
    'bianco': [255, 255, 255],
    'nero': [0, 0, 0],
}

def process_image_with_label(image_pil, sr):
    upscaled_image_cv=image_pil
    average_color = calculate_average_color(upscaled_image_cv)
    color_label = label_color(average_color, reference_colors)
    return upscaled_image_pil, color_label

model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
model.eval()

def parse_string_to_triplets(input_string, minutes, seconds):
    triplet_regex = re.compile(r'\(([^[]+)\[([^]]+)\],\s*([^,]+),\s*([^[]+)\[([^]]+)\]\)')
    triplets = []

    for match in triplet_regex.findall(input_string):
        entity1_name, category1, relation_name, entity2_name, category2 = match
        entity1_name = entity1_name.replace(" ", "").replace("#", "")
        category1 = category1.replace(" ", "").replace("#", "")
        entity2_name = entity2_name.replace(" ", "").replace("#", "")
        category2 = category2.replace(" ", "").replace("#", "")
        relation_name = relation_name.replace(" ", "").replace("#", "")
        triplet = Triplet(entity1_name, category1, entity2_name, category2, relation_name, int(minutes), int(seconds))
        triplets.append(triplet)

    return triplets

def parse_string_to_triplets(input_string, minutes, seconds):
    # Regular expression to match the format of the triplets in the string
    triplet_regex = re.compile(r'\(([^[]+)\[([^]]+)\],\s*([^,]+),\s*([^[]+)\[([^]]+)\]\)')
    triplets = []
    discarded_triplets = []

    # Definisce le categorie e le relazioni valide
    valid_categories = {"Tracker", "Player", "BallTracker", "Pos", "Team"}
    valid_relations = {"is", "has", "interaction", "position", "memberof"}

    for match in triplet_regex.findall(input_string):
        entity1_name, category1, relation_name, entity2_name, category2 = match

        # Rimozione di spazi
        entity1_name = entity1_name.replace(" ", "").replace("#", "")
        category1 = category1.replace(" ", "").replace("#", "")
        entity2_name = entity2_name.replace(" ", "").replace("#", "")
        category2 = category2.replace(" ", "").replace("#", "")

        # Rimozione degli spazi iniziali nella relazione
        relation_name = relation_name.lstrip().replace("#", "")

        # Correzione della "t" minuscola in "Tracker" nei nomi delle entità
        entity1_name = re.sub(r'\bplayer', 'Player', entity1_name, flags=re.IGNORECASE)
        entity2_name = re.sub(r'\bplayer', 'Player', entity2_name, flags=re.IGNORECASE)


        # Correzione della "t" minuscola in "Tracker" nei nomi delle entità
        entity1_name = re.sub(r'\btracker', 'Tracker', entity1_name, flags=re.IGNORECASE)
        entity2_name = re.sub(r'\btracker', 'Tracker', entity2_name, flags=re.IGNORECASE)


        # Verifica delle categorie e relazioni valide
        if category1 not in valid_categories or category2 not in valid_categories:
            discarded_triplets.append((entity1_name, category1, relation_name, entity2_name, category2))
            continue
        if relation_name not in valid_relations:
            discarded_triplets.append((entity1_name, category1, relation_name, entity2_name, category2))
            continue

        # Verifica della corrispondenza tra categoria e prefisso del nome
        if (category1 == "Tracker" and not entity1_name.startswith("Tracker")) or \
           (category1 == "Player" and not entity1_name.startswith("Player")):
            discarded_triplets.append((entity1_name, category1, relation_name, entity2_name, category2))
            continue
        if category2 == "Player" and not (entity2_name.startswith("Player") or entity2_name.startswith("Goalkeeper")):
            discarded_triplets.append((entity1_name, category1, relation_name, entity2_name, category2))
            continue

        # Verifica della presenza di un numero nei nomi delle entità per Tracker o Player
        if ("Tracker" in entity1_name or "Player" in entity1_name) and not any(char.isdigit() for char in entity1_name):
            discarded_triplets.append((entity1_name, category1, relation_name, entity2_name, category2))
            continue
        if ("Tracker" in entity2_name or "Player" in entity2_name) and not any(char.isdigit() for char in entity2_name):
            discarded_triplets.append((entity1_name, category1, relation_name, entity2_name, category2))
            continue

        # Verifica che "Team" nel nome contenga almeno un altro carattere
        if ("Team" in entity1_name and len(entity1_name) == len("Team")) or ("Team" in entity2_name and len(entity2_name) == len("Team")):
            discarded_triplets.append((entity1_name, category1, relation_name, entity2_name, category2))
            continue

        # Creazione della tripletta valida
        triplet = Triplet(entity1_name, category1, entity2_name, category2, relation_name, int(minutes), int(seconds))
        triplets.append(triplet)

    # Stampa delle triplette conservate e scartate
    print("Triplette conservate:")
    for t in triplets:
        print(t)

    print("\nTriplette scartate:")
    for t in discarded_triplets:
        print(f"({t[0]}[{t[1]}], {t[2]}, {t[3]}[{t[4]}])")

    return triplets

#ritorna il numero di maglia a partire dalle detections
def get_class_names(detections):
    if detections['class_name'] is None:
        return None
    print(detections['class_id'])
    num_detections = len(detections['class_name'])
    if num_detections == 1:
        return detections['class_name'][0]

    elif num_detections >= 2:
         print(detections.xyxy)
         if detections.xyxy[0][0] < detections.xyxy[1][0]:
          return detections['class_name'][0]+ detections['class_name'][1]
         else:
          return detections['class_name'][1]+ detections['class_name'][0]

    else:
        return None
    
encodedCA=os.getenv("ARANGO_ENCODEDCA")
db_host=os.getenv("ARANGO_HOST")
db_username=os.getenv("ARANGO_USERNAME")
db_password=os.getenv("ARANGO_PASSWORD")

sys_db, client = connect_to_arangodb(encodedCA, db_host, db_username, db_password)




SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(DEVICE)
EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)

#Versione SigLip e UMAP


def filtra_righe_categorie(string):
    lines = string.splitlines()
    filtered_lines = [line for line in lines if line.strip().startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
                      and line.strip().endswith(')')
                      and line.count(',') == 2
                      and line.count('[') == 2
                      and line.count('(') == 1
                      and line.count(')') == 1
                      and line.count(']') == 2]
    return '\n'.join(filtered_lines)

def analyze_image_interactions(image_patch, question):
    msgs = [{'role': 'user', 'content': question}]
    res = model.chat(
        image=image_patch,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.1,
    )

    return res

def assign_team_based_on_position(cluster_positions):
    cluster_teams = {}

    for cluster_id, positions in cluster_positions.items():
        # Conta quanti giocatori nel cluster si trovano nella parte destra dell'immagine (x > 0.5)
        right_count = sum(1 for pos in positions if pos[0] > 0.5)
        cluster_teams[cluster_id] = right_count

    # Determina quale cluster ha più giocatori nella parte destra
    if cluster_teams[0] > cluster_teams[1]:
        return ['Team2', 'Team1']
    else:
        return ['Team1', 'Team2']





def process_detections(keyframe,sr,triple_team):
    image = keyframe.image
    raw_image=keyframe.raw_image
    results = []
    plt.figure(figsize=(10, 10))  # Adjust these numbers to increase the plot size
    plt.imshow(image)
    plt.axis('off')  # Remove the axis
    plt.show()
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    plt.axis('off')
    plt.show()

    question2 = """each player tracker has an identifier , for each player (identified by # number) tell me in detail the interactions between them. write a list (for example: Tracker#2 interacts with Tracker #30)"""
    generated_text = analyze_image_interactions(image, question2)
    results.append(generated_text)
    print(generated_text)

    question2 = """each player tracker has an identifier  (#number) , there is one player in possession of the ball, write his tracker #number. Write one single word!"""
    generated_text = analyze_image_interactions(image, question2)
    print(generated_text)
    if generated_text != "none" and generated_text != None:
        if generated_text.startswith("Tracker"):
            triple_team += f"1. ({generated_text}[Tracker],has,BallTracker0[BallTracker])\n"
        else:
          if generated_text.isdigit() or (generated_text.startswith('#') and generated_text[1:].isdigit()):
            triple_team += f"1. (Tracker{generated_text}[Tracker],has,BallTracker0[BallTracker])\n"

    results.append(generated_text)
    print(generated_text)

    player_positions = []
    all_images = []  #  immagini dei giocatori per embedding
    tracker_ids = []  # id dei tracker per la successiva assegnazione

    # Itera sui tracker dei giocatori
    if not isinstance(keyframe.tracker_data, list):
        print(f"Errore: 'keyframe.tracker_data' non è una lista, ma {type(keyframe.tracker_data)}")
        return results, triple_team

    for i, data in enumerate(keyframe.tracker_data):
        xmin, ymin, xmax, ymax = data[3], data[4], data[5], data[6]
        width = xmax - xmin
        height = ymax - ymin

        if width > 0 and height > 0 and data[7] == 2:
            image_patch = raw_image.crop((xmin, ymin, xmax, ymax))
            #image_patch = remove_green(image_patch)
            all_images.append(image_patch)
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            player_positions.append((x_center, y_center))
            tracker_ids.append(data[0])

    # Verifica che ci siano immagini prima di procedere
    if len(all_images) == 0:
        print("Nessuna immagine valida trovata per i giocatori.")
        return results, triple_team

    # Estrazione embedding
    BATCH_SIZE = 32
    batches = chunked(all_images, BATCH_SIZE)
    data_embeddings = []

    with torch.no_grad():
        for batch in tqdm(batches, desc='embedding extraction'):
            if len(batch) == 0:
                continue
            inputs = EMBEDDINGS_PROCESSOR(images=batch, return_tensors="pt").to(DEVICE)
            outputs = EMBEDDINGS_MODEL(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
            data_embeddings.append(embeddings)

    # Verifica che ci siano embeddings validi prima di concatenare
    if len(data_embeddings) == 0:
        print("Nessun embedding estratto.")
        return results, triple_team

    # Concatena solo se ci sono embeddings
    data_embeddings = np.concatenate(data_embeddings)

    # Imposta il numero di vicini in base alla dimensione del dataset
    n_neighbors = min(15, len(data_embeddings) - 1) if len(data_embeddings) > 1 else 1  # Almeno 1 vicino

    # Clustering con UMAP e KMeans
    REDUCER = umap.UMAP(n_components=3, n_neighbors=n_neighbors)
    CLUSTERING_MODEL_EMBEDDINGS = KMeans(n_clusters=2)

    projections = REDUCER.fit_transform(data_embeddings)
    clusters_embeddings = CLUSTERING_MODEL_EMBEDDINGS.fit_predict(projections)

    # Raccoglie le posizioni dei giocatori per ogni cluster
    cluster_positions = {0: [], 1: []}

    for i, position in enumerate(player_positions):
        cluster_label = clusters_embeddings[i]
        cluster_positions[cluster_label].append(position)

    # Assegna i team basandosi sulle posizioni medie dei cluster
    team_assignment = assign_team_based_on_position(cluster_positions)

    for cluster_id, team in enumerate(team_assignment):
        print(f"Cluster {cluster_id} assegnato a {team}")

    # Assegna i team ai giocatori e visualizza le patch
    for i, data in enumerate(keyframe.tracker_data):
        xmin, ymin, xmax, ymax = data[3], data[4], data[5], data[6]
        width = xmax - xmin
        height = ymax - ymin

        if width > 0 and height > 0 and data[7] == 2:
            image_patch = raw_image.crop((xmin, ymin, xmax, ymax))
            #image_patch = remove_green(image_patch)
            plt.imshow(image_patch)
            assigned_team = team_assignment[clusters_embeddings[i]]
            plt.title(f"Team: {assigned_team}")  # Mostra il team di appartenenza
            plt.axis('off')
            plt.show()

            triple_team += f"1. (Tracker #{data[0]}[Tracker],memberof,{assigned_team}[Team])\n"

            results_number = model_numbers(image_patch, conf=0.74)[0]
            detections = sv.Detections.from_ultralytics(results_number)
            print(detections['class_name'])

            result = get_class_names(detections)
            if result is not None:
                print(f"Detections found: {result}")
                triple_team += f"1. (Tracker#{data[0]}[Tracker],is,Player{result}[Player])\n"
            else:
                print("No valid detections or no return value.")
        elif data[7] == 1:
            triple_team += f"1. (Tracker#{data[0]}[Tracker],is,Goalkeeper[Player])\n"

    return results, triple_team

def process_keyframe(retrieved_keyframe, sr, client, db_username, db_password):
    """
    Estrae un keyframe dalla coda, elabora i dati di tracking, genera tripli e li salva nel database.

    Args:
        kf_queue (KeyFrameQueue): La coda dei keyframe.
        sr (object): Oggetto di riconoscimento da passare a `process_detections`.
        client (object): Client per accedere al database.
        db_username (str): Username del database.
        db_password (str): Password del database.

    Returns:
        dict: Un dizionario con le informazioni elaborate, inclusi i tripli generati.
    """
    # Estrai il keyframe
    img_frame = retrieved_keyframe.image
    triple_pos = ""
    triple_team = ""
    
    # Stampa tempo del frame
    print(retrieved_keyframe.minute, retrieved_keyframe.second)
    
    # Genera triple di posizione
    for item in retrieved_keyframe.tracker_data:
        if 0 <= item[1] <= 105 and 0 <= item[2] <= 68:  # Controllo posizione valida
            if item[7] != 0:
                triple_pos += f"1. (Tracker #{item[0]}[Tracker], position, {item[1]}:{item[2]}[Pos])\n"
            else:
                triple_pos += f"1. (BallTracker #0[BallTracker], position, {item[1]}:{item[2]}[Pos])\n"
    
    # Elabora le rilevazioni
    results, triple_team = process_detections(retrieved_keyframe, sr, triple_team)
    torch.cuda.empty_cache()
    
    # Genera e categorizza le triple
    tripla = genera_tripla(results)
    tripla = categorizza_tripla(tripla)
    torch.cuda.empty_cache()
    
    # Filtra le righe per categorie
    triple = filtra_righe_categorie(tripla)
    lines = triple.splitlines()
    
    # Modifica le righe per sostituire i nomi
    modified_lines = []
    for line in lines:
        if '(' in line:
            first_comma_index = line.index(',')
            modified_line = line[:first_comma_index].replace('Player', 'Tracker').replace('[Tracker]', '[Tracker]') + line[first_comma_index:]
            modified_lines.append(modified_line)
        else:
            modified_lines.append(line)
    
    modified_text = "\n".join(modified_lines)
    triple_total = modified_text + "\n" + triple_pos + triple_team
    print(triple_total)
    
    # Converte il testo in tripli e li salva nel database
    array_triple = parse_string_to_triplets(triple_total, retrieved_keyframe.minute, retrieved_keyframe.second)
    for triplet in array_triple:
        db = initialize_database(client, retrieved_keyframe.video_name, db_username, db_password)
        push_triplet(db, retrieved_keyframe.video_name, triplet)
    
    # Restituisce le informazioni elaborate
    return {
        "minute": retrieved_keyframe.minute,
        "second": retrieved_keyframe.second,
        "triple_total": triple_total,
        "array_triple": array_triple
    }

def filtra_righe_categorie(string):
    lines = string.splitlines()
    filtered_lines = [line for line in lines if line.strip().startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
                      and line.strip().endswith(')')
                      and line.count(',') == 2
                      and line.count('[') == 2
                      and line.count('(') == 1
                      and line.count(')') == 1
                      and line.count(']') == 2]
    return '\n'.join(filtered_lines)

def analyze_image_interactions(image_patch, question):
    msgs = [{'role': 'user', 'content': question}]
    res = model.chat(
        image=image_patch,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.1,
    )

    return res


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

from itertools import combinations

def are_points_aligned(points):
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return len(set(xs)) == 1 or len(set(ys)) == 1

def has_three_equal_coordinates(points):
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return xs.count(max(set(xs), key=xs.count)) >= 3 or ys.count(max(set(ys), key=ys.count)) >= 3

def is_valid_combination(points):
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x_counts = {x: xs.count(x) for x in set(xs)}
        y_counts = {y: ys.count(y) for y in set(ys)}
        return all(count <= 2 for count in x_counts.values()) and all(count <= 2 for count in y_counts.values())

def pitch_estimator(field_detections):
    name_array = field_detections.data['class_name']
    id_array = [int(x.strip("'")) for x in name_array]
    num_keypoint = len(id_array)

    points_array = field_detections.get_anchors_coordinates(
                anchor=sv.Position.CENTER
            )

    print("ID array:", id_array)
    print("Points array:", points_array)

    if num_keypoint < 4:
        return [], []

    for comb in combinations(range(num_keypoint), 4):
        selected_points = [points_array[i] for i in comb]
        selected_ids = [id_array[i] for i in comb]

        if is_valid_combination(selected_points):
            target_coords = [coordinate[id] for id in selected_ids]
            if not are_points_aligned(target_coords) and not has_three_equal_coordinates(target_coords):
                source_coordinates = np.array(selected_points)
                target_coordinates = np.array(target_coords)
                print("Source points:", source_coordinates)
                print("Target coordinates:", target_coordinates)
                return source_coordinates, target_coordinates
    return [], []


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)
        print("Matrice di trasformazione:", self.m)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        print("Punti trasformati:", transformed_points)
        return transformed_points.reshape(-1, 2)




#Filtra i keypoint scartando i possibili duplicati e ordinanda per confidence crescente
def filter_detections(detections):
    _, unique_indices = np.unique(detections.class_id, return_index=True)
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

        if frame_counter % 5 == 0 and frame_counter != 0:
            ellipse_annotator = sv.EllipseAnnotator(color=sv.Color.YELLOW, thickness=1, start_angle=0, end_angle=200)
            circle_annotator = sv.CircleAnnotator(color=sv.Color.YELLOW, thickness=1)
            label_annotator = sv.LabelAnnotator(color=sv.Color(r=39, g=85, b=156), text_padding=5)

            #Calcolo delle detection sul frame
            results = player_model(frame, imgsz=1280, conf=0.6)[0]
            results2 = field_model(frame, imgsz=1280, conf=0.08)[0]

            # Detections players e keypoints
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
                    #label=f"{detection[4]} xy:({points[j]}){detection[3]} "
                    label = f"#{detection[4]}"
                    labels.append(label)
            else:
                labels = []
                for j, detection in enumerate(detections):
                    label = f"{detection[4]}"
                    labels.append(label)

            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            if frame_counter % 60 == 0 and frame_counter != 0 and len(source)>0:
                tracker_data = []
                for j, detection in enumerate(detections):
                    x, y = points[j][0], points[j][1]
                    x1, y1 = detection[0][0], detection[0][1]
                    x2, y2 = detection[0][2], detection[0][3]
                    #se classe != refree
                    if int(detection[3]) != 3:
                      #tracker, pitch coordinates, bounding boxes coordinate, classe
                      tracker_data.append((int(detection[4]), int(x), int(y), int(x1), int(y1), int(x2), int(y2),int(detection[3])))

                pil_img = cv2_to_pil(annotated_frame)
                raw_pil_img = cv2_to_pil(frame)
                keyframe = Keyframe(pil_img,raw_pil_img, videoname, minutes, seconds, tracker_data)
                result = process_keyframe(keyframe, sr, client, db_username, db_password)

            print("Frame", frame_counter)
            print("Minuti", minutes)
            print("Secondi", seconds)

            # Annotazione dei keypoint e delle ellissi
            annotated_frame = sv.BoundingBoxAnnotator().annotate(scene=annotated_frame, detections=detections2)
            annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=detections)

    except Exception as e:
        print(f"Errore durante l'elaborazione del frame: {e}")
        exit
    return annotated_frame



if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(description="Video processing script")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--player_model_path", type=str, required=True, help="Path to the player detection model")
    parser.add_argument("--field_model_path", type=str, required=True, help="Path to the field detection model")
    parser.add_argument("--output_path", type=str, default="result.mp4", help="Path to save the output video")
    args = parser.parse_args()


    seconds = 0
    minutes = 0
    frame_counter = 0

    # Print current working directory
    HOME = os.getcwd()
    print(HOME)

    # Load models
    player_model = YOLO(args.player_model_path)
    field_model = YOLO(args.field_model_path)

    # Video path
    VIDEO_PATH = args.video_path
    file_name_with_extension = os.path.basename(VIDEO_PATH)
    videoname = os.path.splitext(file_name_with_extension)[0]

    # Tracker initialization
    tracker = sv.ByteTrack()
    source = []
    target = []

    # Process video
    sv.process_video(
        source_path=VIDEO_PATH,
        target_path=args.output_path,
        callback=process_frame
    )
