import torch
from PIL import Image
import supervision as sv
from transformers import AutoModel, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
import torch
import gc
import os
import re
import transformers
import traceback
from redis import Redis
from keyframe_queue_interface import *
from db_control_interface import *
import cv2
from cv2 import dnn_superres
from more_itertools import chunked
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import umap.umap_ as umap
from tqdm import tqdm
from PIL import Image
from more_itertools import chunked
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoProcessor, SiglipVisionModel

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

redis_conn = Redis(
    host=os.getenv("REDIS_HOST"),
    port=os.getenv("REDIS_PORT"),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)


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

def process_keyframe(kf_queue, sr, client, db_username, db_password):
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
    keyframe_ref, retrieved_keyframe = kf_queue.peek_keyframe_backup()
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
        "keyframe_ref": keyframe_ref,
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


while 1:
    try:
        kf_queue = KeyframeQueue(redis_conn)
        back_len = kf_queue.BU_list_len()
        print("Dimensione coda principale:", back_len)
        if back_len == 0:
            break

        if back_len > 0:
            result = process_keyframe(kf_queue, sr, client, db_username, db_password)
            back_len = kf_queue.BU_list_len()
            print("Dimensione coda backup prima della delete:", back_len)
            kf_queue.del_keyframe_from_backup(keyframe_ref)
            back_len = kf_queue.BU_list_len()
            print("Dimensione coda backup dopo la delete:", back_len)

        else:
            exit

    except Exception as e:
        print(f"Errore rilevato: {e}")
        traceback.print_exc()  # Stampa il traceback completo con la riga di errore
    finally:
        torch.cuda.empty_cache()


