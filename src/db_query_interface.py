from arango import ArangoClient
from collections import defaultdict
import matplotlib.pyplot as plt
import base64
import sys

def connect_to_arangodb(encodedCA, db_host, db_username, db_password):
    try:
        # Decode the certificate
        file_content = base64.b64decode(encodedCA)
        
        # Write the certificate to a file
        with open("cert_file.crt", "w+") as f:
            f.write(file_content.decode("utf-8"))
    except Exception as e:
        print(f"Error decoding and writing certificate: {str(e)}")
        sys.exit(1)

    # Initialize the ArangoDB client
    try:
        client = ArangoClient(
            hosts=db_host, verify_override="cert_file.crt"
        )
    except Exception as e:
        print(f"Failed to initialize ArangoDB client: {str(e)}")
        sys.exit(1)

    # Connect to the '_system' database as the specified user
    try:
        sys_db = client.db("_system", username=db_username, password=db_password)
        print("ArangoDB version:", sys_db.version())
        return sys_db, client
    except Exception as e:
        print(f"Failed to connect to ArangoDB: {e}")
        sys.exit(1)

def get_graph(client, db_username,db_password,db_name, graph_name):
    db = client.db(db_name,db_username,db_password)
    graph=db.graph("dd")
    archi=graph.edge_collection('is')
    graph_list = list(archi)
    archi=graph.edge_collection('has')
    archi=list(archi)
    graph_list = graph_list+archi
    archi=graph.edge_collection('position')
    archi=list(archi)
    graph_list = graph_list+archi
    archi=graph.edge_collection('interaction')
    archi=list(archi)
    graph_list = graph_list+archi
    archi=graph.edge_collection('memberof')
    archi=list(archi)
    graph_list = graph_list+archi
    #print(graph_list)
    return graph_list

def get_graph2(client, db_username,db_password,db_name, graph_name):
    db = client.db(db_name,db_username,db_password)
    graph=db.graph("dd")
    archi=graph.edge_collection('is')
    graph_list = list(archi)
    archi=graph.edge_collection('position')
    archi=list(archi)
    graph_list = graph_list+archi
    archi=graph.edge_collection('interaction')
    archi=list(archi)
    graph_list = graph_list+archi
    archi=graph.edge_collection('memberof')
    archi=list(archi)
    graph_list = graph_list+archi
    #print(graph_list)
    return graph_list





def analyze_graph(edges):
    # Contatori per archi, categorie di nodi, categorie di archi e un insieme per i nodi unici
    edge_count = 0
    node_degrees = {}  # Dizionario per tracciare il grado di ciascun nodo
    category_count = {}  # Dizionario per tracciare il numero di nodi per categoria
    edge_category_count = {}  # Dizionario per tracciare il numero di archi per categoria
    unique_nodes = set()  # Insieme per tracciare i nodi unici

    # Iterazione su ogni arco nella lista
    for edge in edges:
        edge_count += 1
        
        # Estrazione dei nodi collegati dall'arco
        _from = edge['_from'].split('/')[1]
        _to = edge['_to'].split('/')[1]

        # Estrai la categoria dell'arco dall'_id (prima dello "/")
        edge_category = edge['_id'].split('/')[0]
        if edge_category in edge_category_count:
            edge_category_count[edge_category] += 1
        else:
            edge_category_count[edge_category] = 1

        # Aggiungi i nodi all'insieme dei nodi unici
        if _from not in unique_nodes:
            unique_nodes.add(_from)
            # Estrai la categoria del nodo _from e aggiorna il conteggio
            category_from = edge['_from'].split('/')[0]
            if category_from in category_count:
                category_count[category_from] += 1
            else:
                category_count[category_from] = 1

        if _to not in unique_nodes:
            unique_nodes.add(_to)
            # Estrai la categoria del nodo _to e aggiorna il conteggio
            category_to = edge['_to'].split('/')[0]
            if category_to in category_count:
                category_count[category_to] += 1
            else:
                category_count[category_to] = 1

        # Aggiorna i gradi per i nodi collegati
        if _from in node_degrees:
            node_degrees[_from] += 1
        else:
            node_degrees[_from] = 1

        if _to in node_degrees:
            node_degrees[_to] += 1
        else:
            node_degrees[_to] = 1

    # Trova il nodo con il massimo grado
    max_degree_node = max(node_degrees, key=node_degrees.get)

    # Numero di nodi unici
    node_count = len(unique_nodes)

    return {
        'node_count': node_count,
        'edge_count': edge_count,
        'max_degree_node': max_degree_node,
        'category_count': category_count,
        'node_degrees': node_degrees,  # Restituisce i gradi dei nodi
        'edge_category_count': edge_category_count  # Restituisce il conteggio per ogni categoria di arco
    }


from collections import defaultdict, Counter

def evaluate_graph(edges):
    # Dizionari per tracciare le associazioni di tracker con team e player
    tracker_to_teams = defaultdict(Counter)
    tracker_to_players = defaultdict(Counter)

    # Conteggio degli archi totali e incoerenti
    total_edges = 0
    inconsistent_edges = 0

    for edge in edges:
        total_edges += 1

        # Ottieni i nodi coinvolti nell'arco
        source_node = edge.get('_from')
        target_node = edge.get('_to')

        # Verifica se entrambi i nodi sono presenti
        if not source_node or not target_node:
            print(f"Warning: Missing '_from' or '_to' in edge: {edge}")
            continue

        # Determina se il nodo è un tracker, un team o un player
        if "Tracker" in source_node or "Tracker" in target_node:
            tracker = source_node if "Tracker" in source_node else target_node
            team_or_player = target_node if "Tracker" in source_node else source_node

            # Verifica se l'arco è associato a un team
            if "Team" in team_or_player:
                team = team_or_player
                tracker_to_teams[tracker][team] += 1
            # Verifica se l'arco è associato a un player
            elif "Player" in team_or_player:
                player = team_or_player
                tracker_to_players[tracker][player] += 1

    # Verifica la coerenza delle associazioni
    for tracker, teams_counter in tracker_to_teams.items():
        max_team_count = max(teams_counter.values(), default=0)
        total_tracker_associations = sum(teams_counter.values())
        inconsistent_edges += total_tracker_associations - max_team_count

    for tracker, players_counter in tracker_to_players.items():
        max_player_count = max(players_counter.values(), default=0)
        total_tracker_associations = sum(players_counter.values())
        inconsistent_edges += total_tracker_associations - max_player_count

    # Calcolo del rapporto di coerenza
    consistent_edges = total_edges - inconsistent_edges
    consistency_ratio = consistent_edges / total_edges if total_edges > 0 else 0

    return consistency_ratio


import os

def plot_graph_analysis(graph_analysis, save_path="./plots"):
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    # 1. Plot the distribution of instances per category
    categories = list(graph_analysis.get('category_count', {}).keys())
    counts = list(graph_analysis.get('category_count', {}).values())

    plt.figure(figsize=(10, 6))
    plt.bar(categories, counts, color='skyblue')
    plt.xlabel('Category')
    plt.ylabel('Number of Instances')
    plt.title('Distribution of Instances per Category')
    plt.xticks(rotation=45)
    
    # Save or show the plot
    if save_path:
        plt.savefig(os.path.join(save_path, 'category_distribution.png'), bbox_inches='tight')
    else:
        plt.show()

    # 2. Plot the distribution of instances per edge category
    categories = list(graph_analysis.get('edge_category_count', {}).keys())
    counts = list(graph_analysis.get('edge_category_count', {}).values())

    plt.figure(figsize=(10, 6))
    plt.bar(categories, counts, color='skyblue')
    plt.xlabel('Edge Category')
    plt.ylabel('Number of Instances')
    plt.title('Distribution of Instances per Edge Category')
    plt.xticks(rotation=45)
    
    # Save or show the plot
    if save_path:
        plt.savefig(os.path.join(save_path, 'edge_category_distribution.png'), bbox_inches='tight')
    else:
        plt.show()

    # 3. Plot node degrees per category
    categories = list(graph_analysis.get('category_count', {}).keys())
    node_degrees = graph_analysis.get('node_degrees', {})

    if not node_degrees:
        print("No data available for 'node_degrees'.")
        return

    for category in categories:
        # Filter nodes that belong to the current category
        category_nodes = {node: degree for node, degree in node_degrees.items()
                          if node.startswith(category)}

        if category_nodes:
            # Sort nodes by degree in descending order
            sorted_nodes = sorted(category_nodes.items(), key=lambda item: item[1], reverse=True)

            # Check if there are at least 15 nodes
            if len(sorted_nodes) > 0:
                # Select only the top 15 nodes or fewer if there are less than 15
                top_nodes = sorted_nodes[:10]

                # Extract node IDs and degrees from the selected nodes
                node_ids, degrees = zip(*top_nodes)

                # Find the node with the highest number of connections (maximum degree) among the top 15
                max_node = node_ids[0]
                max_degree = degrees[0]

                plt.figure(figsize=(10, 6))
                plt.bar(node_ids, degrees, color='lightgreen')
                plt.xlabel('Node ID')
                plt.ylabel('Number of Connections (Degree)')
                plt.title(f'Node Degrees in Category "{category}"')
                plt.xticks(rotation=90)

                # Highlight the node with the maximum degree
                plt.bar(max_node, max_degree, color='red')
                
                # Save or show the plot
                if save_path:
                    plt.savefig(os.path.join(save_path, f'node_degrees_{category}.png'), bbox_inches='tight')
                else:
                    plt.show()
            else:
                print(f"No nodes found in the category '{category}'.")


def find_graph_gaps(edges):
    # Sets to track nodes and relationships
    trackers = set()  # Nodes of type Tracker
    players = set()  # Nodes of type Player
    positions = set()  # Nodes of type Position
    
    # Sets to track connections
    tracker_to_player = set()  # "is" connections from Tracker to Player
    tracker_to_position = set()  # Connections from Tracker to Position
    tracker_to_memberof = set()  # "memberof" connections from Tracker
    
    # Counters for each type of connection (to get totals per relation type)
    total_is_connections = 0
    total_position_connections = 0
    total_memberof_connections = 0
    
    # Iterate over each edge in the graph
    for edge in edges:
        _from_type, _from_id = edge['_from'].split('/')
        _to_type, _to_id = edge['_to'].split('/')
        
        # If the source node is a Tracker, add it to the Tracker set
        if _from_type == 'Tracker':
            trackers.add(_from_id)
        
        # Check if there's an "is" relation between Tracker and Player
        if _from_type == 'Tracker' and _to_type == 'Player' and edge['relation'] == 'is':
            tracker_to_player.add(_from_id)
            total_is_connections += 1  # Increment total "is" connections
        
        # Check if there's a connection between Tracker and Position
        if _from_type == 'Tracker' and _to_type == 'Pos':
            tracker_to_position.add(_from_id)
            total_position_connections += 1  # Increment total position connections
        
        # Check if there's a "memberof" relation between Tracker and any other entity
        if _from_type == 'Tracker' and edge['relation'] == 'memberof':
            tracker_to_memberof.add(_from_id)
            total_memberof_connections += 1  # Increment total "memberof" connections
    
    # Type 1 gaps: Trackers with no "is" connection to any Player
    missing_player_connections = trackers - tracker_to_player
    total_possible_is_connections = len(trackers)  # Trackers should have "is" connections

    # Type 2 gaps: Trackers with no connection to any Position
    missing_position_connections = trackers - tracker_to_position
    total_possible_position_connections = len(trackers)  # Trackers should have position connections

    # Type 3 gaps: Trackers with no "memberof" connection
    missing_memberof_connections = trackers - tracker_to_memberof
    total_possible_memberof_connections = len(trackers)  # Trackers should have "memberof" connections

    # Counts of missing connections
    missing_player_count = len(missing_player_connections)
    missing_position_count = len(missing_position_connections)
    missing_memberof_count = len(missing_memberof_connections)
    
    # Calculate ratios based on the total possible connections of each type
    missing_player_ratio = missing_player_count / total_possible_is_connections if total_possible_is_connections > 0 else 0
    missing_position_ratio = missing_position_count / total_possible_position_connections if total_possible_position_connections > 0 else 0
    missing_memberof_ratio = missing_memberof_count / total_possible_memberof_connections if total_possible_memberof_connections > 0 else 0

    # Total missing connections count (sum of all types of missing connections)
    total_missing_connections = (
        missing_player_count + missing_position_count + missing_memberof_count
    )
    
    return {
        'missing_player_count': missing_player_count,
        'missing_player_ratio': missing_player_ratio,
        'missing_position_count': missing_position_count,
        'missing_position_ratio': missing_position_ratio,
        'missing_memberof_count': missing_memberof_count,
        'missing_memberof_ratio': missing_memberof_ratio,
        'total_missing_connections': total_missing_connections  # Total missing connections
    }




def find_inconsistent_edges(edges):
    # Dizionari per tracciare le associazioni
    tracker_to_player = {}  # Tracker -> Player per la relazione "is"
    tracker_to_team = {}  # Tracker -> Team per la relazione "memberof"
    tracker_to_position_time = {}  # Tracker -> (minuto, secondo) per la posizione
    position_time_to_trackers = {}  # (minuto, secondo) -> Trackers per la relazione "has"
    
    # Insiemi per tracciare gli archi incoerenti
    inconsistent_is_edges = set()
    inconsistent_memberof_edges = set()
    inconsistent_position_edges = set()
    inconsistent_has_position_edges = set()  # Nuova incoerenza per più Tracker con "has" nello stesso minuto e secondo

    # Itera su ogni arco nel grafo
    for edge in edges:
        _from_type, _from_id = edge['_from'].split('/')
        _to_type, _to_id = edge['_to'].split('/')
        
        # Incoerenza 1: Tracker associato a più Player con "is"
        if _from_type == 'Tracker' and _to_type == 'Player' and edge['relation'] == 'is':
            if _from_id in tracker_to_player:
                # Se il Tracker è già associato a un altro Player, aggiungi l'arco incoerente
                if tracker_to_player[_from_id] != _to_id:
                    inconsistent_is_edges.add(edge['_id'])
            else:
                tracker_to_player[_from_id] = _to_id
        
        # Incoerenza 2: Tracker associato a più Team con "memberof"
        if (_from_type == 'Tracker' and _to_type == 'Team' and edge['relation'] == 'memberof'):
            if _from_id in tracker_to_team:
                # Se il Tracker è già associato a un altro Team, aggiungi l'arco incoerente
                if tracker_to_team[_from_id] != _to_id:
                    inconsistent_memberof_edges.add(edge['_id'])
            else:
                tracker_to_team[_from_id] = _to_id
        
        # Incoerenza 3: Tracker associato a più Position con stesso minuto e secondo
        if _from_type == 'Tracker' and _to_type == 'Position':
            time_key = (edge['minutes'], edge['seconds'])
            if _from_id in tracker_to_position_time:
                # Se esiste già una posizione per lo stesso minuto e secondo, aggiungi incoerenza
                if time_key in tracker_to_position_time[_from_id]:
                    inconsistent_position_edges.add(edge['_id'])
                else:
                    tracker_to_position_time[_from_id].add(time_key)
            else:
                tracker_to_position_time[_from_id] = {time_key}
        
        # Nuova Incoerenza 4: Più Tracker con "has" nello stesso minuto e secondo
        if edge['relation'] == 'has':
            time_key = (edge['minutes'], edge['seconds'])
            if time_key in position_time_to_trackers:
                position_time_to_trackers[time_key].add(_from_id)
            else:
                position_time_to_trackers[time_key] = {_from_id}
    
    for time_key, trackers in position_time_to_trackers.items():
        if len(trackers) > 1:
            # Se ci sono più tracker associati allo stesso minuto e secondo
            for tracker in trackers:
                inconsistent_has_position_edges.add(tracker)
    
    # Totale degli archi
    total_edges = len(edges)
    
    # Conteggio degli archi incoerenti per ciascuna tipologia
    inconsistent_is_count = len(inconsistent_is_edges)
    inconsistent_memberof_count = len(inconsistent_memberof_edges)
    inconsistent_position_count = len(inconsistent_position_edges)
    inconsistent_has_position_count = len(inconsistent_has_position_edges)
    
    # Calcolo dei ratio di incoerenza per ogni tipo di incoerenza
    inconsistent_is_ratio = inconsistent_is_count / total_edges if total_edges > 0 else 0
    inconsistent_memberof_ratio = inconsistent_memberof_count / total_edges if total_edges > 0 else 0
    inconsistent_position_ratio = inconsistent_position_count / total_edges if total_edges > 0 else 0
    inconsistent_has_position_ratio = inconsistent_has_position_count / total_edges if total_edges > 0 else 0
    
    # Calcolo del ratio complessivo di incoerenza rispetto al totale degli archi
    total_inconsistent_edges = (inconsistent_is_count + inconsistent_memberof_count +
                                inconsistent_position_count + inconsistent_has_position_count)
    overall_inconsistent_ratio = total_inconsistent_edges / total_edges if total_edges > 0 else 0
    
    # Calcolo dei ratio di coerenza (1 - incoerenza)
    consistent_is_ratio = 1 - inconsistent_is_ratio
    consistent_memberof_ratio = 1 - inconsistent_memberof_ratio
    consistent_position_ratio = 1 - inconsistent_position_ratio
    consistent_has_position_ratio = 1 - inconsistent_has_position_ratio
    overall_consistent_ratio = 1 - overall_inconsistent_ratio
    
    return {
        'inconsistent_player_count': inconsistent_is_count,
        'inconsistent_memberof_count': inconsistent_memberof_count,
        'inconsistent_position_count': inconsistent_position_count,
        'inconsistent_has_possession_count': inconsistent_has_position_count,
        'total_inconsistent_edges': total_inconsistent_edges,
        'consistent_player_ratio': consistent_is_ratio,
        'consistent_memberof_ratio': consistent_memberof_ratio,
        'consistent_position_ratio': consistent_position_ratio,
        'consistent_has_possession_ratio': consistent_has_position_ratio,
        'overall_consistent_ratio': overall_consistent_ratio
    }


def find_max_time(edges):
    max_minutes = 0
    max_seconds = 0

    for edge in edges:
        if 'minutes' in edge and 'seconds' in edge:
            minutes = int(edge['minutes'])
            seconds = int(edge['seconds'])
            
            
            if minutes > max_minutes or (minutes == max_minutes and seconds > max_seconds):
                max_minutes = minutes
                max_seconds = seconds

    return {
        'minutes': max_minutes,
        'seconds': max_seconds
    }

def calculate_match_stats(edges):
    tracker_interactions = defaultdict(int)  # Traccia le interazioni per ogni tracker
    tracker_possession_time = defaultdict(int)  # Tempo di possesso palla per ogni tracker
    tracker_team = {}  # Traccia il team di ogni tracker
    tracker_player = {}  # Traccia il player associato a ogni tracker

    # Ciclo attraverso gli archi per calcolare le statistiche
    for edge in edges:
        _from = edge['_from'].split('/')[1]
        _to = edge['_to'].split('/')[1]
        relation = edge['relation']

        # Associa i tracker ai team
        if relation == 'memberof' and 'Tracker' in edge['_from'] and 'Team' in edge['_to']:
            tracker_team[_from] = edge['_to'].split('/')[1]  # Salva il team di appartenenza del tracker

        # Associa i tracker ai player
        if relation == 'is' and 'Tracker' in edge['_from'] and 'Player' in edge['_to']:
            tracker_player[_from] = _to  # Salva il player associato al tracker

        # Traccia le interazioni totali dei tracker
        if 'Tracker' in edge['_from'] and 'Tracker' in edge['_to']:
            tracker_interactions[_from] += 1
            tracker_interactions[_to] += 1

        # Tempo di possesso palla (per relazioni 'has' con minuti e secondi)
        if relation == 'has' and 'Tracker' in edge['_from'] and 'Ball' in edge['_to']:
            possession_time = edge['minutes'] * 60 + edge['seconds']
            tracker_possession_time[_from] += possession_time

    # Statistiche specifiche
    # 1. Tracker con più interazioni
    if tracker_interactions:
        most_interactions_tracker = max(tracker_interactions, key=tracker_interactions.get)
        most_interactions_tracker_count = tracker_interactions[most_interactions_tracker]
    else:
        most_interactions_tracker = None
        most_interactions_tracker_count = 0

    # 2. Tracker con il maggior tempo di possesso palla
    if tracker_possession_time:
        most_possession_tracker = max(tracker_possession_time, key=tracker_possession_time.get)
        most_possession_tracker_time = tracker_possession_time[most_possession_tracker]
        most_possession_tracker_team = tracker_team.get(most_possession_tracker, 'Unknown')
    else:
        most_possession_tracker = None
        most_possession_tracker_time = 0
        most_possession_tracker_team = None

    # 3. Team con più interazioni totali
    team_interactions = defaultdict(int)
    for tracker, interactions in tracker_interactions.items():
        if tracker in tracker_team:
            team = tracker_team[tracker]
            team_interactions[team] += interactions

    if team_interactions:
        most_interactions_team = max(team_interactions, key=team_interactions.get)
        most_interactions_team_count = team_interactions[most_interactions_team]
    else:
        most_interactions_team = None
        most_interactions_team_count = 0

    # 4. Player associato a un tracker con più interazioni tra i tracker associati ai player
    player_interactions = {tracker: tracker_interactions[tracker] for tracker in tracker_player if tracker in tracker_interactions}
    if player_interactions:
        most_interactions_player_tracker = max(player_interactions, key=player_interactions.get)
        most_interactions_player = tracker_player[most_interactions_player_tracker]
        most_interactions_player_count = player_interactions[most_interactions_player_tracker]
    else:
        most_interactions_player_tracker = None
        most_interactions_player = None
        most_interactions_player_count = 0

    # 5. Player associato a un tracker con il maggior tempo di possesso palla tra i tracker associati ai player
    player_possession = {tracker: tracker_possession_time[tracker] for tracker in tracker_player if tracker in tracker_possession_time}
    if player_possession:
        most_possession_player_tracker = max(player_possession, key=player_possession.get)
        most_possession_player = tracker_player[most_possession_player_tracker]
        most_possession_player_time = player_possession[most_possession_player_tracker]
    else:
        most_possession_player_tracker = None
        most_possession_player = None
        most_possession_player_time = 0

    # Restituisci tutte le statistiche calcolate
    return {
        'most_interactions_tracker': most_interactions_tracker,
        'most_interactions_tracker_count': most_interactions_tracker_count,
        'most_interactions_player': most_interactions_player,
        'most_interactions_player_count': most_interactions_player_count,
        'most_possession_tracker': most_possession_tracker,
        'most_possession_time_seconds': most_possession_tracker_time,
        'most_possession_tracker_team': most_possession_tracker_team,
        'most_possession_player': most_possession_player,
        'most_possession_player_time': most_possession_player_time,
        'most_interactions_team': most_interactions_team,
        'most_interactions_team_count': most_interactions_team_count,
    }


from collections import defaultdict

def postprocess_team_consistency(edges,db,graph_name):

    if db.has_graph(graph_name):
        graph = db.graph(graph_name)
    else:
        graph = db.create_graph(graph_name)
    if graph.has_edge_definition("memberof"):
        edge_def = graph.edge_collection("memberof")
    else:
        edge_def = graph.create_edge_definition(
        edge_collection="memberof",
        from_vertex_collections=["Tracker"],
        to_vertex_collections=["Team"]
    )
    # Dizionario per tracciare tutte le associazioni di ciascun Tracker ai team
    tracker_team_counts = defaultdict(lambda: defaultdict(int))
    
    # Itera su tutti gli archi per contare le associazioni di ciascun Tracker ai team
    for edge in edges:
        _from_type, _from_id = edge['_from'].split('/')
        _to_type, _to_id = edge['_to'].split('/')
        
        # Consideriamo solo gli archi "memberof" tra Tracker e Team
        if _from_type == 'Tracker' and _to_type == 'Team' and edge['relation'] == 'memberof':
            # Aumenta il conteggio del team associato a questo tracker
            tracker_team_counts[_from_id][_to_id] += 1
    
    # Dizionario per tracciare il team di maggioranza per ciascun tracker
    tracker_majority_team = {}
    
    # Determina il team di maggioranza per ciascun tracker
    for tracker, team_counts in tracker_team_counts.items():
        # Trova il team con il numero massimo di associazioni
        majority_team = max(team_counts, key=team_counts.get)
        tracker_majority_team[tracker] = majority_team
    
    # Crea una nuova lista di archi post-processati
    postprocessed_edges = []
    
    # Itera su tutti gli archi per correggere le inconsistenze di appartenenza ai team
    for edge in edges:
        _from_type, _from_id = edge['_from'].split('/')
        _to_type, _to_id = edge['_to'].split('/')
        
        # Se l'arco è una relazione "memberof" e c'è una maggioranza da applicare
        if _from_type == 'Tracker' and _to_type == 'Team' and edge['relation'] == 'memberof':
            majority_team = tracker_majority_team.get(_from_id)
            
            # Se il team dell'arco non corrisponde al team di maggioranza, lo correggiamo
            if majority_team and _to_id != majority_team:
                corrected_edge = edge.copy()
                corrected_edge['_to'] = f'Team/{majority_team}'
                postprocessed_edges.append(corrected_edge)
                print("\n\n Corrected edge")
                print(corrected_edge)
                edge_def.delete({
                '_key' : f'{edge["_key"]}',
                '_id'  : f'{edge["_id"]}',
                '_from': f'{corrected_edge["_from"]}',
                '_to': f'{edge["_to"]}',
                'relation':f'{corrected_edge["relation"]}' ,
                'minutes': f'{corrected_edge["minutes"]}',
                'seconds': f'{corrected_edge["seconds"]}'
                })
                edge_def.insert({
                '_from': f'{corrected_edge["_from"]}',
                '_to': f'{corrected_edge["_to"]}',
                'relation':f'{corrected_edge["relation"]}' ,
                'minutes': f'{corrected_edge["minutes"]}',
                'seconds': f'{corrected_edge["seconds"]}'
                })

    return postprocessed_edges

def postprocess_tracker_player_consistency(edges, db, graph_name):
    # Controllo se il grafo esiste e creazione se non esiste
    if db.has_graph(graph_name):
        graph = db.graph(graph_name)
    else:
        graph = db.create_graph(graph_name)
    
    # Controllo se la definizione degli archi "is" esiste
    if graph.has_edge_definition("is"):
        edge_def = graph.edge_collection("is")
    else:
        edge_def = graph.create_edge_definition(
            edge_collection="is",
            from_vertex_collections=["Tracker"],
            to_vertex_collections=["Player"]
        )
    
    # Dizionario per tracciare tutte le associazioni di ciascun Tracker ai Player
    tracker_player_counts = defaultdict(lambda: defaultdict(int))
    
    # Itera su tutti gli archi per contare le associazioni di ciascun Tracker ai Player
    for edge in edges:
        _from_type, _from_id = edge['_from'].split('/')
        _to_type, _to_id = edge['_to'].split('/')
        
        # Consideriamo solo gli archi "is" tra Tracker e Player
        if _from_type == 'Tracker' and _to_type == 'Player' and edge['relation'] == 'is':
            # Aumenta il conteggio del Player associato a questo Tracker
            tracker_player_counts[_from_id][_to_id] += 1
    
    # Dizionario per tracciare il Player di maggioranza per ciascun Tracker
    tracker_majority_player = {}
    
    # Determina il Player di maggioranza per ciascun Tracker
    for tracker, player_counts in tracker_player_counts.items():
        # Trova il Player con il numero massimo di associazioni
        if player_counts:
            majority_player = max(player_counts, key=player_counts.get)
            tracker_majority_player[tracker] = majority_player
    
    # Crea una nuova lista di archi post-processati
    postprocessed_edges = []
    
    # Itera su tutti gli archi per correggere le inconsistenze di associazione ai Player
    for edge in edges:
        _from_type, _from_id = edge['_from'].split('/')
        _to_type, _to_id = edge['_to'].split('/')
        
        # Se l'arco è una relazione "is" e c'è una maggioranza da applicare
        if _from_type == 'Tracker' and _to_type == 'Player' and edge['relation'] == 'is':
            majority_player = tracker_majority_player.get(_from_id)
            
            # Se il Player dell'arco non corrisponde al Player di maggioranza, lo correggiamo
            if majority_player and _to_id != majority_player:
                corrected_edge = edge.copy()
                corrected_edge['_to'] = f'Player/{majority_player}'
                postprocessed_edges.append(corrected_edge)
                
                print("\n\n Corrected edge")
                print(corrected_edge)

                # Eliminazione dell'arco esistente
                edge_def.delete({
                    '_key': edge["_key"],
                    '_id': edge["_id"],
                    '_from': edge["_from"],
                    '_to': edge["_to"],
                    'relation': edge["relation"],
                    'minutes': edge.get("minutes"),
                    'seconds': edge.get("seconds")
                })
                
                # Inserimento dell'arco corretto
                edge_def.insert({
                    '_from': corrected_edge["_from"],
                    '_to': corrected_edge["_to"],
                    'relation': corrected_edge["relation"],
                    'minutes': corrected_edge.get("minutes"),
                    'seconds': corrected_edge.get("seconds")
                })

    return postprocessed_edges


from collections import defaultdict

def create_tracker_player_position_relationships(edges, db, graph_name):
    # Controlla se il grafo esiste
    if db.has_graph(graph_name):
        graph = db.graph(graph_name)
    else:
        graph = db.create_graph(graph_name)

    # Controlla se la definizione degli archi "is" esiste
    if not graph.has_edge_definition("is"):
        edge_def = graph.create_edge_definition(
            edge_collection="is",
            from_vertex_collections=["Tracker"],
            to_vertex_collections=["Player"]
        )
    else:
        edge_def = graph.edge_collection("is")

    # Dizionario per tracciare le associazioni Tracker -> Player
    tracker_player_map = defaultdict(list)

    # Mappiamo i Tracker ai Player attraverso gli archi "is"
    for edge in edges:
        _from_type, _from_id = edge['_from'].split('/')
        _to_type, _to_id = edge['_to'].split('/')

        if _from_type == 'Tracker' and _to_type == 'Player' and edge['relation'] == 'is':
            tracker_player_map[_from_id].append(_to_id)

    # Lista per archi da inserire
    inserted_edges = []
    existing_edges = set()  # Set per tenere traccia degli archi esistenti

    # Recupera gli archi esistenti "is" per il controllo
    for edge in edge_def:
        _from_type, _from_id = edge['_from'].split('/')
        _to_type, _to_id = edge['_to'].split('/')
        existing_edges.add((f'Tracker/{_from_id}', f'Player/{_to_id}', edge['minutes'], edge['seconds']))

    # Per ogni Tracker, otteniamo le sue posizioni e creiamo le relazioni
    for tracker_id, players in tracker_player_map.items():
        for player_id in players:
            # Recupera tutte le posizioni del Tracker
            for edge in edges:
                _from_type, _from_id = edge['_from'].split('/')
                _to_type, _to_id = edge['_to'].split('/')

                if _from_type == 'Tracker' and _from_id == tracker_id and _to_type == 'Pos':
                    # Creiamo una nuova relazione 'is' per ogni istante
                    new_edge = {
                        '_from': f'Tracker/{tracker_id}',
                        '_to': f'Player/{player_id}',
                        'relation': 'is',
                        'minutes': edge['minutes'],
                        'seconds': edge['seconds']
                    }
                    
                    # Controlla se l'arco esiste già
                    if (new_edge['_from'], new_edge['_to'], new_edge['minutes'], new_edge['seconds']) not in existing_edges:
                        # Inserisci nel grafo
                        edge_def.insert(new_edge)
                        inserted_edges.append(new_edge)
                        existing_edges.add((new_edge['_from'], new_edge['_to'], new_edge['minutes'], new_edge['seconds']))

                        print(f'Inserted edge from {new_edge["_from"]} to {new_edge["_to"]} at {new_edge["minutes"]}:{new_edge["seconds"]}')

    return inserted_edges  # Restituisce gli archi inseriti

def main():
    #encodedCA = "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSURHVENDQWdHZ0F3SUJBZ0lSQUtRZm9RemFHTzRYRGphRGNkOXBCRWN3RFFZSktvWklodmNOQVFFTEJRQXcKSmpFUk1BOEdBMVVFQ2hNSVFYSmhibWR2UkVJeEVUQVBCZ05WQkFNVENFRnlZVzVuYjBSQ01CNFhEVEkwTURreApPREU1TkRrMU4xb1hEVEk1TURreE56RTVORGsxTjFvd0pqRVJNQThHQTFVRUNoTUlRWEpoYm1kdlJFSXhFVEFQCkJnTlZCQU1UQ0VGeVlXNW5iMFJDTUlJQklqQU5CZ2txaGtpRzl3MEJBUUVGQUFPQ0FROEFNSUlCQ2dLQ0FRRUEKdk5xRG5oeXJDQXFGUi91OXRDL1I1MCtPM3BnTjhRZjJWMGorMFV0MXVsMnQwQ0NkOVhhUkh6UTFLc1dJSTJIaAowY3BNNXhnTDRCZXJWeHQyY1FhaXZkYUN4Szloc0tneHhwWG5ydGk0ZXZ4ck83VkFucVZUSVpHa2ppVG5zNEZrCnBmRXYwaDBZc3RhYWI1RUFCY2owK0VaWVMyN3NqenpvcHh4NkFJKzZxaUNPblp6TlZ5V1NiU3pCTjgyQm1HelYKMmNVRGFLTFc1QnVSQkhqUWd2d3FBMmF5N0hSdWFwd09HU1MvOFhya0k1QjJESExSYmw2ck42dmZlWkFSeG15dQpHdzltVVJsRElGcXMrdEtHcS9vRkVyVFVIdmdydjN6OTg2eXdRN2p5V2U5cENtYkxkVzNzSU8wYnhoSnY1T0lmCm0xTFovRFdXUDBBQjNBNnV5d1o2M1FJREFRQUJvMEl3UURBT0JnTlZIUThCQWY4RUJBTUNBcVF3RHdZRFZSMFQKQVFIL0JBVXdBd0VCL3pBZEJnTlZIUTRFRmdRVXNHVHVYSnNldDZlTlltaVQ5bXE1bmo2MGI0Y3dEUVlKS29aSQpodmNOQVFFTEJRQURnZ0VCQUM3MVcxMk1mQmtFL0JhZHJzZjlHWVVCVWc5cEcxc0RyS1kwcTdhcHc2OS96d01XCmQycnRSSXREQjJ1UUM0WXV3a2p3dVg2STVGcEF2SDNDalBRRGtQR2dJWS9aR1dkYVpoN3lqT3dOU1l4eThMb3cKUHpaaWh5UGZ2SVpIZi81cnZIUzBjYm84TUJUZFZqdXphZGg3UmNOWWQ3WUdPRnFTaXE4NlF3UFpScWpCZEt5cwplTGxxcXFaNDI3U3NNWk9sSm1BaWhtK3QwSFNsbWp4T3FoMkZycFRSb3lsZUhaYVNMU0VZcUowWURiK3R0V1BICjNVMVNKYnROUXM0ajlndk0rTFBJVmpkUU04T0hHMnREQlhKdHNtTVoySzk0ZEFhTWt0U2dvajdEeDJDTnhQNEkKNEJBd054cG4rakZnZFFWRVVlWE9kek9pZkZTR3pmQXVwRTBDMjhzPQotLS0tLUVORCBDRVJUSUZJQ0FURS0tLS0tCg=="
    #db_host = "https://47711e33bfa7.arangodb.cloud:18529"
    #db_password = "QNeu1YDXU1sePpoGRn6D"
    #db_username = "root"


    encodedCA = "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSURHVENDQWdHZ0F3SUJBZ0lSQUo4UTI5UUtkVWxiKzBweFd2OXhuOGt3RFFZSktvWklodmNOQVFFTEJRQXcKSmpFUk1BOEdBMVVFQ2hNSVFYSmhibWR2UkVJeEVUQVBCZ05WQkFNVENFRnlZVzVuYjBSQ01CNFhEVEkwTVRBdwpNVEV4TURneE1Gb1hEVEk1TURrek1ERXhNRGd4TUZvd0pqRVJNQThHQTFVRUNoTUlRWEpoYm1kdlJFSXhFVEFQCkJnTlZCQU1UQ0VGeVlXNW5iMFJDTUlJQklqQU5CZ2txaGtpRzl3MEJBUUVGQUFPQ0FROEFNSUlCQ2dLQ0FRRUEKc0JGS1RFN01FYkEwaDJza2VyblFsOG1RRlQvaHVma3Vscm9mRVdsSjFPbkt3ckhLNTZtRVNIQnl3dC8wMXRYRgp4Ukp3K1JSUDRYWm5ZY3RlY0hOdzdiT1hQdkdRMy9RTDF1WEpudGRRVkJ0MlN2R2dvNXRVOWJJWkxNaS9hMDNKCld6WFFKSE1oNHJ3akhqbzBOemlJbEdhdWhBSTRDaHphekNvRUQ5eDlUQmJTUzdyeFNZMjJrK1VDRW81SitCeCsKblkwckNnaWszVGR1bHBUSG8zbG9YV2FZSVhOa1ZZaktrbFBvNVlNRmNrbXJSUlcwRzFyM2g5eno0SGxTakRWagpWWTJWYmR2NXJjM3BIK3RoTnQ1ZDJXRzUzQTNFUXFWb3lWbWVrS2xqS0YrTGQzN2YzMnZpZHRTaGphSVFmT1BlCjAvaFhUZmZJTkZGYTc5MW5zZy9sc1FJREFRQUJvMEl3UURBT0JnTlZIUThCQWY4RUJBTUNBcVF3RHdZRFZSMFQKQVFIL0JBVXdBd0VCL3pBZEJnTlZIUTRFRmdRVWZhbVcwVGtEaTlqWUFIejJROG5IQ0g3VWlRd3dEUVlKS29aSQpodmNOQVFFTEJRQURnZ0VCQUtFem1RWGJLQ296bkUrTEZod0EwQ3YrdkJmRnNQTERoN3V4dUYrcUpwNjM4U25ECmliWW1HOEduYTdrS2kwUzBNRDF5aGZRTCsxOVlBWG0wQ0xIcWNMVUt3ZWtmd3B2dXBPNHE2dzR5K3NuYytPOVgKNnE1ZmREYkNERk5xdGdxVThzbDV3M0tTQVNuZmY3anFpRG5aQ1NvZWpBZUkrMVhMQVpjQ3htS1VtRjlGdkdyTgp0RVM1YjBUUDJUTzQ3WHFoUEZNOWg3WDg4RWs0Wllqbm16L0pEOTI5Z3B3dUJETFYybDRjdWZpTmx6UnpKaktrCmJKcTh3SUd2V0VHdVZvOWcvSzIzRUpROGlRK3AwWHkxa3ZOQWNMN1UrUUNMd2wvekR0N056SjBZWFcybUNzVGYKem1KU2lpOHdSVWNRSWcrUTU4UEI3TGNpSDRpV3FERDgzK3Z6bFNVPQotLS0tLUVORCBDRVJUSUZJQ0FURS0tLS0tCg=="
    db_host = "https://fd994eab500a.arangodb.cloud:18529"
    db_password = "9IVPyXZ8unOIE3PCCLGa"
    db_username = "root"


    graph_name="match_3"
    # Connect to ArangoDB
    sys_db, client = connect_to_arangodb(encodedCA, db_host, db_username, db_password)
    db = client.db(graph_name, username=db_username, password=db_password)
    # Get graph information
    grafo=get_graph(client, db_username,db_password,graph_name, graph_name)
    print("Video name:"+graph_name)
    max=find_max_time(grafo)
    print("Video length:")
    print(max)
    grafo2=get_graph2(client, db_username,db_password,graph_name, graph_name)
    result = analyze_graph(grafo2)
    print("Node Count:"+str(result["node_count"]))
    print("Edge Count:"+str(result["edge_count"]))
    print("Max Degree Node:"+str(result["max_degree_node"]))
    print("Category Count:"+str(result["category_count"]))
    print("Edge Category Count:"+str(result["edge_category_count"]))
    result_lacune = find_graph_gaps(grafo)
    print(result_lacune)
    inconsistenza=find_inconsistent_edges(grafo)
    print(inconsistenza)
    stats=calculate_match_stats(grafo)
    print(stats)
    plot_graph_analysis(result,'./plots/'+graph_name)
    


    postprocess_team_consistency(grafo,db,graph_name)
    postprocess_tracker_player_consistency(grafo,db,graph_name)
    grafo=get_graph(client, db_username,db_password,graph_name, graph_name)

    create_tracker_player_position_relationships(grafo,db,graph_name)
if __name__ == "__main__":
    main()
