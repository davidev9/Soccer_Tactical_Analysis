from arango import ArangoClient
import base64
import sys

class Triplet:
    def __init__(self, entity1_name, category1, entity2_name, category2, relation_name, minutes, seconds):
        self.entity1_name = entity1_name
        self.category1 = category1
        self.entity2_name = entity2_name
        self.category2 = category2
        self.relation_name = relation_name
        self.minutes = minutes
        self.seconds = seconds


def connect_to_arangodb(encodedCA, db_host, db_username, db_password):
    try:
        file_content = base64.b64decode(encodedCA)
        
        with open("cert_file.crt", "w+") as f:
            f.write(file_content.decode("utf-8"))
    except Exception as e:
        print(f"Error decoding and writing certificate: {str(e)}")
        sys.exit(1)

    client = ArangoClient(
        hosts=db_host, verify_override="cert_file.crt"
    )

    try:
        sys_db = client.db("_system", username=db_username, password=db_password)
        print("ArangoDB version:", sys_db.version())
        return sys_db,client
    except Exception as e:
        print(f"Failed to connect to ArangoDB: {e}")
        sys.exit(1)

def initialize_database(connector, db_name, username, password, user=None):
    sys_db = connector.db('_system', username=username, password=password)
    
    if not sys_db.has_database(db_name):
        # Create the database with the specified user if it does not exist
        if user:
            sys_db.create_database(
                name=db_name,
                users=[user]
            )
        else:
            sys_db.create_database(name=db_name)
    
    # Ritorna l'oggetto database
    db = connector.db(db_name, username=username, password=password)
    return db

def push_triplet(db, graph_name, triplet):
    if db.has_graph(graph_name):
        graph = db.graph(graph_name)
    else:
        graph = db.create_graph(graph_name)
    
    if graph.has_vertex_collection(triplet.category1):
        collection1 = graph.vertex_collection(triplet.category1)
    else:
        collection1 = graph.create_vertex_collection(triplet.category1)
    
    if graph.has_vertex_collection(triplet.category2):
        collection2 = graph.vertex_collection(triplet.category2)
    else:
        collection2 = graph.create_vertex_collection(triplet.category2)
    
    if not collection1.has(triplet.entity1_name):
        collection1.insert({'_key': triplet.entity1_name, 'name': triplet.entity1_name})
    
    if not collection2.has(triplet.entity2_name):
        collection2.insert({'_key': triplet.entity2_name, 'name': triplet.entity2_name})
    
    if graph.has_edge_definition(triplet.relation_name):
        edge_def = graph.edge_collection(triplet.relation_name)
    else:
        edge_def = graph.create_edge_definition(
            edge_collection=triplet.relation_name,
            from_vertex_collections=[triplet.category1],
            to_vertex_collections=[triplet.category2]
        )
    
    edge_def.insert({
        '_from': f'{triplet.category1}/{triplet.entity1_name}',
        '_to': f'{triplet.category2}/{triplet.entity2_name}',
        'relation': triplet.relation_name,
        'minutes': triplet.minutes,
        'seconds': triplet.seconds
    })

