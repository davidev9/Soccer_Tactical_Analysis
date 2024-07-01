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
        # Decode the certificate
        file_content = base64.b64decode(encodedCA)
        
        # Write the certificate to a file
        with open("cert_file.crt", "w+") as f:
            f.write(file_content.decode("utf-8"))
    except Exception as e:
        print(f"Error decoding and writing certificate: {str(e)}")
        sys.exit(1)

    # Initialize the ArangoDB client
    client = ArangoClient(
        hosts=db_host, verify_override="cert_file.crt"
    )

    # Connect to the '_system' database as the specified user
    try:
        sys_db = client.db("_system", username=db_username, password=db_password)
        print("ArangoDB version:", sys_db.version())
        return sys_db,client
    except Exception as e:
        print(f"Failed to connect to ArangoDB: {e}")
        sys.exit(1)

def initialize_database(connector, db_name, username, password, user=None):
    sys_db = connector.db('_system', username=username, password=password)
    
    # Check esistenza del db
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

def main():
    encodedCA = "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSURHVENDQWdHZ0F3SUJBZ0lSQU5pWS81VmlPMk82bXFpVG1HUDZ2RFF3RFFZSktvWklodmNOQVFFTEJRQXcKSmpFUk1BOEdBMVVFQ2hNSVFYSmhibWR2UkVJeEVUQVBCZ05WQkFNVENFRnlZVzVuYjBSQ01CNFhEVEkwTURZeQpOekV4TkRneU5sb1hEVEk1TURZeU5qRXhORGd5Tmxvd0pqRVJNQThHQTFVRUNoTUlRWEpoYm1kdlJFSXhFVEFQCkJnTlZCQU1UQ0VGeVlXNW5iMFJDTUlJQklqQU5CZ2txaGtpRzl3MEJBUUVGQUFPQ0FROEFNSUlCQ2dLQ0FRRUEKdHV1bEJWUmNMSDMrNndDVnYxejE1Ymh4c2Ixdko2NTM2cnlEb2g0Q0s3cmlZRmM0Y1JuVlNhN3NZaUhzMmEyagpaR2RPb2NiZWwzTExiNTlFR1hIWGhoSSs5U0RZUlFUdTRVckZwUjZaTk51S2VYOUp3d3NMWGZpSVdSM29JSjFrCk8vM2xVSzJUdFQ5azZOWGNCZUdnY2o0SEJLb0JtNVRrSnVUdDU5WHVWU1NHQzVqa05pNWZRaTJqaDBlMG5WcUQKSnFhU1VkVkdxUjZJaXdTL3pGQmlQZSsvOEFEYmxSeTljQUR4aDNFUmVSaGhJOVUrS2NBZm1CcFI1OGpyeU9PYQpNNUlyZlBhckxmYVhIbWJqQ1krZG1WVThPNTNkL1RRVi9obW15OGJINXVZTkJwamlTL2dBaVNwM3FDN1pOUm1TClRRUVh3cVduMXRYRFp5YTBKZWcwK1FJREFRQUJvMEl3UURBT0JnTlZIUThCQWY4RUJBTUNBcVF3RHdZRFZSMFQKQVFIL0JBVXdBd0VCL3pBZEJnTlZIUTRFRmdRVWx2RUdUK2FaUTV1SVhlTnZvd282VGtvODZzVXdEUVlKS29aSQpodmNOQVFFTEJRQURnZ0VCQUNJS21kR1lqbWhKdDBqZzhJY1lDUGZwL2I2Ny9scmlkKzVub3dZVXRPZWQycXRNCnpoM2pOd3JpNjIzaVFZOFd1anc4SmJpdWgvdTVlbWdCQUJZRGN3cFRDZFdzUUVsemxtdVFuNGV2bHVVVkxZSlUKMVVFTGVpclRxb2g3YjlYdXFUbnBuaG9tdFdLdW9NZm9sOEVJZ2FOTTVZR3Q4UkhUQW9hRVdyWDNyR0phMmJoKwpOQUJkQzJmWUp0MWFsaFc5U3JaT1N6dis0UWN2WTBqa0hpaXpibDFEZ3RGYjcxaEdSeS96UnhORkNrZWp0V2ZQCkR2UFR4OEppNzlNZjNXMnRxcE05UlVQS2pkOUJrTUxvTFZkbTJsM0VENm5kWjhvZkgyNjVCQ2ZRWnpOQ0JsNlMKN2xES2FpdTVtejFLaTVFaGlYSUlLS1Uzc1NDOHFoYWlVSmt1MWhVPQotLS0tLUVORCBDRVJUSUZJQ0FURS0tLS0tCg=="
    db_host = "https://1be362bd9a71.arangodb.cloud:18529"
    db_username = "root"
    db_password = "VEzSjRTobBNejFOvRbQg"

    # Connect to ArangoDB
    sys_db,client = connect_to_arangodb(encodedCA, db_host, db_username, db_password)

    # Initialize the "test" database
    db = initialize_database(client, "test", db_username, db_password)

    # Parameters for the triplet
    entity1_name = "Alice"
    category1 = "students"
    entity2_name = "Math101"
    category2 = "lectures"
    relation_name = "enrolled_in"
    minutes = 90
    seconds = 0

    # Push the triplet into the graph
    push_triplet(db, "school", entity1_name, category1, entity2_name, category2, relation_name, minutes, seconds)

    # Verify the inserted data
    school_graph = db.graph("school")
    print(school_graph.vertex_collection("students").all())
    print(school_graph.vertex_collection("lectures").all())
    print(school_graph.edge_collection("enrolled_in").all())

if __name__ == "__main__":
    main()