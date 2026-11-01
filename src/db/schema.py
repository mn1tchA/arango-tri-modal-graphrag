from arango import ArangoClient
from arango.database import StandardDatabase

DB_NAME = "tri_modal_graphrag"
GRAPH_NAME = "knowledge_graph"
VIEW_NAME = "hybrid_search_view"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension


def init_database(host: str = "http://localhost:8529") -> StandardDatabase:
    client = ArangoClient(hosts=host)
    sys_db = client.db("_system")

    if not sys_db.has_database(DB_NAME):
        sys_db.create_database(DB_NAME)

    db = client.db(DB_NAME)

    # 1. Create Vertex Collections
    for vertex_col in ["chunks", "entities"]:
        if not db.has_collection(vertex_col):
            db.create_collection(vertex_col, edge=False)

    # 2. Create Edge Collections
    for edge_col in ["mentions", "relations"]:
        if not db.has_collection(edge_col):
            db.create_collection(edge_col, edge=True)

    # 3. Create Named Property Graph
    if not db.has_graph(GRAPH_NAME):
        graph = db.create_graph(GRAPH_NAME)
        graph.create_edge_definition(
            edge_collection="mentions",
            from_vertex_collections=["chunks"],
            to_vertex_collections=["entities"],
        )
        graph.create_edge_definition(
            edge_collection="relations",
            from_vertex_collections=["entities"],
            to_vertex_collections=["entities"],
        )

    # 4. Create Persistent Indexes
    chunks_col = db.collection("chunks")
    chunk_index_fields = [idx.get("fields") for idx in chunks_col.indexes()]
    if ["content_hash"] not in chunk_index_fields:
        chunks_col.add_persistent_index(fields=["content_hash"], unique=True)

    entities_col = db.collection("entities")
    entity_index_fields = [idx.get("fields") for idx in entities_col.indexes()]
    if ["canonical_name", "entity_type"] not in entity_index_fields:
        entities_col.add_persistent_index(fields=["canonical_name", "entity_type"], unique=False)

    # 5. Create or Update ArangoSearch View for BM25 Full-Text Search
    existing_views = [v["name"] for v in db.views()]
    view_properties = {
        "links": {
            "chunks": {
                "fields": {
                    "text": {"analyzers": ["text_en"]},
                    "doc_source": {"analyzers": ["identity"]},
                }
            },
            "entities": {
                "fields": {
                    "canonical_name": {"analyzers": ["text_en", "identity"]},
                    "aliases": {"analyzers": ["text_en", "identity"]},
                    "description": {"analyzers": ["text_en"]},
                }
            },
        }
    }
    if VIEW_NAME not in existing_views:
        db.create_arangosearch_view(name=VIEW_NAME, properties=view_properties)
    else:
        db.update_arangosearch_view(name=VIEW_NAME, properties=view_properties)

    return db
