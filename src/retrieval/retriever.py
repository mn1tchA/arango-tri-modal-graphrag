"""
Tri-Modal AQL Retriever for ArangoDB.
General-purpose multi-modal retrieval engine unifying FAISS dense vector search,
ArangoSearch BM25 inverted lexical indexing with Reciprocal Rank Fusion (RRF),
and k-hop property graph traversals in a single database roundtrip.
"""

import time
from arango.database import StandardDatabase
from src.ingestion.seed_data import embed_text


class TriModalRetriever:
    def __init__(self, db: StandardDatabase):
        self.db = db

    def retrieve_vector_only(self, query: str, top_k: int = 2) -> dict:
        t0 = time.perf_counter()
        query_vec = embed_text(query)

        aql = """
        FOR doc IN chunks
            // Cosine similarity on L2-normalized vectors via dot product
            LET score = SUM(
                FOR i IN 0..(LENGTH(doc.embedding) - 1)
                    RETURN doc.embedding[i] * @query_vec[i]
            )
            SORT score DESC
            LIMIT @top_k
            RETURN {
                chunk_id: doc._key,
                source: doc.doc_source,
                text: doc.text,
                score: ROUND(score * 10000) / 10000
            }
        """
        cursor = self.db.aql.execute(aql, bind_vars={"query_vec": query_vec, "top_k": top_k})
        results = list(cursor)
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "mode": "Pure Vector Search",
            "latency_ms": round(latency_ms, 2),
            "chunks": results,
            "graph_paths": [],
            "discovered_entity_types": [],
            "discovered_entities": [],
        }

    def retrieve_hybrid_rrf(self, query: str, top_k: int = 2, rrf_k: int = 60) -> dict:
        t0 = time.perf_counter()
        query_vec = embed_text(query)

        aql = """
        // 1. Dense Vector Ranking
        LET vec_results = (
            FOR doc IN chunks
                LET v_score = SUM(
                    FOR i IN 0..(LENGTH(doc.embedding) - 1)
                        RETURN doc.embedding[i] * @query_vec[i]
                )
                SORT v_score DESC
                LIMIT 10
                RETURN doc._key
        )

        // 2. Sparse BM25 Ranking via ArangoSearch View
        LET bm25_results = (
            FOR doc IN hybrid_search_view
                SEARCH ANALYZER(doc.text IN TOKENS(@query_text, "text_en"), "text_en")
                FILTER IS_SAME_COLLECTION("chunks", doc)
                LET b_score = BM25(doc)
                SORT b_score DESC
                LIMIT 10
                RETURN doc._key
        )

        // 3. Reciprocal Rank Fusion (RRF) inside AQL
        FOR doc IN chunks
            LET v_pos = POSITION(vec_results, doc._key, true)
            LET b_pos = POSITION(bm25_results, doc._key, true)
            LET rrf_vec = v_pos != -1 ? (1.0 / (@rrf_k + v_pos + 1)) : 0.0
            LET rrf_bm25 = b_pos != -1 ? (1.0 / (@rrf_k + b_pos + 1)) : 0.0
            LET rrf_score = rrf_vec + rrf_bm25
            FILTER rrf_score > 0
            SORT rrf_score DESC
            LIMIT @top_k
            RETURN {
                chunk_id: doc._key,
                source: doc.doc_source,
                text: doc.text,
                score: ROUND(rrf_score * 10000) / 10000
            }
        """
        cursor = self.db.aql.execute(
            aql,
            bind_vars={"query_vec": query_vec, "query_text": query, "top_k": top_k, "rrf_k": rrf_k},
        )
        results = list(cursor)
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "mode": "Hybrid Search (Vector + BM25 RRF)",
            "latency_ms": round(latency_ms, 2),
            "chunks": results,
            "graph_paths": [],
            "discovered_entity_types": [],
            "discovered_entities": [],
        }

    def retrieve_tri_modal_graphrag(
        self,
        query: str,
        top_k: int = 2,
        max_hops: int = 1,
        min_confidence: float = 0.75,
    ) -> dict:
        t0 = time.perf_counter()
        query_vec = embed_text(query)

        aql = """
        // Step 1: Discover Top Anchor Chunks via Vector Similarity
        LET anchor_chunks = (
            FOR doc IN chunks
                LET v_score = SUM(
                    FOR i IN 0..(LENGTH(doc.embedding) - 1)
                        RETURN doc.embedding[i] * @query_vec[i]
                )
                SORT v_score DESC
                LIMIT @top_k
                RETURN {
                    chunk_id: doc._key,
                    source: doc.doc_source,
                    text: doc.text,
                    score: ROUND(v_score * 10000) / 10000,
                    _id: doc._id
                }
        )

        // Step 2: Find Anchor Entities mentioned in those chunks
        LET seed_entities = UNIQUE(
            FOR c IN anchor_chunks
                FOR ent IN 1..1 OUTBOUND c._id mentions
                    RETURN {
                        _id: ent._id,
                        canonical_name: ent.canonical_name,
                        entity_type: ent.entity_type
                    }
        )

        // Step 3: Multi-Hop Graph Traversal (1..max_hops) across Entity Relations
        LET traversed_subgraph = (
            FOR start_ent IN seed_entities
                FOR v, e, p IN 1..@max_hops ANY start_ent._id relations
                    OPTIONS { uniqueVertices: "path", bfs: true }
                    FILTER e.confidence >= @min_confidence
                    LET from_node = DOCUMENT(e._from)
                    LET to_node = DOCUMENT(e._to)
                    LET evidence_chunk = e.source_chunk ? (
                        IS_STRING(e.source_chunk) AND STARTS_WITH(e.source_chunk, "chunks/") 
                        ? DOCUMENT(e.source_chunk) 
                        : DOCUMENT(CONCAT("chunks/", e.source_chunk))
                    ) : null
                    RETURN DISTINCT {
                        "hop_ edge": CONCAT(from_node.canonical_name, " --[", e.relation_type, "]--> ", to_node.canonical_name),
                        evidence: e.evidence,
                        confidence: e.confidence,
                        from_name: from_node.canonical_name,
                        from_type: from_node.entity_type,
                        to_name: to_node.canonical_name,
                        to_type: to_node.entity_type,
                        bridged_chunk_id: evidence_chunk ? evidence_chunk._key : null,
                        bridged_chunk_text: evidence_chunk ? evidence_chunk.text : null,
                        bridged_source: evidence_chunk ? evidence_chunk.doc_source : null
                    }
        )

        RETURN {
            anchors: anchor_chunks,
            seed_entities: seed_entities,
            graph_paths: traversed_subgraph
        }
        """
        cursor = self.db.aql.execute(
            aql,
            bind_vars={
                "query_vec": query_vec,
                "top_k": top_k,
                "max_hops": max_hops,
                "min_confidence": min_confidence,
            },
        )
        raw = next(cursor)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Collect all discovered entity names and types from seeds and traversed paths
        discovered_entities_set = set()
        discovered_types_set = set()

        for s in raw.get("seed_entities", []):
            if s.get("canonical_name"):
                discovered_entities_set.add(s["canonical_name"])
            if s.get("entity_type"):
                discovered_types_set.add(s["entity_type"])

        for p in raw.get("graph_paths", []):
            if p.get("from_name"):
                discovered_entities_set.add(p["from_name"])
            if p.get("from_type"):
                discovered_types_set.add(p["from_type"])
            if p.get("to_name"):
                discovered_entities_set.add(p["to_name"])
            if p.get("to_type"):
                discovered_types_set.add(p["to_type"])

        discovered_entities = sorted(list(discovered_entities_set))
        discovered_entity_types = sorted(list(discovered_types_set))

        # Merge anchor chunks + newly discovered graph-bridged chunks
        seen_ids = {c["chunk_id"] for c in raw["anchors"]}
        combined_chunks = list(raw["anchors"])
        for path in raw["graph_paths"]:
            b_id = path.get("bridged_chunk_id")
            if b_id and b_id not in seen_ids:
                seen_ids.add(b_id)
                combined_chunks.append({
                    "chunk_id": b_id,
                    "source": path.get("bridged_source", "unknown"),
                    "text": path.get("bridged_chunk_text", ""),
                    "score": f"Graph-Bridged (conf={path['confidence']})",
                })

        return {
            "mode": f"Tri-Modal GraphRAG (Vector + {max_hops}-Hop Subgraph)",
            "latency_ms": round(latency_ms, 2),
            "chunks": combined_chunks,
            "graph_paths": raw["graph_paths"],
            "discovered_entity_types": discovered_entity_types,
            "discovered_entities": discovered_entities,
        }
