"""
Incremental SHA-256 Graph Ingestion Pipeline.
Coordinates content hash verification, dense embedding generation, structured entity extraction,
and hybrid canonical entity resolution with zero redundant processing.
"""

import hashlib
import time
from typing import Optional
from arango.database import StandardDatabase
from src.ingestion.extractor import StructuredGraphExtractor
from src.ingestion.entity_resolver import HybridEntityResolver
from src.ingestion.seed_data import embed_text


class GraphIngestionPipeline:
    def __init__(
        self,
        db: StandardDatabase,
        extractor: Optional[StructuredGraphExtractor] = None,
        resolver: Optional[HybridEntityResolver] = None,
    ):
        self.db = db
        self.extractor = extractor or StructuredGraphExtractor(mode="deterministic")
        self.resolver = resolver or HybridEntityResolver(db=db)
        self.chunks_col = db.collection("chunks")
        self.entities_col = db.collection("entities")
        self.mentions_col = db.collection("mentions")
        self.relations_col = db.collection("relations")

    def truncate_all_collections(self):
        for col in [self.chunks_col, self.entities_col, self.mentions_col, self.relations_col]:
            col.truncate()

    def ingest_documents(
        self,
        documents: list[dict],
        enable_deduplication: bool = True,
    ) -> dict:
        t0 = time.perf_counter()

        chunks_processed = 0
        cache_hits = 0
        raw_entities_extracted = 0
        canonical_entities_created = 0
        entities_merged = 0
        edges_created = 0
        merge_log: list[dict] = []

        for doc in documents:
            text = doc["text"]
            doc_source = doc["doc_source"]
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

            # 1. SHA-256 Incremental Deduplication Check
            cursor = self.db.aql.execute(
                "FOR c IN chunks FILTER c.content_hash == @h LIMIT 1 RETURN c",
                bind_vars={"h": content_hash},
            )
            existing_chunks = list(cursor)
            if existing_chunks:
                cache_hits += 1
                merge_log.append({
                    "raw_name": f"Document [{doc_source}]",
                    "canonical_name": f"Cached Chunk [{existing_chunks[0]['_key']}]",
                    "entity_type": "DocumentChunk",
                    "score": 1.0,
                    "signal": f"SHA-256 Cache Hit (hash={content_hash[:8]}...)",
                    "canonical_key": existing_chunks[0]["_key"],
                    "merged": True,
                })
                continue

            # 2. Embed and Insert Chunk
            chunk_embedding = embed_text(text)
            # Use deterministic chunk key if provided or derived from hash
            chunk_key = doc.get("_key") or f"chunk_{content_hash[:10]}"
            chunk_doc = {
                "_key": chunk_key,
                "doc_source": doc_source,
                "text": text,
                "content_hash": content_hash,
                "embedding": chunk_embedding,
            }
            if self.chunks_col.has(chunk_key):
                self.chunks_col.replace(chunk_doc)
            else:
                self.chunks_col.insert(chunk_doc)
            chunks_processed += 1

            # 3. Structured Graph Extraction
            extraction = self.extractor.extract(doc_source, text)
            raw_entities_extracted += len(extraction.entities)

            # 4. Resolve Entities & Create Mentions Edges
            entity_key_map: dict[str, str] = {}
            for ent in extraction.entities:
                ent_embedding = embed_text(f"{ent.name} - {ent.description}")
                resolved_key, log_entry = self.resolver.resolve_entity(
                    entity=ent,
                    embedding=ent_embedding,
                    enable_deduplication=enable_deduplication,
                )
                entity_key_map[ent.name] = resolved_key
                merge_log.append(log_entry)

                if log_entry["merged"]:
                    entities_merged += 1
                else:
                    canonical_entities_created += 1

                # Link Chunk -> Entity (Mentions Edge)
                mention_edge = {
                    "_from": f"chunks/{chunk_key}",
                    "_to": f"entities/{resolved_key}",
                }
                # Avoid duplicate mention edge
                check_mention = self.db.aql.execute(
                    "FOR m IN mentions FILTER m._from == @f AND m._to == @t LIMIT 1 RETURN m",
                    bind_vars={"f": mention_edge["_from"], "t": mention_edge["_to"]},
                )
                if not list(check_mention):
                    self.mentions_col.insert(mention_edge)
                    edges_created += 1

            # 5. Insert Relations Edges
            for rel in extraction.relations:
                src_key = entity_key_map.get(rel.source_entity)
                tgt_key = entity_key_map.get(rel.target_entity)

                # Fallback if names varied slightly
                if not src_key:
                    for k in entity_key_map:
                        if k.lower() in rel.source_entity.lower() or rel.source_entity.lower() in k.lower():
                            src_key = entity_key_map[k]
                            break
                if not tgt_key:
                    for k in entity_key_map:
                        if k.lower() in rel.target_entity.lower() or rel.target_entity.lower() in k.lower():
                            tgt_key = entity_key_map[k]
                            break

                if src_key and tgt_key:
                    rel_edge = {
                        "_from": f"entities/{src_key}",
                        "_to": f"entities/{tgt_key}",
                        "relation_type": rel.relation_type,
                        "confidence": rel.confidence,
                        "evidence": rel.evidence,
                        "source_chunk": chunk_key,
                    }
                    # Avoid duplicate relation edge
                    check_rel = self.db.aql.execute(
                        """
                        FOR r IN relations 
                        FILTER r._from == @f AND r._to == @t AND r.relation_type == @rtype 
                        LIMIT 1 RETURN r
                        """,
                        bind_vars={
                            "f": rel_edge["_from"],
                            "t": rel_edge["_to"],
                            "rtype": rel_edge["relation_type"],
                        },
                    )
                    if not list(check_rel):
                        self.relations_col.insert(rel_edge)
                        edges_created += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "elapsed_ms": round(elapsed_ms, 2),
            "chunks_processed": chunks_processed,
            "cache_hits": cache_hits,
            "raw_entities_extracted": raw_entities_extracted,
            "canonical_entities_created": canonical_entities_created,
            "entities_merged": entities_merged,
            "merge_log": merge_log,
            "edges_created": edges_created,
        }

