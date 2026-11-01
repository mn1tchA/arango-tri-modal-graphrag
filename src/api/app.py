import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from src.db.schema import init_database
from src.retrieval.retriever import TriModalRetriever
from src.agent.workflow import build_graphrag_agent
from src.agent.state import GraphRAGAgentState
from src.ingestion.pipeline import GraphIngestionPipeline
from src.ingestion.seed_data import embed_text, MULTI_HOP_CORPUS_6_DOCS
from benchmarks.evaluate_suite import run_benchmark_suite

# Initialize singletons
_db = None
_retriever = None
_agent = None
_pipeline = None


def get_db():
    global _db
    if _db is None:
        _db = init_database()
    # Guarantee auto-seeding if database has 0 chunks (e.g. fresh container in CI)
    try:
        chunks_col = _db.collection("chunks")
        if chunks_col.count() == 0:
            pipeline = GraphIngestionPipeline(_db)
            pipeline.ingest_documents(MULTI_HOP_CORPUS_6_DOCS, enable_deduplication=True)
    except Exception:
        pass
    return _db


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = TriModalRetriever(get_db())
    return _retriever


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_graphrag_agent(get_retriever())
    return _agent


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = GraphIngestionPipeline(get_db())
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Warm up database schema and seed if empty
    db = get_db()

    # 2. Warm up sentence-transformer embeddings in memory
    _ = embed_text("warmup query embedding initialization")

    # 3. Pre-initialize retriever and agent singletons
    _ = get_retriever()
    _ = get_agent()

    yield


app = FastAPI(
    title="ArangoDB Tri-Modal GraphRAG API",
    description="Explainable Subgraph Studio and Multi-Model Reasoning Engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class IngestRequest(BaseModel):
    doc_source: str
    text: str
    enable_deduplication: bool = True


@app.get("/api/health")
def health_check():
    db = get_db()
    try:
        ver = db.version()
        chunks_count = db.collection("chunks").count()
        entities_count = db.collection("entities").count()
        mentions_count = db.collection("mentions").count()
        relations_count = db.collection("relations").count()

        return {
            "status": "healthy",
            "database": "tri_modal_graphrag",
            "version": ver,
            "counts": {
                "chunks": chunks_count,
                "entities": entities_count,
                "mentions": mentions_count,
                "relations": relations_count,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph")
def get_graph():
    db = get_db()
    try:
        cursor_nodes = db.aql.execute("""
            FOR e IN entities
                RETURN {
                    id: e._id,
                    key: e._key,
                    label: e.canonical_name,
                    canonical_name: e.canonical_name,
                    entity_type: e.entity_type,
                    aliases: e.aliases,
                    mention_count: e.mention_count,
                    description: e.description
                }
        """)
        cursor_edges = db.aql.execute("""
            FOR r IN relations
                LET src = DOCUMENT(r._from)
                LET tgt = DOCUMENT(r._to)
                RETURN {
                    id: r._id,
                    source: r._from,
                    target: r._to,
                    source_label: src ? src.canonical_name : r._from,
                    target_label: tgt ? tgt.canonical_name : r._to,
                    label: r.relation_type,
                    relation_type: r.relation_type,
                    confidence: r.confidence,
                    evidence: r.evidence,
                    source_chunk: r.source_chunk
                }
        """)

        nodes = [{"data": n} for n in cursor_nodes]
        edges = [{"data": e} for e in cursor_edges]

        return {
            "nodes": nodes,
            "edges": edges,
            "elements": {
                "nodes": nodes,
                "edges": edges,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
def execute_query(req: QueryRequest):
    agent = get_agent()
    retriever = get_retriever()
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    t_start = time.perf_counter()

    # 1. Run Adaptive LangGraph Agent
    initial_state: GraphRAGAgentState = {
        "query": query,
        "agent_trace": [],
        "retrieved_chunks": [],
        "graph_paths": [],
        "discovered_entity_types": [],
        "correction_attempts": 0,
    }
    agent_state = agent.invoke(initial_state)
    agent_lat = (time.perf_counter() - t_start) * 1000
    agent_state["total_latency_ms"] = round(agent_lat, 2)

    # 2. Run Baselines for Side-by-Side Comparison
    vec_res = retriever.retrieve_vector_only(query, top_k=2)
    hyb_res = retriever.retrieve_hybrid_rrf(query, top_k=2)

    tri_chunk_ids = [c["chunk_id"] for c in agent_state.get("retrieved_chunks", [])]
    vec_chunk_ids = [c["chunk_id"] for c in vec_res.get("chunks", [])]
    hyb_chunk_ids = [c["chunk_id"] for c in hyb_res.get("chunks", [])]

    # Extract highlighted entity names and IDs along the active reasoning path
    highlighted = set()
    for p in agent_state.get("graph_paths", []):
        if p.get("from_name"):
            highlighted.add(p["from_name"])
        if p.get("to_name"):
            highlighted.add(p["to_name"])
    for ent_name in agent_state.get("discovered_entities", []):
        highlighted.add(ent_name)

    comparison = {
        "pure_vector": {
            "chunk_ids": vec_chunk_ids,
            "latency_ms": vec_res["latency_ms"],
            "missed_chunks": [cid for cid in tri_chunk_ids if cid not in vec_chunk_ids],
        },
        "hybrid_rrf": {
            "chunk_ids": hyb_chunk_ids,
            "latency_ms": hyb_res["latency_ms"],
            "missed_chunks": [cid for cid in tri_chunk_ids if cid not in hyb_chunk_ids],
        },
        "tri_modal_graphrag": {
            "chunk_ids": tri_chunk_ids,
            "latency_ms": agent_state["total_latency_ms"],
            "bridged_chunks_count": len(
                [c for c in agent_state.get("retrieved_chunks", []) if "Graph-Bridged" in str(c.get("score"))]
            ),
        },
    }

    return {
        "agent_result": agent_state,
        "highlighted_entities": list(highlighted),
        "comparison": comparison,
    }


@app.post("/api/ingest")
def ingest_document(req: IngestRequest):
    pipeline = get_pipeline()
    try:
        telemetry = pipeline.ingest_documents(
            documents=[{"doc_source": req.doc_source, "text": req.text}],
            enable_deduplication=req.enable_deduplication,
        )
        return telemetry
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/benchmarks")
def get_benchmarks():
    base_dir = Path(__file__).resolve().parent.parent.parent
    benchmarks_json = base_dir / "benchmarks" / "benchmark_results.json"

    if benchmarks_json.exists():
        try:
            with open(benchmarks_json, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Generate if not cached
    db = get_db()
    payload = run_benchmark_suite(db)
    return payload


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    base_dir = Path(__file__).resolve().parent.parent.parent
    ui_path = base_dir / "src" / "ui" / "index.html"
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="UI index.html not found.")
    with open(ui_path, "r", encoding="utf-8") as f:
        return f.read()

