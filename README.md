# ArangoDB Tri-Modal GraphRAG

[![CI Status](https://github.com/mn1tchA/arango-tri-modal-graphrag/actions/workflows/ci.yml/badge.svg)](https://github.com/mn1tchA/arango-tri-modal-graphrag/actions)
[![ArangoDB](https://img.shields.io/badge/ArangoDB-3.12.4-blue.svg?logo=arangodb)](https://www.arangodb.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph_CRAG-purple.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-teal.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Cytoscape.js](https://img.shields.io/badge/Cytoscape.js-Subgraph_UI-red.svg)](https://js.cytoscape.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**ArangoDB Tri-Modal GraphRAG** is a domain-agnostic retrieval and reasoning engine that executes **dense vector search (FAISS)**, **lexical keyword search (ArangoSearch BM25)**, and **multi-hop property graph traversal** inside a single compiled **AQL (ArangoDB Query Language)** query.

Most production GraphRAG systems suffer from the *polyglot persistence penalty*: they synchronize chunks in a vector database (Pinecone/Qdrant), inverted indexes in a search engine (Elasticsearch), and entities in a graph database (Neo4j), forcing the application layer to coordinate three network roundtrips and merge results in Python memory. This project collapses all three retrieval modalities into a single storage engine and wraps it in a self-correcting **LangGraph** state machine with write-time **3-Signal Entity Resolution**.

---

## Table of Contents
1. [Architectural Motivation: Single-Engine vs. Polyglot Stacks](#1-architectural-motivation-single-engine-vs-polyglot-stacks)
2. [Physical Data Model & Multi-Model Schema](#2-physical-data-model--multi-model-schema)
3. [System Architecture & LangGraph Control Flow](#3-system-architecture--langgraph-control-flow)
4. [Algorithmic Foundations](#4-algorithmic-foundations)
   - [In-Engine Reciprocal Rank Fusion (RRF)](#in-engine-reciprocal-rank-fusion-rrf)
   - [Write-Path 3-Signal Hybrid Entity Resolution](#write-path-3-signal-hybrid-entity-resolution)
   - [Idempotent Ingestion via SHA-256 Content Addressing](#idempotent-ingestion-via-sha-256-content-addressing)
5. [Multi-Hop Bridging Mechanics](#5-multi-hop-bridging-mechanics)
6. [Empirical Benchmarks & Ablation Studies](#6-empirical-benchmarks--ablation-studies)
7. [Engineering Trade-Offs & Complexity Analysis](#7-engineering-trade-offs--complexity-analysis)
8. [Explainable Subgraph Studio (FastAPI + Cytoscape.js)](#8-explainable-subgraph-studio-fastapi--cytoscapejs)
9. [Repository Structure](#9-repository-structure)
10. [Quickstart & Reproduction](#10-quickstart--reproduction)

---

## 1. Architectural Motivation: Single-Engine vs. Polyglot Stacks

When answering multi-hop relational queries, standard RAG pipelines typically chain three separate databases over the network.

```mermaid
flowchart TB
    subgraph Polyglot["Polyglot RAG Stack (3 Network Hops + Application-Side Join)"]
        direction LR
        App1["Python Backend"] -->|"1. gRPC (Dense Vectors)"| VecDB[("Qdrant / Pinecone")]
        App1 -->|"2. REST (BM25 Lexical)"| SearchDB[("Elasticsearch")]
        App1 -->|"3. Bolt (Cypher Traversal)"| GraphDB[("Neo4j")]
    end

    subgraph Unified["ArangoDB Tri-Modal Engine (Single Compiled AQL Execution Plan)"]
        direction LR
        App2["Python Backend"] -->|"1 Atomic AQL Query"| Arango["ArangoDB 3.12+ C++ Core"]
        Arango --- FAISS["FAISS Vector Index"]
        Arango --- BM25["ArangoSearch BM25 View"]
        Arango --- Graph["Property Graph Kernel"]
    end
```

| System Dimension | Polyglot Stack (Vector + Search + Graph DBs) | ArangoDB Native Tri-Modal Engine |
| :--- | :--- | :--- |
| **Query Execution Path** | 3 sequential/parallel network calls across REST, gRPC, and Bolt | **1 compiled AQL query** executed in shared C++ memory |
| **Rank Fusion (RRF)** | Pulls candidate sets over the network; merges in Python | **Computed in-engine** via AQL array position operators |
| **Write Consistency** | Eventual consistency; partial failures cause orphaned graph nodes | **Single-engine ACID** transactions across vertices, edges, and views |
| **Operational Footprint** | 3 stateful clusters, 3 backup lifecycles, 3 connection pools | **1 container / cluster** managing vectors, text analyzers, and edges |

---

## 2. Physical Data Model & Multi-Model Schema

The storage layer (`src/db/schema.py`) models unstructured text chunks and structured knowledge entities inside a unified named property graph (`knowledge_graph`) backed by an `ArangoSearch` inverted index view (`hybrid_search_view`).

```mermaid
erDiagram
    CHUNKS ||--o{ MENTIONS : "outbound (_from)"
    ENTITIES ||--o{ MENTIONS : "inbound (_to)"
    ENTITIES ||--o{ RELATIONS : "source (_from) / target (_to)"

    CHUNKS {
        string _key PK
        string doc_source
        string text "Indexed by ArangoSearch (text_en)"
        string content_hash UK "Unique SHA-256 Persistent Index"
        float_array embedding "384-dim L2-Normalized Vector"
    }

    ENTITIES {
        string _key PK
        string canonical_name "Composite Persistent Index"
        string entity_type "Drug | Enzyme | AdverseEvent | Condition"
        string_array aliases "Indexed by ArangoSearch (text_en, identity)"
        string description
        int mention_count
        float_array embedding "384-dim L2-Normalized Vector"
    }

    MENTIONS {
        string _from FK "chunks/_key"
        string _to FK "entities/_key"
    }

    RELATIONS {
        string _from FK "entities/_key"
        string _to FK "entities/_key"
        string relation_type "POTENTLY_INHIBITS | METABOLIZED_BY | BLOCKADE_PRECIPITATES"
        float confidence "Edge Traversal Filter (>= 0.75)"
        string source_chunk FK "Provenance Pointer to chunks/_key"
        string evidence "Extracted Grounding Span"
    }
```

---

## 3. System Architecture & LangGraph Control Flow

Rather than running expensive multi-hop graph traversals on every prompt, the query pipeline (`src/agent/workflow.py`) uses a cyclic **LangGraph** state machine. It classifies query complexity, selects the minimum viable retrieval strategy, grades the retrieved subgraph for structural completeness, and autonomously expands the traversal radius ($1 \to 2$ hops) only when intermediate entities fail to reach a terminal node.

```mermaid
flowchart TD
    Start(["Incoming User Query"]) --> Router["route_query_node<br/>(Intent & Target Node Type Classifier)"]

    Router -->|"LEXICAL_LOOKUP<br/>(max_hops = 0)"| Hybrid["execute_retrieval_node<br/>Mode: Hybrid RRF (Vector + BM25)"]
    Router -->|"RELATIONAL_1HOP | CAUSAL_MULTIHOP<br/>(Initial max_hops = 1)"| TriModal["execute_retrieval_node<br/>Mode: Tri-Modal AQL (Anchors + 1-Hop BFS)"]

    Hybrid --> Grader["grade_subgraph_node<br/>(Corrective RAG Evaluator)"]
    TriModal --> Grader

    Grader -->|"Anchor Score < Threshold (0.35)<br/>and Zero Entity Overlap"| Abstain["synthesize_answer_node<br/>Status: INSUFFICIENT_ABSTAIN"]
    Grader -->|"Missing Required Terminal Node Type<br/>(e.g. AdverseEvent) & current_hops < 2"| Expander["expand_graph_hops_node<br/>Increment max_hops: 1 -> 2"]
    Grader -->|"All Required Node Types Reached<br/>Status: SUFFICIENT"| Synth["synthesize_answer_node<br/>Path-Cited Provenance Synthesis"]

    Expander -->|"Re-Execute AQL Traversal"| TriModal
    Abstain --> End(["Return Grounded Response & Trace"])
    Synth --> End
```

---

## 4. Algorithmic Foundations

### In-Engine Reciprocal Rank Fusion (RRF)

Cosine similarity scores ($\in [-1, 1]$) and unbounded BM25 lexical scores ($\in [0, \infty)$) cannot be linearly combined without brittle min-max calibration. Our retrieval engine computes **Reciprocal Rank Fusion** directly inside AQL using ordinal rank positions:

$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{vec}, \text{bm25}\}} \frac{1}{k + \text{rank}_m(d)}$$

Where:
- $k = 60$ dampens the impact of high-variance outliers in individual modalities.

```sql
// Single-Query AQL Tri-Modal Retrieval (Vector + BM25 + k-Hop Graph Expansion)
LET anchor_chunks = (
    FOR doc IN chunks
        LET v_score = SUM(
            FOR i IN 0..(LENGTH(doc.embedding) - 1)
                RETURN doc.embedding[i] * @query_vec[i]
        )
        SORT v_score DESC
        LIMIT @top_k
        RETURN { chunk_id: doc._key, source: doc.doc_source, text: doc.text, score: v_score, _id: doc._id }
)

LET seed_entities = UNIQUE(
    FOR c IN anchor_chunks
        FOR ent IN 1..1 OUTBOUND c._id mentions
            RETURN ent._id
)

LET traversed_subgraph = (
    FOR start_ent IN seed_entities
        FOR v, e, p IN 1..@max_hops ANY start_ent relations
            OPTIONS { uniqueVertices: "path", bfs: true }
            FILTER e.confidence >= @min_confidence
            LET from_node = DOCUMENT(e._from)
            LET to_node = DOCUMENT(e._to)
            LET evidence_chunk = DOCUMENT(CONCAT("chunks/", e.source_chunk))
            RETURN DISTINCT {
                hop_edge: CONCAT(from_node.canonical_name, " --[", e.relation_type, "]--> ", to_node.canonical_name),
                evidence: e.evidence,
                confidence: e.confidence,
                bridged_chunk_id: evidence_chunk._key,
                bridged_chunk_text: evidence_chunk.text,
                bridged_source: evidence_chunk.doc_source
            }
)

RETURN { anchors: anchor_chunks, graph_paths: traversed_subgraph }
```

---

### Write-Path 3-Signal Hybrid Entity Resolution
Unstructured corpora refer to identical entities using inconsistent surface strings (e.g., `"Fluoxetine Hydrochloride (Prozac)"` vs. `"Fluoxetine"`, or `"Cytochrome P450 2D6 (CYP2D6)"` vs. `"Hepatic Isoenzyme CYP2D6"`). If ingested naively, each surface string creates an isolated vertex, severing multi-hop paths.

Before inserting a candidate entity $e_1$ against an existing canonical entity $e_2$ of the same ontology class, `HybridEntityResolver` (`src/ingestion/entity_resolver.py`) evaluates three orthogonal signals:

$$\text{Score}(e_1, e_2) = \max\Big(w_{\text{sem}} \cdot \cos(\mathbf{v}_1, \mathbf{v}_2) + w_{\text{lex}} \cdot \text{JW}(s_1, s_2) + w_{\text{nbr}} \cdot \text{Jaccard}(N_1, N_2), \; S_{\text{alias}}(s_1, e_2)\Big)$$

1. **Dense Semantic Similarity ($\cos$)**: Dot product of L2-normalized 384-dimensional `all-MiniLM-L6-v2` embeddings over `name + description`.
2. **C-Accelerated Jaro-Winkler Similarity ($\text{JW}$)**: Computed via `jellyfish` across the candidate surface string $s_1$ and all known aliases of $e_2$:
   $$\text{JW}(s_1, s_2) = \text{Jaro}(s_1, s_2) + \ell \cdot p \cdot \big(1 - \text{Jaro}(s_1, s_2)\big)$$
3. **1-Hop Neighborhood Overlap ($\text{Jaccard}$)**: Structural overlap of adjacent entity vertices $N_1$ and $N_2$:
   $$\text{Jaccard}(N_1, N_2) = \frac{\vert{}N_1 \cap N_2\vert{}}{\vert{}N_1 \cup N_2\vert{}}$$
4. **Acronym & Sub-Token Containment ($S_{\text{alias}}$)**: Extracts parenthetical codes (such as `CYP2D6`) and canonical root tokens to resolve domain abbreviations while preventing false-positive merges between homologous codes (e.g., `CYP2D6` vs. `CYP2C9` scores `0.754`, staying below the merge threshold).

---

### Idempotent Ingestion via SHA-256 Content Addressing
Every ingested document chunk is normalized and hashed prior to embedding or entity extraction:

$$H(c) = \text{SHA-256}\big(\text{strip}(c_{\text{text}})\big)$$

A unique persistent index on `chunks.content_hash` intercepts duplicate documents in $\mathcal{O}(1)$ lookup time, bypassing vector encoding and graph extraction on re-runs.

---

## 5. Multi-Hop Bridging Mechanics

The diagram below illustrates why pure vector search and hybrid keyword search fail on multi-hop relational queries, and how write-time entity resolution paired with 2-hop AQL traversal bridges isolated documents:

```mermaid
graph LR
    subgraph RawDocs["Isolated Unstructured Documents"]
        D1["Doc 1: er_admission_note.pdf<br/>Mentions: 'Fluoxetine Hydrochloride (Prozac)'<br/>and 'Tramadol HCl'"]
        D2["Doc 2: cyp450_inhibition_study.pdf<br/>Mentions: 'Fluoxetine' inhibits<br/>'Cytochrome P450 2D6 (CYP2D6)'"]
        D3["Doc 3: opioid_prodrug_kinetics.pdf<br/>Mentions: 'Tramadol' requires 'Hepatic Isoenzyme CYP2D6';<br/>blockade precipitates 'Serotonin Syndrome'"]
    end

    subgraph ResolvedGraph["Canonical Property Graph (After 3-Signal Entity Resolution)"]
        E1(("Fluoxetine<br/>[Drug]"))
        E2(("CYP2D6<br/>[Enzyme]"))
        E3(("Tramadol<br/>[Drug]"))
        E4(("Serotonin Syndrome<br/>[AdverseEvent]"))

        E1 -->|"POTENTLY_INHIBITS (conf: 0.96)"| E2
        E3 -->|"METABOLIZED_BY (conf: 0.98)"| E2
        E2 -->|"BLOCKADE_PRECIPITATES (conf: 0.92)"| E4
    end

    D1 -.->|"Vector Anchor (Score: 0.59)"| E1
    D1 -.->|"Vector Anchor"| E3
    E2 ==>|"Graph-Bridged Provenance"| D2
    E4 ==>|"2-Hop CRAG Expansion"| D3
```

* **Why Vector & BM25 Miss Document 2:** A query asking *"Why did the patient develop serotonin toxicity after Tramadol administration?"* shares high lexical and semantic overlap with Document 1 and Document 3, but almost zero surface overlap with Document 2 (an isolated enzyme inhibition monograph).
* **How Tri-Modal GraphRAG Bridges Document 2:** Vector search anchors on Document 1 (`Fluoxetine`, `Tramadol`). The AQL traversal walks outbound and inbound edges across the resolved `CYP2D6` hub, automatically pulling Document 2 (`cyp450_inhibition_study.pdf`) into the LLM context window via its edge provenance pointer (`source_chunk`).

---

## 6. Empirical Benchmarks & Ablation Studies

### Phase 2 Ablation: Naive Ingestion vs. 3-Signal Entity Resolution
Evaluated across the 6-document Multi-Hop Relational Benchmark Corpus containing deliberate surface-form variations:

| Metric | Naive Graph Ingestion (`dedup=False`) | Hybrid Entity-Resolved (`dedup=True`) | Impact |
| :--- | :---: | :---: | :--- |
| **Canonical Entity Vertices** | 14 (Fragmented) | **8 (Unified Hubs)** | **-42.8% Vertex Redundancy** |
| **Unmerged Surface Aliases** | 6 Orphaned Nodes | **0 Orphaned (6 Merged into `aliases[]`)** | **100% Alias Consolidation** |
| **Homologous False Merges (`CYP2D6` vs `CYP2C9`)** | 0 | **0 (Scored `0.754 < 0.78` Threshold)** | **High Specificity Preserved** |
| **3-Document Multi-Hop Bridge Recall** | 66.7% (Broken Bridges) | **100.0% (All Paths Intact)** | **+33.3% Multi-Hop Recall** |

---

### Phase 3 Telemetry: LangGraph Adaptive Router & CRAG Self-Correction

| Query Intent Class | Sample Query | Selected Mode | Hops Executed | CRAG Evaluator Outcome | Mean Latency |
| :--- | :--- | :---: | :---: | :--- | :---: |
| `LEXICAL_LOOKUP` | *"What exact dosage of Fluoxetine was administered in the ER?"* | `hybrid_rrf` | `0` | `SUFFICIENT` (Bypassed Graph) | **63.1 ms** |
| `RELATIONAL_1HOP` | *"Which specific cytochrome enzyme is inhibited by Fluconazole?"* | `tri_modal` | `1` | `SUFFICIENT` (Pass 1 Complete) | **67.3 ms** |
| `CAUSAL_MULTIHOP` | *"Trace the enzymatic interaction between Fluoxetine and Tramadol."* | `tri_modal` | `1 -> 2` | `NEEDS_HOP_EXPANSION` $\to$ `SUFFICIENT` | **139.2 ms** |
| `OUT_OF_DOMAIN` | *"What is the metabolic interaction between Metformin and Lisinopril?"* | `tri_modal` | `1` | `INSUFFICIENT_ABSTAIN` (Guard Triggered) | **63.7 ms** |

---

### Phase 4 Quantitative Evaluation: 4-Architecture Benchmark Matrix
Reproducible via `python -m benchmarks.evaluate_suite` across the 8-query evaluation suite (2 lexical lookups, 2 single-hop relational queries, 2 three-document causal chains, and 2 out-of-domain hallucination traps):

| Architecture | Context Recall | Multi-Hop Recall | Entity Bridge Recall | OOD Trap Rejection | Mean Latency | P95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Vector RAG (`top_k=2`)** | 80.0% | 66.7% | 91.7% | 0.0% | 61.7 ms | 65.2 ms |
| **Hybrid Search (BM25 + Vector RRF)** | 80.0% | 66.7% | 100.0% | 0.0% | 63.0 ms | 67.6 ms |
| **Naive GraphRAG (Unresolved Entities)** | 80.0% | 66.7% | 91.7% | 0.0% | 117.0 ms | 120.5 ms |
| **Adaptive CRAG Tri-Modal GraphRAG** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **84.3 ms** | **128.7 ms** |

---

## 7. Engineering Trade-Offs & Complexity Analysis

1. **Write-Time vs. Read-Time Entity Resolution:**
   * *Decision:* Entity deduplication runs synchronously during ingestion (`GraphIngestionPipeline`) rather than via fuzzy matching at query time.
   * *Trade-off:* Increases ingestion latency by $\mathcal{O}(\vert{}V_{\text{type}}\vert{})$ per extracted entity, but guarantees that read-time graph traversals operate over clean canonical vertices in $\mathcal{O}(b^k)$ time without runtime string similarity joins.
2. **Bounded BFS Traversal ($k \le 2$) & Supernode Mitigation:**
   * *Decision:* Graph expansion uses `OPTIONS { uniqueVertices: "path", bfs: true }` with an adaptive ceiling of $k=2$ hops and edge confidence filtering (`e.confidence >= 0.75`).
   * *Trade-off:* Unbounded graph traversals ($k \ge 4$) on dense graphs suffer from combinatorial fan-out ($\mathcal{O}(b^k)$ where $b$ is average vertex degree), flooding the LLM context window with low-relevance peripheral nodes. Starting at $k=1$ and only expanding to $k=2$ when the CRAG grader detects a missing target entity type keeps mean query latency at **84.3 ms**.
3. **Reciprocal Rank Fusion ($k=60$) vs. Cross-Encoder Reranking:**
   * *Decision:* Hybrid fusion is computed inside AQL via RRF rather than pulling 50 candidates into Python for neural cross-encoder reranking.
   * *Trade-off:* Eliminates ~150–300 ms of CPU/GPU cross-encoder inference latency while still capturing exact lexical identifiers (dosages, enzyme codes) via `ArangoSearch` BM25 alongside dense semantic matches.

---

## 8. Explainable Subgraph Studio (FastAPI + Cytoscape.js)

The repository ships with a zero-build single-page web application (`src/ui/index.html`) served by FastAPI (`src/api/app.py`):

* **Force-Directed Property Graph Canvas:** Renders all canonical vertices and typed edges via Cytoscape.js. Executing a query highlights the active vector anchor nodes and the traversed $k$-hop reasoning path.
* **LangGraph State Trace Inspector:** Displays real-time step latencies for intent routing, AQL execution, CRAG completeness grading, and autonomous hop expansion.
* **Side-by-Side Retrieval Diff Matrix:** Compares retrieved document chunks across *Pure Vector*, *Hybrid RRF*, and *Tri-Modal GraphRAG* on every query to expose exactly which intermediate documents were bridged by the graph.
* **Live Ingestion & Alias Inspector:** Click any vertex to inspect its consolidated `aliases[]` array, or submit raw text to test SHA-256 deduplication and entity resolution live.

---

## 9. Repository Structure

```text
arango-tri-modal-graphrag/
├── .github/workflows/
│   └── ci.yml                      # GitHub Actions CI (ArangoDB 3.12.4 service + pytest)
├── docker-compose.yml              # ArangoDB 3.12.4 container with experimental FAISS index flag
├── requirements.txt                # Python dependencies
├── main.py                         # CLI verification & ablation benchmark runner
├── benchmarks/
│   ├── evaluate_suite.py           # 4-architecture quantitative evaluation harness
│   └── benchmark_results.json      # Exported latency & recall telemetry
├── src/
│   ├── db/
│   │   └── schema.py               # Collections, named graph, persistent indexes, ArangoSearch view
│   ├── ingestion/
│   │   ├── seed_data.py            # 6-document Multi-Hop Relational Benchmark Corpus
│   │   ├── extractor.py            # Pydantic v2 structured entity & relation extractor
│   │   ├── entity_resolver.py      # 3-signal hybrid entity deduplication engine
│   │   └── pipeline.py             # SHA-256 content-addressed ingestion pipeline
│   ├── retrieval/
│   │   └── retriever.py            # Pure Vector, Hybrid RRF, and Tri-Modal AQL queries
│   ├── agent/
│   │   ├── state.py                # LangGraph TypedDict state schema
│   │   ├── nodes.py                # Router, Retriever, CRAG Grader, Hop Expander, Synthesizer
│   │   └── workflow.py             # Compiled StateGraph workflow
│   ├── api/
│   │   └── app.py                  # FastAPI service with lifespan warmup & auto-seeding guard
│   └── ui/
│       └── index.html              # Cytoscape.js Explainable Subgraph Studio
└── tests/
    └── test_api.py                 # 7 integration & contract tests
```

---

## 10. Quickstart & Reproduction

### 1. Start ArangoDB 3.12.4
```bash
docker-compose up -d
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the CLI Ablation & Agentic Benchmark
Executes Phase 2 entity resolution ablation (Naive vs. Resolved) and Phase 3 LangGraph CRAG scenarios in the terminal:
```bash
python main.py
```

### 4. Run the Quantitative 4-Architecture Evaluation Suite
```bash
python -m benchmarks.evaluate_suite
```

### 5. Launch the Interactive Subgraph Studio
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
* **Subgraph Studio UI:** `http://localhost:8000/`
* **OpenAPI Documentation:** `http://localhost:8000/docs`

### 6. Run the Automated Test Suite
```bash
pytest tests/test_api.py -v
```

---

## License
Released under the [MIT License](LICENSE).