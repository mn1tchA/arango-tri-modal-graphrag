"""
Automated FastAPI Integration & Contract Tests for ArangoDB Tri-Modal GraphRAG Studio.
Verifies all API endpoints return HTTP 200 OK with expected JSON/HTML schemas.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Ensure sys.stdout is utf-8 friendly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_endpoint():
    """Verify health check returns ArangoDB version and collection counts."""
    response = client.get("/api/health")
    assert response.status_code == 200, f"Health check failed: {response.text}"
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "tri_modal_graphrag"
    assert "version" in data
    counts = data["counts"]
    assert counts["chunks"] > 0
    assert counts["entities"] > 0
    assert counts["relations"] > 0
    print(f"\n[OK] Health check passed: {counts['entities']} entities, {counts['relations']} relations")


def test_graph_endpoint():
    """Verify graph endpoint delivers Cytoscape.js compatible nodes and edges."""
    response = client.get("/api/graph")
    assert response.status_code == 200, f"Graph endpoint failed: {response.text}"
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) > 0
    assert len(data["edges"]) > 0

    # Verify Cytoscape element structure
    first_node = data["nodes"][0]
    assert "data" in first_node
    assert "id" in first_node["data"]
    assert "canonical_name" in first_node["data"]

    first_edge = data["edges"][0]
    assert "data" in first_edge
    assert "source" in first_edge["data"]
    assert "target" in first_edge["data"]
    assert "relation_type" in first_edge["data"]

    print(f"\n[OK] Graph endpoint verified: {len(data['nodes'])} Cytoscape nodes, {len(data['edges'])} edges")


def test_query_causal_multihop():
    """Verify multi-hop causal query triggers CRAG hop expansion and path-cited answer."""
    query = "Why did the patient develop severe serotonin toxicity after Tramadol administration?"
    response = client.post("/api/query", json={"query": query})
    assert response.status_code == 200, f"Query endpoint failed: {response.text}"
    data = response.json()

    assert "agent_result" in data
    assert "highlighted_entities" in data
    assert "comparison" in data

    agent_result = data["agent_result"]
    assert agent_result["query_intent"] == "CAUSAL_MULTIHOP"
    assert agent_result["grader_status"] == "SUFFICIENT"
    assert len(agent_result["retrieved_chunks"]) >= 3
    assert len(agent_result["citations"]) > 0
    assert "Serotonin Syndrome" in str(agent_result["final_answer"]) or "serotonin" in str(agent_result["final_answer"]).lower()

    # Verify comparison object
    comp = data["comparison"]
    assert "pure_vector" in comp
    assert "hybrid_rrf" in comp
    assert "tri_modal_graphrag" in comp
    assert comp["tri_modal_graphrag"]["bridged_chunks_count"] >= 1

    print(f"\n[OK] Causal Multi-Hop query passed in {agent_result.get('total_latency_ms', 0):.2f} ms")


def test_query_lexical_lookup():
    """Verify lexical query is routed to 0-hop hybrid lookup."""
    query = "What dosage of Fluoxetine was administered?"
    response = client.post("/api/query", json={"query": query})
    assert response.status_code == 200, f"Query endpoint failed: {response.text}"
    data = response.json()

    agent_result = data["agent_result"]
    assert agent_result["query_intent"] == "LEXICAL_LOOKUP"
    assert agent_result["retrieval_strategy"] == "hybrid_rrf"
    assert agent_result["current_max_hops"] == 0
    assert len(agent_result["retrieved_chunks"]) > 0

    print(f"\n[OK] Lexical lookup passed with strategy: {agent_result['retrieval_strategy']}")


def test_query_out_of_domain_abstain():
    """Verify out-of-domain query is correctly flagged and abstained."""
    query = "What is the capital of France and its GDP growth?"
    response = client.post("/api/query", json={"query": query})
    assert response.status_code == 200, f"Query endpoint failed: {response.text}"
    data = response.json()

    agent_result = data["agent_result"]
    assert agent_result["grader_status"] == "INSUFFICIENT_ABSTAIN"
    assert "insufficient" in agent_result["final_answer"].lower() or "not contain" in agent_result["final_answer"].lower()

    print(f"\n[OK] Out-of-domain guardrail passed with status: {agent_result['grader_status']}")


def test_benchmarks_endpoint():
    """Verify benchmarks endpoint returns quantitative metrics across all 4 architectures."""
    response = client.get("/api/benchmarks")
    assert response.status_code == 200, f"Benchmarks endpoint failed: {response.text}"
    data = response.json()

    assert "results" in data
    assert len(data["results"]) == 4

    # Locate Adaptive CRAG Tri-Modal GraphRAG system metrics
    adaptive = next((s for s in data["results"] if "Adaptive CRAG" in s.get("system_name", "")), None)
    assert adaptive is not None, "Adaptive CRAG system metrics not found"
    assert adaptive["multihop_recall_pct"] == 100.0
    assert adaptive["ood_rejection_pct"] == 100.0
    assert adaptive["context_recall_pct"] == 100.0

    print(f"\n[OK] Benchmark metrics verified. Adaptive Tri-Modal Multi-hop recall: {adaptive['multihop_recall_pct']}%, OOD Rejection: {adaptive['ood_rejection_pct']}%")


def test_ui_html_endpoint():
    """Verify root / serves the Explainable Subgraph Studio UI with Cytoscape container."""
    response = client.get("/")
    assert response.status_code == 200, f"Root UI failed: {response.text}"
    assert "text/html" in response.headers["content-type"]
    html_content = response.text

    assert "cytoscape" in html_content.lower()
    assert 'id="cy"' in html_content
    assert "Tri-Modal GraphRAG" in html_content

    print(f"\n[OK] Studio HTML UI successfully served ({len(html_content)} bytes)")


if __name__ == "__main__":
    print("Running automated FastAPI tests directly...")
    test_health_endpoint()
    test_graph_endpoint()
    test_query_causal_multihop()
    test_query_lexical_lookup()
    test_query_out_of_domain_abstain()
    test_benchmarks_endpoint()
    test_ui_html_endpoint()
    print("\nALL 7 API INTEGRATION TESTS PASSED SUCCESSFULLY!")
