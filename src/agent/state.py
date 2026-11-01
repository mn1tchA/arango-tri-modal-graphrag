from typing import Literal, TypedDict


class GraphRAGAgentState(TypedDict):
    query: str
    query_intent: Literal["LEXICAL_LOOKUP", "RELATIONAL_1HOP", "CAUSAL_MULTIHOP"]
    required_node_types: list[str]
    retrieval_strategy: str  # "hybrid_rrf" or "tri_modal_graphrag"
    current_max_hops: int
    correction_attempts: int
    retrieved_chunks: list[dict]
    graph_paths: list[dict]
    discovered_entity_types: list[str]
    grader_status: Literal["SUFFICIENT", "NEEDS_HOP_EXPANSION", "INSUFFICIENT_ABSTAIN"]
    grader_reasoning: str
    final_answer: str
    citations: list[str]
    agent_trace: list[str]
    total_latency_ms: float

