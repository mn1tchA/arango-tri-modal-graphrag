"""
LangGraph StateGraph Workflow Compiler for Adaptive Tri-Modal GraphRAG.
Orchestrates adaptive query routing, multi-modal retrieval, corrective subgraph grading (CRAG),
autonomous hop expansion, and path-cited answer synthesis.
"""

from langgraph.graph import StateGraph, START, END
from src.agent.state import GraphRAGAgentState
from src.agent.nodes import GraphRAGAgentNodes
from src.retrieval.retriever import TriModalRetriever


def build_graphrag_agent(retriever: TriModalRetriever):
    nodes = GraphRAGAgentNodes(retriever)
    workflow = StateGraph(GraphRAGAgentState)

    # 1. Add Nodes
    workflow.add_node("route_query", nodes.route_query_node)
    workflow.add_node("execute_retrieval", nodes.execute_retrieval_node)
    workflow.add_node("grade_subgraph", nodes.grade_subgraph_node)
    workflow.add_node("expand_graph_hops", nodes.expand_graph_hops_node)
    workflow.add_node("synthesize_answer", nodes.synthesize_answer_node)

    # 2. Add Edges
    workflow.add_edge(START, "route_query")
    workflow.add_edge("route_query", "execute_retrieval")
    workflow.add_edge("execute_retrieval", "grade_subgraph")
    workflow.add_edge("expand_graph_hops", "execute_retrieval")
    workflow.add_edge("synthesize_answer", END)

    # 3. Add Conditional Edge for Corrective RAG (CRAG)
    def decide_post_grade_action(state: GraphRAGAgentState) -> str:
        status = state.get("grader_status")
        if status == "NEEDS_HOP_EXPANSION":
            return "expand_graph_hops"
        return "synthesize_answer"

    workflow.add_conditional_edges(
        "grade_subgraph",
        decide_post_grade_action,
        {
            "expand_graph_hops": "expand_graph_hops",
            "synthesize_answer": "synthesize_answer",
        },
    )

    return workflow.compile()

