import sys
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.db.schema import init_database
from src.ingestion.pipeline import GraphIngestionPipeline
from src.ingestion.seed_data import CLINICAL_CORPUS_6_DOCS
from src.retrieval.retriever import TriModalRetriever
from src.agent.workflow import build_graphrag_agent
from src.agent.state import GraphRAGAgentState

console = Console()


def display_scenario_results(scenario_num: int, title: str, query: str, state: GraphRAGAgentState):
    console.print(
        Panel(
            f"[bold cyan]SCENARIO {scenario_num}: {title}[/bold cyan]\n"
            f"[bold white]Query:[/bold white] \"{query}\"",
            border_style="cyan",
        )
    )

    # 1. Decision & Telemetry Table
    status_style = (
        "green"
        if state["grader_status"] == "SUFFICIENT"
        else "yellow"
        if state["grader_status"] == "NEEDS_HOP_EXPANSION"
        else "red"
    )
    decision_table = Table(title="Agentic Decision & Execution Telemetry", show_lines=True)
    decision_table.add_column("Parameter", style="cyan", width=26)
    decision_table.add_column("Value / Details", style="white")

    decision_table.add_row("Classified Query Intent", f"[bold magenta]{state.get('query_intent')}[/bold magenta]")
    decision_table.add_row("Selected Retrieval Strategy", f"[bold green]{state.get('retrieval_strategy')}[/bold green]")
    decision_table.add_row("Graph Traversal Radius", f"{state.get('current_max_hops')} hops")
    decision_table.add_row("CRAG Autonomous Hop Expansions", f"{state.get('correction_attempts')} expansions")
    decision_table.add_row("Corrective Grader (CRAG) Status", f"[bold {status_style}]{state.get('grader_status')}[/bold {status_style}]")
    decision_table.add_row("CRAG Evaluator Reasoning", state.get("grader_reasoning", "N/A"))
    decision_table.add_row("Total Chunks Retrieved", str(len(state.get("retrieved_chunks", []))))
    decision_table.add_row("Discovered Entity Types", str(state.get("discovered_entity_types", [])))
    decision_table.add_row("End-to-End Agent Latency", f"[bold yellow]{state.get('total_latency_ms')} ms[/bold yellow]")

    console.print(decision_table)
    console.print("\n")

    # 2. Step-by-Step Agent Trace
    console.print("[bold yellow]Agent Execution Trace:[/bold yellow]")
    for step in state.get("agent_trace", []):
        console.print(f"  • {step}")
    console.print("\n")

    # 3. Traversed Knowledge Graph Paths (if applicable)
    paths = state.get("graph_paths", [])
    if paths:
        console.print("[bold yellow]Traversed Multi-Hop Knowledge Graph Paths:[/bold yellow]")
        for p in paths:
            hop_name = p.get("hop_ edge") or p.get("hop_edge", "")
            console.print(f"  • [bold gold1]{hop_name}[/bold gold1]  [dim]({p.get('evidence', '')})[/dim]")
        console.print("\n")
    else:
        console.print("[dim]  (Zero graph traversal edges engaged for this strategy/result)[/dim]\n")

    # 4. Grounded Final Answer & Citations
    border_color = "red" if state["grader_status"] == "INSUFFICIENT_ABSTAIN" else "green"
    console.print(
        Panel(
            state["final_answer"],
            title=f"[bold {border_color}]Grounded Agent Response & Path-Cited Synthesis[/bold {border_color}]",
            border_style=border_color,
        )
    )

    if state.get("citations"):
        console.print("[bold cyan]Citations & Provenance Tags:[/bold cyan]")
        for cit in state["citations"]:
            console.print(f"  • [dim]{cit}[/dim]")
    console.print("\n" + "=" * 80 + "\n")


def main():
    console.print(
        Panel.fit(
            "[bold white]PHASE 3 BENCHMARK: LangGraph Adaptive Router, Corrective Subgraph Grader (CRAG),\n"
            "Autonomous Hop Expansion & Path-Cited Synthesis Engine[/bold white]",
            border_style="blue",
        )
    )

    # 1. Setup & Ingest Entity-Resolved Knowledge Graph
    console.print("[bold blue]Initializing Database & Seeding Entity-Resolved Knowledge Graph...[/bold blue]")
    db = init_database()
    pipeline = GraphIngestionPipeline(db)
    pipeline.truncate_all_collections()
    ingest_stats = pipeline.ingest_documents(CLINICAL_CORPUS_6_DOCS, enable_deduplication=True)
    console.print(
        f"[green][OK] Ingestion Complete:[/green] Chunks={ingest_stats['chunks_processed']} | "
        f"Entities Created={ingest_stats['canonical_entities_created']} | "
        f"Entities Merged={ingest_stats['entities_merged']} | "
        f"Edges={ingest_stats['edges_created']}\n"
    )

    # 2. Compile LangGraph Agent
    retriever = TriModalRetriever(db)
    agent = build_graphrag_agent(retriever)

    # 3. Define 4 Test Scenarios
    scenarios = [
        {
            "num": 1,
            "title": "Fast Lexical Lookup (Zero Graph Overhead)",
            "query": "What exact dosage of Fluoxetine Hydrochloride and Tramadol HCl was administered in the emergency ward admission note?",
        },
        {
            "num": 2,
            "title": "Direct 1-Hop Relational Query",
            "query": "Which specific hepatic cytochrome P450 enzyme is potently inhibited by Fluconazole?",
        },
        {
            "num": 3,
            "title": "Deep Multi-Hop Causal Query (Autonomous CRAG Hop Expansion 1 -> 2)",
            "query": "A patient taking Fluoxetine is prescribed Tramadol. Trace the complete enzymatic interaction mechanism and identify the severe adverse syndrome that results.",
        },
        {
            "num": 4,
            "title": "Out-of-Domain Hallucination Guard",
            "query": "What is the metabolic interaction between Metformin and Lisinopril in renal failure patients?",
        },
    ]

    benchmark_summary = []

    # 4. Execute Scenarios
    for sc in scenarios:
        initial_state: GraphRAGAgentState = {
            "query": sc["query"],
            "agent_trace": [],
            "retrieved_chunks": [],
            "graph_paths": [],
            "discovered_entity_types": [],
            "correction_attempts": 0,
        }

        t_start = time.perf_counter()
        final_state = agent.invoke(initial_state)
        total_latency_ms = (time.perf_counter() - t_start) * 1000
        final_state["total_latency_ms"] = round(total_latency_ms, 2)

        display_scenario_results(sc["num"], sc["title"], sc["query"], final_state)

        benchmark_summary.append({
            "scenario": f"Scenario {sc['num']}",
            "intent": final_state.get("query_intent"),
            "strategy": final_state.get("retrieval_strategy"),
            "hops": f"{final_state.get('current_max_hops')} hops",
            "expansions": final_state.get("correction_attempts"),
            "grader": final_state.get("grader_status"),
            "latency": f"{final_state.get('total_latency_ms')} ms",
        })

    # 5. Phase 3 Benchmark Summary Table
    summary_table = Table(
        title="Phase 3 LangGraph Corrective GraphRAG Benchmark Summary",
        show_lines=True,
    )
    summary_table.add_column("Scenario", style="cyan", width=14)
    summary_table.add_column("Classified Intent", style="magenta", width=18)
    summary_table.add_column("Strategy", style="green", width=22)
    summary_table.add_column("Traversal Hops", style="white", width=16)
    summary_table.add_column("CRAG Expansions", style="yellow", justify="center", width=16)
    summary_table.add_column("CRAG Outcome", style="bold", width=22)
    summary_table.add_column("E2E Latency", style="bold yellow", justify="right", width=14)

    for row in benchmark_summary:
        grader_style = (
            "[bold green]SUFFICIENT[/bold green]"
            if row["grader"] == "SUFFICIENT"
            else "[bold red]INSUFFICIENT_ABSTAIN[/bold red]"
        )
        summary_table.add_row(
            row["scenario"],
            row["intent"],
            row["strategy"],
            row["hops"],
            str(row["expansions"]),
            grader_style,
            row["latency"],
        )

    console.print(summary_table)
    console.print("\n")


if __name__ == "__main__":
    main()
