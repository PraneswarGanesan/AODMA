
from langgraph.graph import StateGraph, END

from agents.state import AODMAState
from agents.research_agent import research_agent
from agents.strategy_agent import strategy_agent
from agents.execution_agent import execution_agent
from agents.evaluation_agent import evaluation_agent
from agents.reporting_agent import reporting_agent


def build_aodma_graph():
    workflow = StateGraph(AODMAState)

    workflow.add_node("research", research_agent)
    workflow.add_node("strategy", strategy_agent)
    workflow.add_node("execution", execution_agent)
    workflow.add_node("evaluation", evaluation_agent)
    workflow.add_node("reporting", reporting_agent)

    workflow.set_entry_point("research")

    workflow.add_edge("research", "strategy")
    workflow.add_edge("strategy", "execution")
    workflow.add_edge("execution", "evaluation")
    workflow.add_edge("evaluation", "reporting")
    workflow.add_edge("reporting", END)

    return workflow.compile()


def run_aodma(
    portfolio_path,
    runtime,
    budget=None,
    selection_method="bqrps",
    samples=1000,
    batch_size=16,
    seed=42,
):
    graph = build_aodma_graph()

    initial_state = {
        "experiment_name": "AODMA Multi-Agent CIFAR-10",
        "portfolio_path": str(portfolio_path),
        "runtime": runtime,
        "budget": budget or {
            "max_latency_ms": 350.0,
            "max_memory_mb": 40.0,
            "max_nfe": 50,
        },
        "selection_method": selection_method,
        "execution_config": {
            "samples": samples,
            "batch_size": batch_size,
            "seed": seed,
            "output_dir": "results/agentic/generated",
        },
        "messages": [],
        "errors": [],
        "status": "started",
    }

    return graph.invoke(initial_state)
