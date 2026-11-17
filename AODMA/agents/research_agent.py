
import json
from pathlib import Path

from agents.llm import get_llm


def research_agent(state: dict) -> dict:
    portfolio_path = Path(
        state.get(
            "portfolio_path",
            "results/cifar10/bqrps/unified_portfolio.json",
        )
    )

    if not portfolio_path.exists():
        raise FileNotFoundError(
            f"Portfolio not found: {portfolio_path}"
        )

    with portfolio_path.open("r", encoding="utf-8") as file:
        portfolio = json.load(file)

    candidates = portfolio.get("candidates", [])

    if not candidates:
        raise ValueError("The candidate portfolio is empty.")

    candidate_summary = json.dumps(
        candidates,
        indent=2,
    )

    prompt = f"""
You are the Research Agent in AODMA.

Analyze the supplied diffusion inference candidate portfolio.

Your responsibilities:
1. Identify available samplers and NFE configurations.
2. Compare measured quality, latency and memory information.
3. Identify missing measurements or uncertainty.
4. Highlight possible trade-offs.
5. Do not invent results.
6. Do not select the final configuration.
   That responsibility belongs to the BQRPS Strategy Agent.

Candidate portfolio:
{candidate_summary}

Provide a concise technical research summary.
"""

    response = get_llm().invoke(prompt)

    summary = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )

    messages = list(state.get("messages", []))
    messages.append("Research Agent completed portfolio analysis.")

    return {
        "portfolio": portfolio,
        "research_summary": summary,
        "messages": messages,
        "status": "research_completed",
    }
