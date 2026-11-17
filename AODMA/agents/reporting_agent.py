    
from datetime import datetime, timezone
import json

from agents.llm import get_llm


def reporting_agent(state: dict) -> dict:
    report_data = {
        "experiment": state.get("experiment_name", "AODMA"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "research": state.get("research_summary"),
        "selection": state.get("selection_result"),
        "execution": state.get("execution_result"),
        "evaluation": state.get("evaluation_result"),
    }

    prompt = f"""
You are the Reporting Agent of AODMA.

Create a concise technical experiment report.

Use only the supplied information.
Do not invent values or claim statistical significance.
Clearly identify missing or unavailable measurements.

Experiment data:
{json.dumps(report_data, indent=2, default=str)}

Structure:
1. Experiment overview
2. Research findings
3. Selected configuration and budget
4. Execution outcome
5. Evaluation results
6. Limitations and next steps
"""

    response = get_llm().invoke(prompt)

    report = str(response.content)

    messages = list(state.get("messages", []))
    messages.append("Reporting Agent generated the final report.")

    return {
        "final_report": report,
        "messages": messages,
        "status": "completed",
    }
