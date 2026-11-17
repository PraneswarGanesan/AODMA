
from typing import Any, TypedDict


class AODMAState(TypedDict, total=False):
    request: str
    experiment_name: str

    portfolio_path: str
    portfolio: dict[str, Any]
    research_summary: str

    budget: dict[str, float]
    selection_method: str
    selected_candidate: dict[str, Any]
    selection_result: dict[str, Any]

    runtime: dict[str, Any]
    execution_config: dict[str, Any]

    execution_result: dict[str, Any]
    evaluation_result: dict[str, Any]

    final_report: str

    messages: list[str]
    errors: list[str]
    status: str
