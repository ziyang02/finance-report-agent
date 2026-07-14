"""组装 LangGraph 状态图（核心文件）。

  START -> planner -> retriever -> analyst -> critic --pass/超轮次--> writer -> END
                          ^                        |
                          +------- revise ---------+   (反思回路，最多 max_revise 轮)
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents.analyst import analyst_node
from src.agents.critic import critic_node
from src.agents.planner import planner_node
from src.agents.retriever import make_retriever_node
from src.agents.state import ReportState
from src.agents.writer import writer_node
from src.config import settings


def route_after_critic(state: ReportState) -> str:
    """Critic 通过 或 达到轮次上限 -> 出报告；否则回退重检索（Self-RAG 式反思）。"""
    max_revise = settings()["agent"]["max_revise"]
    if state["critique"]["verdict"] == "pass" or state.get("revise_count", 0) >= max_revise:
        return "writer"
    return "retriever"


def build_graph(pipeline=None):
    """pipeline 可注入（生产传 RagPipeline.from_dir()，测试传 mock）。"""
    g = StateGraph(ReportState)
    g.add_node("planner", planner_node)
    g.add_node("retriever", make_retriever_node(pipeline))
    g.add_node("analyst", analyst_node)
    g.add_node("critic", critic_node)
    g.add_node("writer", writer_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "analyst")
    g.add_edge("analyst", "critic")
    g.add_conditional_edges("critic", route_after_critic,
                            {"retriever": "retriever", "writer": "writer"})
    g.add_edge("writer", END)
    return g.compile()


def run_report(target: str, pipeline=None) -> str:
    app = build_graph(pipeline)
    final = app.invoke({"target": target})
    return final["report"]
