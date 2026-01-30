from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from typing import TypedDict
import os

# 👉 SET YOUR API KEY DIRECTLY FOR COLAB
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# -----------------------------
# State definition
# -----------------------------
class AgentState(TypedDict):
    company: str
    data: str
    analysis: str

# -----------------------------
# Agent 1: Data Collector
# -----------------------------
def data_collector(state: AgentState):
    prompt = f"""
    Give a short business overview, recent performance,
    and risks for the company {state['company']}.
    """
    response = llm.invoke(prompt).content
    return {"data": response}

# -----------------------------
# Agent 2: Analyst
# -----------------------------
def analyst(state: AgentState):
    prompt = f"""
    Analyze the following company data and provide:
    1. Key insights
    2. Risk factors
    3. Future outlook

    Data:
    {state['data']}
    """
    response = llm.invoke(prompt).content
    return {"analysis": response}

# -----------------------------
# LangGraph Workflow
# -----------------------------
workflow = StateGraph(AgentState)
workflow.add_node("collector", data_collector)
workflow.add_node("analyst", analyst)

workflow.set_entry_point("collector")
workflow.add_edge("collector", "analyst")
workflow.set_finish_point("analyst")

app = workflow.compile()

# -----------------------------
# Execute 
# -----------------------------
company_name = "Tesla"

result = app.invoke({"company": company_name})

print("=== MULTI-AGENT SYSTEM OUTPUT ===\n")
print("Company:", company_name)
print("\n--- DATA COLLECTED ---")
print(result["data"])

print("\n--- ANALYSIS ---")
print(result["analysis"])


