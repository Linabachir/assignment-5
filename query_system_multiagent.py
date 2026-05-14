from __future__ import annotations
from typing import Any
from agents.a5_template import build_template_pipeline

PIPELINE = build_template_pipeline()

def answer_question(question: str) -> dict[str, Any]:
    nlu = PIPELINE["nlu"]
    security_agent = PIPELINE["security"]
    planner = PIPELINE["planner"]
    executor = PIPELINE["executor"]
    diagnosis_agent = PIPELINE["diagnosis"]
    repair_agent = PIPELINE["repair"]
    explanation_agent = PIPELINE["explanation"]

    # 1. Understanding & Security
    intent = nlu.run(question)
    security = security_agent.run(question, intent)

    if security["decision"] == "REJECT":
        return {
            "answer": "Request rejected by security policy.",
            "safety_decision": "REJECT",
            "diagnosis": "QUERY_ERROR",
            "repair_attempted": False,
            "repair_changed": False,
            "explanation": "Security policy blocked the input due to unsafe patterns."
        }

    # 2. Planning & Execution
    plan = planner.run(intent)
    execution = executor.run(plan)
    diagnosis = diagnosis_agent.run(execution)

    # 3. Diagnosis & Repair Flow
    repair_attempted = False
    repair_changed = False
    
    if diagnosis["label"] in {"QUERY_ERROR", "NO_DATA"}:
        repair_attempted = True
        new_plan = repair_agent.run(diagnosis, plan, intent)
        if new_plan["cypher"] != plan["cypher"]:
            repair_changed = True
            execution = executor.run(new_plan)
            diagnosis = diagnosis_agent.run(execution)

    # 4. Final Answer Construction
    if diagnosis["label"] == "SUCCESS":
        # Format the Neo4j rows into a readable answer
        rows = execution["rows"]
        evidence = [f"[{r['source']}] {r['action']} -> {r['result']}" for r in rows]
        answer = "Found the following rules:\n" + "\n".join(evidence)
    elif diagnosis["label"] == "NO_DATA":
        answer = "I could not find any regulations matching your request in the database."
    else:
        answer = f"An error occurred during query execution: {diagnosis['reason']}"

    explanation = explanation_agent.run(question, intent, security, diagnosis, answer, repair_attempted)

    return {
        "answer": answer,
        "safety_decision": "ALLOW",
        "diagnosis": diagnosis["label"],
        "repair_attempted": repair_attempted,
        "repair_changed": repair_changed,
        "explanation": explanation,
    }

def run_multiagent_qa(question: str) -> dict[str, Any]:
    return answer_question(question)

if __name__ == "__main__":
    while True:
        q = input("\nQuestion (type exit): ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break
        result = answer_question(q)
        print("-" * 30)
        for k, v in result.items():
            print(f"{k.upper()}: {v}")