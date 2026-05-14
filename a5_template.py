from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any
from neo4j import GraphDatabase
from llm_loader import get_raw_pipeline, load_local_llm

@dataclass
class Intent:
    question_type: str
    keywords: list[str]
    aspect: str
    ambiguous: bool = False

class NLUnderstandingAgent:
    def run(self, question: str) -> Intent:
        pipeline = get_raw_pipeline()
        prompt = f"""Extract keywords and the main aspect from this university regulation question.
        Question: {question}
        Return JSON: {{"keywords": ["list", "of", "entities"], "aspect": "graduation/credits/etc", "type": "requirement/penalty"}}"""
        
        # Simple extraction logic (can be replaced with LLM call)
        keywords = [word for word in question.split() if len(word) > 3]
        return Intent(question_type="query", keywords=keywords, aspect="regulation_info")

class SecurityAgent:
    def run(self, question: str, intent: Intent) -> dict[str, str]:
        blocked = ["delete", "drop", "merge", "create", "set ", "bypass"]
        if any(p in question.lower() for p in blocked):
            return {"decision": "REJECT", "reason": "DML operations prohibited."}
        return {"decision": "ALLOW", "reason": "Read-only request."}

class QueryPlannerAgent:
    def run(self, intent: Intent) -> dict[str, Any]:
        # Strategy: Search rules first, then articles.
        # We target the 'action' and 'result' properties of Rule nodes.
        keyword_str = " ".join(intent.keywords)
        cypher = f"""
        CALL db.index.fulltext.queryNodes("rule_idx", "{keyword_str}") 
        YIELD node, score
        RETURN node.action AS action, node.result AS result, node.reg_name AS source
        LIMIT 5
        """
        return {"strategy": "fulltext_search", "cypher": cypher}

class QueryExecutionAgent:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        auth = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def run(self, plan: dict[str, Any]) -> dict[str, Any]:
        try:
            with self.driver.session() as session:
                result = session.run(plan["cypher"])
                rows = [dict(record) for record in result]
                return {"rows": rows, "error": None}
        except Exception as e:
            return {"rows": [], "error": str(e)}

class DiagnosisAgent:
    def run(self, execution: dict[str, Any]) -> dict[str, str]:
        if execution.get("error"):
            return {"label": "QUERY_ERROR", "reason": execution["error"]}
        if not execution.get("rows"):
            return {"label": "NO_DATA", "reason": "No results found in KG."}
        return {"label": "SUCCESS", "reason": "Found matching regulations."}

class QueryRepairAgent:
    def run(self, diagnosis: dict[str, str], original_plan: dict[str, Any], intent: Intent) -> dict[str, Any]:
        # If full-text rule search fails, try searching Article content directly
        keyword_str = " ".join(intent.keywords)
        repaired_cypher = f"""
        CALL db.index.fulltext.queryNodes("article_content_idx", "{keyword_str}") 
        YIELD node, score
        RETURN node.content AS action, 'Article Content' AS result, node.reg_name AS source
        LIMIT 3
        """
        return {"cypher": repaired_cypher, "strategy": "article_fallback"}

class ExplanationAgent:
    def run(self, question, intent, security, diagnosis, answer, repair_attempted) -> str:
        return f"System processed '{question}' via {intent.question_type}. Diagnosis: {diagnosis['label']}. Repair: {repair_attempted}."

def build_template_pipeline() -> dict[str, Any]:
    # Ensure LLM is loaded for agents that need it
    load_local_llm()
    return {
        "nlu": NLUnderstandingAgent(),
        "security": SecurityAgent(),
        "planner": QueryPlannerAgent(),
        "executor": QueryExecutionAgent(),
        "diagnosis": DiagnosisAgent(),
        "repair": QueryRepairAgent(),
        "explanation": ExplanationAgent(),
    }