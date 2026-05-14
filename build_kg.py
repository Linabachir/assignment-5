import os
import sqlite3
import json
import re
from typing import Any
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Assuming these are provided in your environment
from llm_loader import load_local_llm, get_raw_pipeline

# ========== 0) Initialization ==========
load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))

def extract_entities(article_number: str, reg_name: str, content: str) -> dict[str, Any]:
    """Uses Qwen2.5-3B to extract structured rules from regulation text."""
    pipeline = get_raw_pipeline()
    
    prompt = f"""
    Task: Extract legal rules from the following university regulation article.
    Format: Return ONLY a JSON object with a "rules" list.
    Each rule must have:
    - "type": (e.g., Requirement, Permission, Prohibition, Penalty)
    - "action": The condition or behavior described.
    - "result": The consequence or outcome.

    Article: {article_number} ({reg_name})
    Content: {content}

    Example JSON output:
    {{
        "rules": [
            {{"type": "Requirement", "action": "Students must complete 128 credits", "result": "Eligible for graduation"}}
        ]
    }}
    JSON:
    """
    
    try:
        # Local inference
        outputs = pipeline(prompt, max_new_tokens=512, do_sample=False)
        result_text = outputs[0]['generated_text'].split("JSON:")[-1].strip()
        
        # Clean up common LLM markdown noise
        result_text = re.sub(r'```json|```', '', result_text).strip()
        data = json.loads(result_text)
        return data
    except Exception as e:
        print(f"Extraction failed for {article_number}: {e}")
        return {"rules": []}

def build_graph() -> None:
    """Build KG from SQLite into Neo4j using the fixed assignment schema."""
    sql_conn = sqlite3.connect("ncu_regulations.db")
    cursor = sql_conn.cursor()
    driver = GraphDatabase.driver(URI, auth=AUTH)

    # Warm up local LLM
    print("Loading LLM...")
    load_local_llm()

    with driver.session() as session:
        print("Cleaning existing graph...")
        session.run("MATCH (n) DETACH DELETE n")

        # 1) Create Regulation nodes
        cursor.execute("SELECT reg_id, name, category FROM regulations")
        regulations = cursor.fetchall()
        reg_map = {reg[0]: (reg[1], reg[2]) for reg in regulations}

        for rid, name, cat in regulations:
            session.run(
                "MERGE (r:Regulation {id:$rid}) SET r.name=$name, r.category=$cat",
                rid=rid, name=name, cat=cat
            )

        # 2) Create Article nodes
        cursor.execute("SELECT reg_id, article_number, content FROM articles")
        articles = cursor.fetchall()

        print(f"Processing {len(articles)} articles...")
        for rid, art_num, content in articles:
            reg_name, reg_category = reg_map.get(rid, ("Unknown", "Unknown"))
            
            # Create Article node
            session.run(
                """
                MATCH (r:Regulation {id: $rid})
                CREATE (a:Article {
                    number:   $num,
                    content:  $content,
                    reg_name: $reg_name,
                    category: $reg_category
                })
                MERGE (r)-[:HAS_ARTICLE]->(a)
                """,
                rid=rid, num=art_num, content=content,
                reg_name=reg_name, reg_category=reg_category
            )

            # 3) Extract and Create Rule nodes
            extracted_data = extract_entities(art_num, reg_name, content)
            rules = extracted_data.get("rules", [])
            
            for i, rule_data in enumerate(rules):
                # Skip empty/invalid extractions
                if not rule_data.get("action") or not rule_data.get("result"):
                    continue
                
                rule_id = f"Rule-{art_num.replace(' ', '')}-{i}"
                
                session.run(
                    """
                    MATCH (a:Article {number: $art_num, reg_name: $reg_name})
                    CREATE (r:Rule {
                        rule_id:  $rule_id,
                        type:     $type,
                        action:   $action,
                        result:   $result,
                        art_ref:  $art_num,
                        reg_name: $reg_name
                    })
                    MERGE (a)-[:CONTAINS_RULE]->(r)
                    """,
                    art_num=art_num,
                    reg_name=reg_name,
                    rule_id=rule_id,
                    type=rule_data.get("type", "General"),
                    action=rule_data.get("action"),
                    result=rule_data.get("result")
                )

        # 4) Create full-text indexes
        print("Creating indexes...")
        session.run("CREATE FULLTEXT INDEX article_content_idx IF NOT EXISTS FOR (a:Article) ON EACH [a.content]")
        session.run("CREATE FULLTEXT INDEX rule_idx IF NOT EXISTS FOR (r:Rule) ON EACH [r.action, r.result]")

        # 5) Coverage audit
        coverage = session.run(
            """
            MATCH (a:Article)
            OPTIONAL MATCH (a)-[:CONTAINS_RULE]->(r:Rule)
            WITH a, count(r) AS rule_count
            RETURN count(a) AS total_articles,
                   sum(CASE WHEN rule_count > 0 THEN 1 ELSE 0 END) AS covered_articles,
                   sum(CASE WHEN rule_count = 0 THEN 1 ELSE 0 END) AS uncovered_articles
            """
        ).single()

        print("-" * 40)
        print(f"KG Construction Complete!")
        print(f"[Coverage] covered={coverage['covered_articles']}/{coverage['total_articles']}, "
              f"uncovered={coverage['uncovered_articles']}")

    driver.close()
    sql_conn.close()

if __name__ == "__main__":
    build_graph()