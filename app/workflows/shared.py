from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.neo4j import (
    CypherQueryCorrector,
    Neo4jPropertyGraphStore,
    Schema,
)
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Event


class SseEvent(Event):
    label: str
    message: str


graph_store = Neo4jPropertyGraphStore(
    username="recommendations",
    password="recommendations",
    database="recommendations",
    url="neo4j+s://demo.neo4jlabs.com:7687",
    enhanced_schema=True,
    create_indexes=False,
    timeout=10,
)

default_llm = OpenAI(model="gpt-4o-2024-11-20", temperature=0)

embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Cypher query corrector is experimental
corrector_schema = [
    Schema(el["start"], el["type"], el["end"])
    for el in graph_store.get_schema().get("relationships")
]
cypher_query_corrector = CypherQueryCorrector(corrector_schema)

"""
llm = Gemini(
    model="models/gemini-1.5-flash",
    api_key=""
)
"""

fewshot_examples = [
    {
        "question": "How many artists are there?",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie) RETURN count(DISTINCT a)",
    },
    {
        "question": "Which actors played in the movie Casino?",
        "query": "MATCH (m:Movie {title: 'Casino'})<-[:ACTED_IN]-(a) RETURN a.name",
    },
    {
        "question": "How many movies has Tom Hanks acted in?",
        "query": "MATCH (a:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie) RETURN count(m)",
    },
    {
        "question": "List all the genres of the movie Schindler's List",
        "query": "MATCH (m:Movie {title: 'Schindler's List'})-[:IN_GENRE]->(g:Genre) RETURN g.name",
    },
    {
        "question": "Which actors have worked in movies from both the comedy and action genres?",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g1:Genre), (a)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g2:Genre) WHERE g1.name = 'Comedy' AND g2.name = 'Action' RETURN DISTINCT a.name",
    },
    {
        "question": "Which directors have made movies with at least three different actors named 'John'?",
        "query": "MATCH (d:Person)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(a:Person) WHERE a.name STARTS WITH 'John' WITH d, COUNT(DISTINCT a) AS JohnsCount WHERE JohnsCount >= 3 RETURN d.name",
    },
    {
        "question": "Identify movies where directors also played a role in the film.",
        "query": "MATCH (p:Person)-[:DIRECTED]->(m:Movie), (p)-[:ACTED_IN]->(m) RETURN m.title, p.name",
    },
    {
        "question": "Find the actor with the highest number of movies in the database.",
        "query": "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) RETURN a.name, COUNT(m) AS movieCount ORDER BY movieCount DESC LIMIT 1",
    },
]