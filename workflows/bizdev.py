from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
import re
from workflows.shared.local_fewshot_manager import LocalFewshotManager
from workflows.shared.sse_event import SseEvent
from workflows.steps.naive_text2cypher import (
    get_naive_final_answer_prompt,
)
from workflows.steps.bizdev import (
    generate_cypher_step
)


class SummarizeEvent(Event):
    question: str
    cypher: str
    context: str


class ExecuteCypherEvent(Event):
    question: str
    cypher: str

def extract_cypher_query(text):
    match = re.search(r"<cypher>(.*?)</cypher>", text, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_visualization_type(text):
    match = re.search(r"<visualization>(.*?)</visualization>", text)
    return match.group(1).strip() if match else None


class BizDev(Workflow):
    def __init__(self, llm, db, embed_model, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.llm = llm
        self.graph_store = db["graph_store"]
        self.fewshot_retriever = LocalFewshotManager()
        self.db_name = db["name"]

    @step
    async def generate_cypher(self, ctx: Context, ev: StartEvent) -> ExecuteCypherEvent:
        question = ev.input

        fewshot_examples = self.fewshot_retriever.get_fewshot_examples(
            question, self.db_name
        )

        output = await generate_cypher_step(
            self.llm,
            self.graph_store,
            question,
            fewshot_examples,
        )
        cypher_query = extract_cypher_query(output)
        visualization = extract_visualization_type(output)
        print(cypher_query, visualization)
        ctx.write_event_to_stream(
            SseEvent(
                label="Cypher generation",
                message=f"Generated Cypher: {cypher_query}",
            )
        )

        # Return for the next step
        return ExecuteCypherEvent(question=question, cypher=cypher_query)

    @step
    async def execute_query(
        self, ctx: Context, ev: ExecuteCypherEvent
    ) -> SummarizeEvent:
        try:
            # Hard limit to 100 records
            database_output = str(self.graph_store.structured_query(ev.cypher)[:100])
        except Exception as e:
            database_output = str(e)
        ctx.write_event_to_stream(
            SseEvent(
                message=f"Database output: {database_output}", label="Database output"
            )
        )
        return SummarizeEvent(
            question=ev.question, cypher=ev.cypher, context=database_output
        )

    @step
    async def summarize_answer(self, ctx: Context, ev: SummarizeEvent) -> StopEvent:
        naive_final_answer_prompt = get_naive_final_answer_prompt()
        gen = await self.llm.astream_chat(
            naive_final_answer_prompt.format_messages(
                context=ev.context, question=ev.question, cypher_query=ev.cypher
            )
        )
        final_answer = ""
        async for response in gen:
            final_answer += response.delta
            ctx.write_event_to_stream(
                SseEvent(
                    label="Final answer",
                    message=response.delta,
                )
            )

        stop_event = StopEvent(
            result={
                "cypher": ev.cypher,
                "question": ev.question,
                "answer": final_answer,
            }
        )

        # Return the final result
        return stop_event
