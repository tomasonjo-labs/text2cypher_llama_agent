from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from app.workflows.shared import (
    SseEvent,
    default_llm,
    embed_model,
    fewshot_examples,
    graph_store,
)
from app.workflows.steps.naive_text2cypher import (
    generate_cypher_step,
    get_naive_final_answer_prompt,
)


class SummarizeEvent(Event):
    question: str
    cypher: str
    context: str


class ExecuteCypherEvent(Event):
    question: str
    cypher: str


class NaiveText2CypherFlow(Workflow):
    def __init__(self, llm=None, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Call the parent init
        self.llm = llm or default_llm  # Add child-specific logic

        # Add fewshot in-memory vector db
        few_shot_nodes = []
        for example in fewshot_examples:
            few_shot_nodes.append(
                TextNode(
                    text=f"{{'query':{example['query']}, 'question': {example['question']}))"
                )
            )
        few_shot_index = VectorStoreIndex(few_shot_nodes, embed_model=embed_model)
        self.few_shot_retriever = few_shot_index.as_retriever(similarity_top_k=5)

    @step
    async def generate_cypher(self, ctx: Context, ev: StartEvent) -> ExecuteCypherEvent:
        question = ev.input

        cypher_query = await generate_cypher_step(
            self.llm, question, self.few_shot_retriever
        )

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
            database_output = str(graph_store.structured_query(ev.cypher)[:100])
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
