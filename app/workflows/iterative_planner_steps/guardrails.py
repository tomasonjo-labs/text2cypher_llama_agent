from llama_index.core import ChatPromptTemplate
from pydantic import BaseModel, Field

from typing import Literal

from app.workflows.utils import llm


class Guardrail(BaseModel):
    """Guardrail"""

    decision: Literal["movie", "end"] = Field(
        description="Decision on whether the question is related to movies"
        
    )
guardrails_system_prompt = """As an intelligent assistant, your primary objective is to decide whether a given question is related to movies or not.
If the question is related to movies, output "movie". Otherwise, output "end".
To make this decision, assess the content of the question and determine if it refers to any movie, actor, director, film industry,
or related topics. Provide only the specified output: "movie" or "end"."""
# Refine Prompt
chat_refine_msgs = [
    (
        "system",
        guardrails_system_prompt,
    ),
    ("user", "The question is: {question}"),
]
guardrails_template = ChatPromptTemplate.from_messages(chat_refine_msgs)

def guardrails_step(question):
    guardrails_output = (
        llm.as_structured_llm(Guardrail)
        .complete(guardrails_template.format(question=question))
        .raw
    ).decision
    if guardrails_output == 'end':
        context = "The question is not about movies or their case, so I cannot answer this question"
        return {"next_event": "generate_final_answer", "arguments": {"context": context, "question": question}}
    return {"next_event": "generate_plan", "arguments": {}}