from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.config import get_llm

class Scores(BaseModel):
    faithfulness: int = Field(ge=1, le=5, description="Are all claims supported by the context?")
    relevance: int = Field(ge=1, le=5, description="Does the answer address the question?")
    correctness: int = Field(ge=1, le=5, description="Matches the reference answer?")

PROMPT = ChatPromptTemplate.from_template(
    "You are a strict evaluator.\nContext:\n{ctx}\n\nQuestion: {q}\nReference: {ref}\nAnswer: {ans}\n\n"
    "Score 1-5 for faithfulness (grounded in context only), relevance, correctness.")

def judge(ctx, q, ref, ans) -> Scores:
    return (PROMPT | get_llm().with_structured_output(Scores)).invoke({"ctx": ctx, "q": q, "ref": ref, "ans": ans})
