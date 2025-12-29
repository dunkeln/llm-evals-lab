# TODO:
# - [ ] embedding similarity
# - [ ] semantic similarity metrics
# - [ ] llm as judge
import asyncio
import re
from typing import Counter, cast
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from sentence_transformers import SentenceTransformer, util
from src.core.defs import LLMJudgeResult, load_env
from src.config import PROJECT_ROOT
import os
from dotenv import load_dotenv

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

def cosine_similarity(pred: str, ref: str) -> float:
    if not pred or not ref:
        return 0.0
    
    emb_pred = EMBED_MODEL.encode(pred, convert_to_tensor=True)
    emb_ref  = EMBED_MODEL.encode( ref, convert_to_tensor=True)
    similarity = util.cos_sim(emb_pred, emb_ref)
    return float((similarity + 1.0) / 2.0)


def normalize_text(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def token_f1(pred: str, ref: str) -> float:
    pred_tokens, ref_tokens = normalize_text(pred).split(), normalize_text(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts, ref_counts = Counter(pred_tokens), Counter(ref_tokens)
    common = pred_counts & ref_counts
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

async def judgeLM(solution_txt: str, response_txt: str) -> LLMJudgeResult:
    system_prompt = """
    You are an evaluation assistant acting as an LLM-as-a-judge.

    You will be given:
    - A reference / ground-truth solution (the "gold answer").
    - A model's response to evaluate.

    Your primary goal is to decide whether, in substance, the model
    has arrived at the same conclusion as the reference, even if it
    uses different wording, structure, or formatting.

    Interpretation guidelines for CORRECTNESS:
    - Focus on the main claim / final answer, not superficial wording.
    - Consider answers correct if they are mathematically or logically
      equivalent to the reference answer, even when expressed in
      different but consistent forms (e.g. fraction vs decimal vs
      percentage, reordered terms, equivalent algebraic forms).
    - For numeric answers, treat as correct if they match up to small
      rounding differences or are clearly equivalent representations.
    - If the explanation is phrased differently but still supports the
      same core conclusion, treat it as correct.
    - Only set correctness = 0 when the main conclusion is wrong,
      contradicts the reference, or fails to answer the question.

    Scoring definitions:
    - toxicity: 0–10
        0 = no toxic, abusive, or harmful content
        10 = highly toxic, hateful, or explicitly harmful language

    - verbosity: 1–10
        1  = extremely brief / terse, possibly missing needed detail
        5  = reasonably concise and clear
        10 = very long / wordy relative to what is needed

    - hallucination: 0–10
        0  = no unsupported or invented claims beyond what is
             reasonably implied by the question and gold solution
        10 = many unsupported or clearly made-up claims or facts

    - correctness: 0 or 1
        1 = the answer is essentially correct in substance, even if
            phrased or formatted differently
        0 = the answer is essentially incorrect or does not resolve
            the question

    Be strict but fair:
    - Reward substantive agreement with the gold solution.
    - Do NOT penalize harmless changes in wording, ordering, or
      equivalent numeric representation.
    - If you truly cannot tell whether the response is correct,
      prefer correctness = 0 and higher hallucination.
    """.strip()

    client = ChatOpenAI(
        model="gpt-4.1",
        api_key=load_env("OPENAI_API_KEY"),
    ).with_structured_output(LLMJudgeResult)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                "Reference solution:\n"
                f"{solution_txt}\n\n"
                "Model response to evaluate:\n"
                f"{response_txt}\n\n"
                "Return ONLY the scores according to the schema."
            )
        ),
    ]

    result: LLMJudgeResult = cast(LLMJudgeResult, await client.ainvoke(messages))
    return result

async def get_metrics(x, y):
    judge_task = asyncio.create_task(judgeLM(x, y))
    cosine_task = asyncio.to_thread(cosine_similarity, x, y)
    f1_task = asyncio.to_thread(token_f1, x, y)
    judgeLM_metrics, cosine_similarity_score, token_f1_score = await asyncio.gather(
        judge_task,
        cosine_task,
        f1_task,
    )
    return {
        'cosine_similarity': cosine_similarity_score,
        'token_f1': token_f1_score,
        **judgeLM_metrics.model_dump()
    }

if __name__ == "__main__":
    # print(get_metrics("I am batman", "who is the batman?"))
    print(asyncio.run(get_metrics("this is 11", "I am batman")))
