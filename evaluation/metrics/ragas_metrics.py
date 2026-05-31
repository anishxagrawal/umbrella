from __future__ import annotations


from evaluation.schema import EvalResult, MetricScore


GROQ_MODEL = "llama-3.1-8b-instant"


def build_ragas_dataset(results: list[EvalResult]):
    """
    Columns: question, answer, contexts (list[str]), ground_truth.
    Skip errored results.
    """
    from datasets import Dataset

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for result in results:
        if result.error:
            continue
        questions.append(result.sample.question)
        answers.append(result.generated_answer)
        contexts.append([chunk.text for chunk in result.retrieved_chunks])
        ground_truths.append(result.sample.gold_answer)

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def _metric_score(
    metric_name: str,
    score: float,
    passed_threshold: float,
) -> MetricScore:
    return MetricScore(
        metric_name=metric_name,
        score=float(score),
        reasoning="",
        passed=score >= passed_threshold,
    )


def score_ragas(results: list[EvalResult]) -> dict[str, list[MetricScore]]:
    """
    Run ragas_evaluate() with all 4 metrics.
    Return dict: metric_name -> list[MetricScore].
    passed threshold = 0.5 for all metrics.
    """
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from langchain_groq import ChatGroq

    dataset = build_ragas_dataset(results)
    if len(dataset) == 0:
        return {
            "faithfulness": [],
            "answer_relevancy": [],
            "context_precision": [],
            "context_recall": [],
        }

    llm = ChatGroq(model=GROQ_MODEL)
    result = ragas_evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
    )

    scores: dict[str, list[MetricScore]] = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
    }
    threshold = 0.5

    for row in result.to_pandas().to_dict("records"):
        scores["faithfulness"].append(
            _metric_score("faithfulness", row.get("faithfulness", 0.0), threshold)
        )
        scores["answer_relevancy"].append(
            _metric_score(
                "answer_relevancy", row.get("answer_relevancy", 0.0), threshold
            )
        )
        scores["context_precision"].append(
            _metric_score(
                "context_precision", row.get("context_precision", 0.0), threshold
            )
        )
        scores["context_recall"].append(
            _metric_score(
                "context_recall", row.get("context_recall", 0.0), threshold
            )
        )

    return scores
