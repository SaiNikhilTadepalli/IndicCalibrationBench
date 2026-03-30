import os
import re
from typing import Any, Callable

from inspect_ai.dataset import Dataset, Sample, csv_dataset

from inspect_evals.utils import create_stable_id

from prompts import CONFIDENCE_SCORE_REGEX, QUESTION_ANSWERING_TEMPLATE


# Resolve local QA dataset path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATASET_PATH = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "data", "indic_multicultural_qa.csv")
)


def apply_question_answering_template(question: str) -> str:
    """Wrap the raw question in the question-answering template."""
    return QUESTION_ANSWERING_TEMPLATE.format(question=question)


def get_record_to_sample() -> Callable[[dict[str, Any]], Sample]:
    def record_to_sample(record: dict[str, Any]) -> Sample:
        question = record.get("Question", "")

        return Sample(
            input=apply_question_answering_template(question),
            target=record["Answer"],
            id=create_stable_id(record["Question"]),
            metadata={"raw_question": question, "category": record["Category"]},
        )

    return record_to_sample


def load_indic_multicultural_qa_dataset(file_path: str = LOCAL_DATASET_PATH) -> Dataset:
    """
    Load the local CSV dataset of Indic multi-cultural QA pairs and
    apply the question-answering template to each sample.
    """
    return csv_dataset(csv_file=file_path, sample_fields=get_record_to_sample)


def extract_confidence_score(model_response: str) -> int | None:
    """Extract the confidence score from the model's question-answering response."""
    match = re.search(CONFIDENCE_SCORE_REGEX, model_response)

    if match:
        confidence_score = int(match.group(1))
        return min(max(confidence_score, 0), 100)  # Ensure score is between 0 and 100

    return None  # Return None if Regex parsing fails
