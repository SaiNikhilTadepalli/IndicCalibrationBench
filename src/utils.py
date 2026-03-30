import os
from typing import Any

from inspect_ai.dataset import Dataset, Sample, csv_dataset

from inspect_evals.utils import create_stable_id


# Resolve local QA dataset path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATASET_PATH = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "data", "indic_multicultural_qa.csv")
)


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["Question"],
        target=record["Answer"],
        id=create_stable_id(record["Question"]),
        metadata={"category": record["Category"]},
    )


def load_indic_multicultural_qa_dataset(file_path: str = LOCAL_DATASET_PATH) -> Dataset:
    """
    Load the local CSV dataset of Indic multi-cultural QA pairs.
    """
    return csv_dataset(csv_file=file_path, sample_fields=record_to_sample)
