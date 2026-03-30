import re
from typing import Any

from inspect_ai.model import Model, get_model
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    Value,
    metric,
    scorer,
)
from inspect_ai.solver import TaskState

import numpy as np

from prompts import ACCURACY_GRADING_PROMPT
from utils import extract_confidence_score


@metric
def ece_metric(n_bins: int = 10) -> Metric:
    """Calculate the Expected Calibration Error (ECE)."""

    def metric(scores: list[SampleScore]) -> Value:
        confidence_values = []
        accuracy_values = []

        for sample_score in scores:
            # Extract confidence and accuracy from metadata
            confidence = sample_score.score.metadata.get("confidence")
            accuracy = 1.0 if sample_score.score.value == "CORRECT" else 0.0

            if confidence is not None:
                # Normalise confidence score to [0, 1]
                confidence_values.append(confidence / 100.0)
                accuracy_values.append(accuracy)

        if not confidence_values:
            return 0.0

        confidence_values = np.array(confidence_values)
        accuracy_values = np.array(accuracy_values)

        # Define bin boundaries
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            # Find indices of samples that fall into the current bin
            bin_mask = (confidence_values > bin_boundaries[i]) & (
                confidence_values <= bin_boundaries[i + 1]
            )

            if np.any(bin_mask):  # If the bin is not empty
                bin_size = np.sum(bin_mask)
                bin_accuracy = np.mean(accuracy_values[bin_mask])
                bin_confidence = np.mean(confidence_values[bin_mask])

                # Weight the absolute difference by the proportion of samples in the bin
                ece += (bin_size / len(confidence_values)) * np.abs(
                    bin_accuracy - bin_confidence
                )

        return float(ece)

    return metric


@scorer(metrics=[ece_metric(n_bins=10)])
def indic_calibration_scorer(judge_model: str | Model | None) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        # Resolve judge model
        if judge_model is None:
            resolved_model = get_model(role="grader")
        else:
            resolved_model = get_model(judge_model)

        # Extract the question, correct answer and submitted answer
        question = state.metadata.get("raw_question", state.input_text)
        correct_answer = target.text
        submitted_answer = state.output.completion

        input_to_judge_model = ACCURACY_GRADING_PROMPT.format(
            question=question,
            correct_answer=correct_answer,
            submitted_answer=submitted_answer,
        )

        output_from_judge_model = await resolved_model.generate(input_to_judge_model)
        completion = output_from_judge_model.completion.strip()

        # Parse "CORRECT" or "INCORRECT" from the judge model's output
        is_correct = (
            "CORRECT"
            if "CORRECT" in completion.upper() and "INCORRECT" not in completion.upper()
            else "INCORRECT"
        )

        # Extract confidence score from the original model output
        confidence = extract_confidence_score(submitted_answer)

        return Score(
            value=is_correct,
            explanation=completion,
            metadata={"confidence": confidence, "completion": completion},
        )

    return score
