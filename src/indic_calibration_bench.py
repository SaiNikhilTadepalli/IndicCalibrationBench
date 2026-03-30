import os

from dotenv import load_dotenv

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate, system_message

from prompts import INDIC_CALIBRATION_SYSTEM_PROMPT
from scorers import indic_calibration_scorer
from utils import load_indic_multicultural_qa_dataset

# Load environment variables
load_dotenv()

# Default models for question answering and evaluation judge
DEFAULT_QA_LLM = os.getenv("INSPECT_MODEL", "groq/llama-3.3-70b-versatile")
DEFAULT_JUDGE_LLM = os.getenv("INSPECT_EVAL_MODEL", "mistral/mistral-medium-latest")

# Default evaluation settings
DEFAULT_EPOCHS = 1
MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.3


@task
def indic_calibration_bench(
    use_system_prompt: bool | str = True,
    judge_llm: str | None = DEFAULT_JUDGE_LLM,
    epochs: int | None = DEFAULT_EPOCHS,
) -> Task:
    # Allow boolean arguments to also be passed as strings
    if isinstance(use_system_prompt, str):
        use_system_prompt = use_system_prompt.lower() == "true"

    # Load questions and apply the question-answering template
    dataset = load_indic_multicultural_qa_dataset()

    solvers = []
    if use_system_prompt:
        solvers.append(system_message(INDIC_CALIBRATION_SYSTEM_PROMPT))

    # Generate the LLM answer to the question
    solvers.append(generate())

    # Create and return the evaluation task
    return Task(
        dataset=dataset,
        solvers=solvers,
        scorer=indic_calibration_scorer(judge_llm),
        config=GenerateConfig(temperature=DEFAULT_TEMPERATURE, max_tokens=MAX_TOKENS),
        epochs=epochs,
    )
