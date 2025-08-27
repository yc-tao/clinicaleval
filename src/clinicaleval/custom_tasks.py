"""
Custom lighteval tasks for clinical evaluation.
Following the official guide: https://huggingface.co/docs/lighteval/en/adding-a-custom-task
"""
from typing import Dict, Any

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics


def readmission_prompt_fn(line: Dict[str, Any], task_name: str = None) -> Doc:
    """
    Prompt function for 30-day readmission prediction task.
    Converts a dataset line to a Doc object for evaluation.
    """
    # Extract the medical text and format it with the template
    medical_text = line.get("text", line.get("input", ""))
    
    # Create the prompt with explicit yes/no instruction
    prompt = f"{medical_text}\n\nAnswer (yes or no):"
    
    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[" yes", " no"],  # Lighteval expects choices with leading space
        gold_index=0 if str(line.get("label", "no")).lower().strip() in {"yes", "1", "true"} else 1,
        instruction="",
        target_for_fewshot_sorting=line.get("label", "no")
    )


# Task configuration following the official guide format
# For now, use a simple HuggingFace dataset to test the task works
# Later we can adapt this to load custom local data
readmission_30day_task = LightevalTaskConfig(
    name="readmission_30day",
    prompt_function=readmission_prompt_fn,
    hf_repo="hf-internal-testing/fixtures_boolq",  # Use a test dataset with similar structure
    hf_subset="default", 
    hf_avail_splits=["train"],  # Available splits in the dataset
    evaluation_splits=["train"],  # Which splits to use for evaluation
    few_shots_split=None,  # No few-shot examples needed
    few_shots_select=None,  # No few-shot selection needed
    suite=["clinical"],  # Suite name for grouping
    generation_size=10,  # Max tokens to generate
    stop_sequence=["\n"],  # Stop sequences
    metric=[Metrics.exact_match, Metrics.f1_score],  # Metrics to compute
    trust_dataset=True,  # Trust dataset loading
)

# Register tasks in the required format
TASKS_TABLE = [readmission_30day_task]

# Group tasks by suite
TASKS_GROUPS = {
    "clinical": ["readmission_30day"]
}