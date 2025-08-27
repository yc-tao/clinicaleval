"""
Direct vLLM integration for clinical evaluation without lighteval dependencies.
"""
import os
from typing import Dict, Any, List

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None


def generate_with_vllm(inputs: List[str], cfg: Dict[str, Any]) -> List[str]:
    """
    Generate responses using vLLM directly.
    """
    if LLM is None:
        raise ImportError("vLLM not installed. Install with: pip install vllm")
    
    le_cfg = cfg.get('lighteval', {})
    model_path = le_cfg.get('model_path')
    
    if not model_path:
        raise ValueError("lighteval.model_path must be set for vLLM generation")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    # Initialize vLLM model
    model_params = le_cfg.get('model_parameters', {})
    llm = LLM(
        model=model_path,
        tensor_parallel_size=model_params.get('tensor_parallel_size', 1),
        trust_remote_code=True
    )
    
    # Configure sampling parameters
    gen_params = le_cfg.get('generation_parameters', {})
    sampling_params = SamplingParams(
        temperature=gen_params.get('temperature', 0.0),
        max_tokens=gen_params.get('max_new_tokens', 10),
        stop=["\n", "Answer:", "."]  # Stop early for yes/no responses
    )
    
    # Generate responses
    outputs = llm.generate(inputs, sampling_params)
    
    # Extract text from outputs
    responses = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        responses.append(generated_text)
    
    return responses