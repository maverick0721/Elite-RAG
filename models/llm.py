import os
import multiprocessing

# Fix Thunder GPU fork issue
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Force V0 and disable the experimental V1 components entirely
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_V1_INPROC"] = "0"

from vllm import LLM, SamplingParams
class LocalLLM:

    def __init__(self, model_name: str):
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            dtype="float16"
        )

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        params = SamplingParams(
            temperature=0,
            max_tokens=max_tokens
        )

        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text.strip()