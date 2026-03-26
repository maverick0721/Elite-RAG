import os
import re
from typing import Optional


def _get_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class RuleBasedLLM:
    """Deterministic fallback that keeps the pipeline runnable everywhere."""

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        if "Follow-up Query:" in prompt:
            question = self._extract_section(prompt, "Question:", "Context:")
            return question.strip() or "What is the key fact needed to answer the question?"

        if "Final Answer:" in prompt:
            answer = self._extract_section(prompt, "Answer:", "Final Answer:").strip()
            context = self._extract_section(prompt, "Context:", "Answer:").strip()
            if answer and self._answer_supported(answer, context):
                return answer
            return self._extractive_answer(prompt)

        return self._extractive_answer(prompt)

    def _extract_section(self, text: str, start: str, end: str) -> str:
        try:
            return text.split(start, 1)[1].split(end, 1)[0]
        except Exception:
            return ""

    def _answer_supported(self, answer: str, context: str) -> bool:
        answer_terms = {t for t in re.findall(r"[a-zA-Z0-9]+", answer.lower()) if len(t) > 3}
        if not answer_terms:
            return True
        context_terms = set(re.findall(r"[a-zA-Z0-9]+", context.lower()))
        overlap = len(answer_terms.intersection(context_terms))
        return overlap >= max(1, len(answer_terms) // 5)

    def _extractive_answer(self, prompt: str) -> str:
        question = self._extract_section(prompt, "Question:", "Context:").strip()
        context = self._extract_section(prompt, "Context:", "Answer:").strip()
        if not context:
            context = self._extract_section(prompt, "Context:", "Final Answer:").strip()
        if not context:
            return "I don't know."

        q_terms = {t for t in re.findall(r"[a-zA-Z0-9]+", question.lower()) if len(t) > 2}
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
        if not sentences:
            return "I don't know."

        def score(sentence: str) -> int:
            s_terms = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
            return len(q_terms.intersection(s_terms))

        best = max(sentences, key=score)
        return best if score(best) > 0 else "I don't know."


class LocalLLM:
    def __init__(self, model_name: str, backend: str = "auto", device: str = "auto"):
        self.backend = backend
        self.device = _get_device(device)
        self.model_name = model_name
        self.rule_llm = RuleBasedLLM()
        self.llm: Optional[object] = None
        self.generator = None
        self.sampling_params = None

        if backend in {"auto", "vllm"} and self.device == "cuda":
            try:
                # Fix Thunder GPU fork issue
                os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
                os.environ["VLLM_USE_V1"] = "0"
                os.environ["VLLM_V1_INPROC"] = "0"
                os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "300"
                from vllm import LLM, SamplingParams

                self.llm = LLM(
                    model=model_name,
                    tensor_parallel_size=1,
                    dtype="float16",
                    gpu_memory_utilization=0.7,
                )
                self.sampling_params = SamplingParams(temperature=0, max_tokens=512)
                self.backend = "vllm"
                return
            except Exception:
                if backend == "vllm":
                    raise

        if backend in {"auto", "transformers"}:
            try:
                from transformers import pipeline

                self.generator = pipeline(
                    "text-generation",
                    model=model_name,
                    device=0 if self.device == "cuda" else -1,
                )
                self.backend = "transformers"
                return
            except Exception:
                if backend == "transformers":
                    raise

        self.backend = "rule_based"

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        if self.backend == "vllm" and self.llm is not None:
            self.sampling_params.max_tokens = max_tokens
            outputs = self.llm.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text.strip()

        if self.backend == "transformers" and self.generator is not None:
            outputs = self.generator(prompt, max_new_tokens=max_tokens, do_sample=False)
            text = outputs[0]["generated_text"]
            if text.startswith(prompt):
                return text[len(prompt) :].strip() or "I don't know."
            return text.strip()

        return self.rule_llm.generate(prompt, max_tokens=max_tokens)