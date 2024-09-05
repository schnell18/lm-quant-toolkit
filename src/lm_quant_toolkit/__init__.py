"""Packages for LLM Quantization Evaluation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llm_quant_eval")
except PackageNotFoundError:
    __version__ = "unknown version"
