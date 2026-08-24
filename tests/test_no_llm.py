"""The organism must not contain an LLM substrate, API, or hidden backend."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROHIBITED_IMPORT_ROOTS = {
    "openai",
    "anthropic",
    "transformers",
    "llama",
    "llama_cpp",
    "llama_cpp_python",
    "vllm",
    "together",
    "groq",
    "mistralai",
    "google.generativeai",
    "vertexai",
    "cohere",
    "ollama",
    "langchain",
    "llama_index",
    "litellm",
    "huggingface_hub",
    "peft",
    "bitsandbytes",
    "ctransformers",
    "exllamav2",
    "mlx_lm",
    "guidance",
    "semantic_kernel",
    "autogen",
    "crewai",
}

PROHIBITED_DEP_TOKENS = {
    "openai",
    "anthropic",
    "transformers",
    "llama-cpp",
    "llama_cpp",
    "vllm",
    "together",
    "groq",
    "mistralai",
    "google-generativeai",
    "vertexai",
    "cohere",
    "ollama",
    "langchain",
    "llama-index",
    "litellm",
    "huggingface-hub",
    "peft",
    "bitsandbytes",
    "ctransformers",
    "exllamav2",
    "mlx-lm",
}

PROHIBITED_URLS = (
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "api.groq.com",
    "api.together.xyz",
    "api.mistral.ai",
    "openrouter.ai",
    "api.deepseek.com",
)

ORGANISM_RUNTIME_DIRS = (
    "core",
    "organism",
    "language",
    "cognition",
    "world_model",
    "perception",
    "planning",
    "motivation",
    "memory",
    "self_model",
    "imagination",
    "learning",
    "embodiment",
    "simulation",
    "training",
    "apps/organism_runtime",
)

SKIP_PARTS = {".git", "node_modules", ".venv", "dist", "__pycache__", "runs", "research", "docs", "benchmarks"}


def _iter_files(*suffixes: str):
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(p in SKIP_PARTS for p in path.parts):
            continue
        if path.suffix in suffixes or path.name in suffixes:
            yield path


def _module_root(name: str) -> str:
    return name.split(".")[0]


def test_source_ast_has_no_llm_imports():
    hits = []
    for path in _iter_files(".py"):
        if path.name == "test_no_llm.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                root = _module_root(name)
                if root in PROHIBITED_IMPORT_ROOTS or name in PROHIBITED_IMPORT_ROOTS:
                    hits.append((str(path.relative_to(ROOT)), name))
    assert hits == [], hits


def test_dependency_manifests_have_no_llm_packages():
    hits = []
    manifests = [
        ROOT / "requirements.txt",
        ROOT / "pyproject.toml",
        ROOT / "Pipfile",
        ROOT / "apps" / "research-ui" / "package.json",
    ]
    for path in manifests:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for token in PROHIBITED_DEP_TOKENS:
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith("//"):
                    continue
                if token in stripped:
                    hits.append((str(path.relative_to(ROOT)), token, stripped))
    assert hits == [], hits


def test_configs_have_no_llm_backends():
    hits = []
    for path in _iter_files(".json", ".yml", ".yaml", ".toml", ".ini"):
        if "package-lock" in path.name:
            continue
        text = path.read_text(encoding="utf-8").lower()
        for token in ("model_provider", "llm_backend", "chat_model", "openai_api_key", "anthropic_api_key"):
            if token in text:
                hits.append((str(path.relative_to(ROOT)), token))
        for url in PROHIBITED_URLS:
            if url in text:
                hits.append((str(path.relative_to(ROOT)), url))
    assert hits == [], hits


def test_runtime_code_has_no_hosted_llm_urls():
    hits = []
    for rel in ORGANISM_RUNTIME_DIRS:
        root = ROOT / rel
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            for url in PROHIBITED_URLS:
                if url in text:
                    hits.append((str(path.relative_to(ROOT)), url))
            if "authorization: bearer" in text and "api." in text:
                hits.append((str(path.relative_to(ROOT)), "bearer-api"))
    assert hits == [], hits


def test_no_transformer_language_engine_in_core():
    hits = []
    for path in (ROOT / "core").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "nn.Transformer" in text or "TransformerDecoder" in text or "GPT2" in text:
            hits.append(str(path.relative_to(ROOT)))
    assert hits == [], hits
