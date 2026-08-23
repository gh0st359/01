from pathlib import Path


IMPORT_RE = (
    "import openai",
    "from openai",
    "import anthropic",
    "from anthropic",
    "import transformers",
    "from transformers",
    "import llama",
    "from llama",
)


def test_repository_does_not_import_llms():
    root = Path(__file__).resolve().parents[1]
    hits = []
    for path in root.rglob("*.py"):
        if "node_modules" in path.parts or ".git" in path.parts or path.name == "test_no_llm.py":
            continue
        text = path.read_text(encoding="utf-8").lower()
        for token in IMPORT_RE:
            if token in text:
                hits.append((str(path), token))
    assert hits == [], hits
