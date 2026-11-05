import json
import tomllib
from pathlib import Path

from lazybridge.model_tiers import MODEL_TIERS

ROOT = Path(__file__).resolve().parents[2]
INSTALL = 'pip install "lazytoolkit[mcp] @ git+https://github.com/selvaz/LazyTools@v0.10.6"'


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def test_agent_docs_version_and_model_tiers_match_source():
    contract = json.loads(read("lazybridge/llms.json"))
    project = tomllib.loads(read("pyproject.toml"))
    assert contract["version"] == project["project"]["version"]
    assert contract["coding_model_translation"]["tiers"] == {
        tier: [models["anthropic"], models["openai"]] for tier, models in MODEL_TIERS.items()
    }


def test_lazytools_install_is_consistent():
    contract = json.loads(read("lazybridge/llms.json"))
    context7 = json.loads(read("context7.json"))
    assert contract["integrations"]["install"] == INSTALL
    assert any(INSTALL in rule for rule in context7["rules"])
    for path in (
        "README.md",
        "lazybridge/skill/SKILL.md",
        "lazybridge/skill/references/tools-and-nesting.md",
        "docs/for-llms/codegen-contract.md",
    ):
        assert INSTALL in read(path), path
