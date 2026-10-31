from lazybridge import ClaudeCodeEngine, CodexEngine
from lazybridge.model_tiers import MODEL_TIERS, translate_model


def test_table_is_one_to_one():
    assert MODEL_TIERS["mid"] == {"anthropic": "claude-sonnet-5-5", "openai": "gpt-6.1-terra"}
    assert len({v["openai"] for v in MODEL_TIERS.values()}) == 4


def test_claude_names_translate_to_openai():
    assert translate_model("sonnet", "codex") == "gpt-6.1-terra"
    assert translate_model("opus", "openai") == "gpt-6.1-sol"
    assert translate_model("haiku", "codex") == "gpt-6-luna"
    assert translate_model("claude-fable-5-1", "codex") == "gpt-6-astra"
    assert translate_model("claude-sonnet-5", "codex") == "gpt-6.1-terra"


def test_openai_names_translate_to_claude():
    assert translate_model("gpt-6.1-terra", "claude_code") == "claude-sonnet-5-5"
    assert translate_model("gpt-6-astra", "anthropic") == "claude-fable-5-1"


def test_passthrough():
    assert translate_model(None, "codex") is None
    assert translate_model("gpt-5.6-sol", "codex") == "gpt-5.6-sol"
    assert translate_model("claude-opus-5-5", "claude_code") == "claude-opus-5-5"
    assert translate_model("sonnet", "claude_code") == "sonnet"


def test_engines_translate():
    assert CodexEngine(model="sonnet").model == "gpt-6.1-terra"
    assert CodexEngine(model=None).model is None
    assert ClaudeCodeEngine(model="gpt-6.1-sol").model == "claude-opus-5-5"
    assert ClaudeCodeEngine().model == "sonnet"
