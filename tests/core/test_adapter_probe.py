from llm_test.core.adapter_probe import AdapterStatus, available_adapters


def test_raw_adapter_always_available(monkeypatch):
    monkeypatch.delenv("CLAUDE_CLI_PATH", raising=False)
    monkeypatch.delenv("CODEX_CLI_PATH", raising=False)
    result = available_adapters(path_lookup=lambda _: None)
    assert isinstance(result["raw"], AdapterStatus)
    assert result["raw"].available is True
    assert result["raw"].reason is None


def test_hermes_available_when_in_path():
    result = available_adapters(
        path_lookup=lambda name: "/usr/local/bin/hermes" if name == "hermes" else None,
        env={},
    )
    assert result["hermes"].available is True


def test_hermes_unavailable_when_not_in_path():
    result = available_adapters(path_lookup=lambda _: None, env={})
    assert result["hermes"].available is False
    assert "PATH" in (result["hermes"].reason or "")


def test_claude_code_available_via_env():
    result = available_adapters(
        path_lookup=lambda _: None, env={"CLAUDE_CLI_PATH": "/opt/bin/claude"}
    )
    assert result["claude_code"].available is True


def test_claude_code_available_via_path():
    result = available_adapters(
        path_lookup=lambda name: "/usr/bin/claude" if name == "claude" else None,
        env={},
    )
    assert result["claude_code"].available is True


def test_claude_code_unavailable_without_either():
    result = available_adapters(path_lookup=lambda _: None, env={})
    assert result["claude_code"].available is False
    assert "CLAUDE_CLI_PATH" in (result["claude_code"].reason or "")


def test_codex_available_via_env():
    result = available_adapters(
        path_lookup=lambda _: None, env={"CODEX_CLI_PATH": "/opt/bin/codex"}
    )
    assert result["codex"].available is True


def test_codex_available_via_path():
    result = available_adapters(
        path_lookup=lambda name: "/usr/bin/codex" if name == "codex" else None,
        env={},
    )
    assert result["codex"].available is True


def test_codex_unavailable_without_either():
    result = available_adapters(path_lookup=lambda _: None, env={})
    assert result["codex"].available is False
    assert "CODEX_CLI_PATH" in (result["codex"].reason or "")
