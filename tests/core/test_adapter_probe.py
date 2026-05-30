from toolery.core.adapter_probe import AdapterStatus, available_adapters


def test_raw_adapter_always_available():
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
