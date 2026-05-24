from llm_test.core.endpoint_scanner import EndpointInfo


def test_endpoint_info_serializes_basic_fields():
    info = EndpointInfo(
        port=8888,
        base_url="http://localhost:8888",
        model_id="MiniMax-M2.7",
        models=["MiniMax-M2.7"],
        server_hint="vLLM",
    )
    assert info.port == 8888
    assert info.base_url == "http://localhost:8888"
    assert info.model_id == "MiniMax-M2.7"
    assert info.models == ["MiniMax-M2.7"]
    assert info.server_hint == "vLLM"
