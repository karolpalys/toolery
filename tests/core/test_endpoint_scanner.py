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


import httpx
import pytest
import respx

from llm_test.core.endpoint_scanner import scan


@pytest.mark.asyncio
@respx.mock
async def test_scan_finds_single_endpoint():
    respx.get("http://localhost:8888/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "MiniMax-M2.7", "object": "model"}]},
            headers={"Server": "vLLM/0.7.0"},
        )
    )
    result = await scan([8888])
    assert len(result) == 1
    assert result[0].port == 8888
    assert result[0].base_url == "http://localhost:8888"
    assert result[0].model_id == "MiniMax-M2.7"
    assert result[0].models == ["MiniMax-M2.7"]
    assert result[0].server_hint == "vLLM"
