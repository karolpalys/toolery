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


@pytest.mark.asyncio
@respx.mock
async def test_scan_skips_404():
    respx.get("http://localhost:9999/v1/models").mock(return_value=httpx.Response(404))
    assert await scan([9999]) == []


@pytest.mark.asyncio
@respx.mock
async def test_scan_handles_empty_models_list():
    respx.get("http://localhost:8081/v1/models").mock(
        return_value=httpx.Response(200, json={"object": "list", "data": []})
    )
    result = await scan([8081])
    assert len(result) == 1
    assert result[0].model_id == "(none loaded)"
    assert result[0].models == []


@pytest.mark.asyncio
@respx.mock
async def test_scan_skips_non_json_body():
    respx.get("http://localhost:8000/v1/models").mock(
        return_value=httpx.Response(200, content=b"<html>not json</html>",
                                    headers={"content-type": "text/html"})
    )
    assert await scan([8000]) == []


@pytest.mark.asyncio
@respx.mock
async def test_scan_skips_connection_refused():
    respx.get("http://localhost:7777/v1/models").mock(
        side_effect=httpx.ConnectError("refused")
    )
    assert await scan([7777]) == []


@pytest.mark.asyncio
@respx.mock
async def test_scan_skips_timeout():
    respx.get("http://localhost:6666/v1/models").mock(
        side_effect=httpx.TimeoutException("slow")
    )
    assert await scan([6666], timeout=0.1) == []


@pytest.mark.asyncio
@respx.mock
async def test_scan_skips_response_without_data_array():
    respx.get("http://localhost:5555/v1/models").mock(
        return_value=httpx.Response(200, json={"foo": "bar"})
    )
    assert await scan([5555]) == []


@pytest.mark.asyncio
@respx.mock
async def test_scan_returns_only_reachable_among_many():
    respx.get("http://localhost:8000/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "model-a"}]})
    )
    respx.get("http://localhost:8888/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "model-b"}]})
    )
    respx.get("http://localhost:9000/v1/models").mock(return_value=httpx.Response(404))
    respx.get("http://localhost:9100/v1/models").mock(side_effect=httpx.ConnectError("nope"))
    result = await scan([8000, 8888, 9000, 9100])
    assert {ep.port for ep in result} == {8000, 8888}
