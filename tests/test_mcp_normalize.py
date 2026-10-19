from mcp_client.manager import MCPClientManager


def test_invalid_parameter_not_auth_error():
    payload = {
        "isError": False,
        "structuredContent": {"message_detail": "invalid parameter 'x'"},
    }
    out = MCPClientManager._normalize_mcp_result(payload)
    assert out["success"] is True


def test_missing_api_key_is_auth_error():
    payload = {
        "isError": False,
        "structuredContent": {"message_detail": "missing api key"},
    }
    out = MCPClientManager._normalize_mcp_result(payload)
    assert out["success"] is False


def test_is_error_true():
    payload = {"isError": True, "content": [{"type": "text", "text": "boom"}]}
    out = MCPClientManager._normalize_mcp_result(payload)
    assert out["success"] is False
    assert "boom" in out["error"]
