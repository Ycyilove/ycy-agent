"""Thin wrapper: actual implementation in tools/tools_def/network.py."""

from .tools_def.network import *  # noqa: F401,F403
from .tools_def.network import (  # noqa: F401
    http_get,
    http_post,
    http_put,
    http_delete,
    check_url_status,
    parse_html,
    fetch_json,
    is_url_safe,
    make_request,
)