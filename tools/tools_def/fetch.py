"""URL 内容下载工具。

与 network.py 的 http_get 区别：
    - http_get 专注**文本**（API、网页文本）
    - fetch_url 专注**资源**（图片、PDF、压缩包等二进制）

安全约束：
    - SSRF 防护：拒绝内网 / 环回 / 元数据 IP
    - 单文件大小限制（20MB）
    - 超时限制
    - 只允许 http/https

返回契约：
    - 文本类（text/* 且非 html）→ {"success": True, "text": "..."}
    - 二进制类 → {"success": True, "resources": [{"kind", "filename", "mime", "content": bytes}]}
"""

from __future__ import annotations

import ipaddress
import logging
import mimetypes
import socket
from typing import Any, Dict
from urllib.parse import urlparse

import httpx

from .. import tool


logger = logging.getLogger(__name__)


_MAX_BYTES = 20 * 1024 * 1024      # 20MB
_TIMEOUT = 20.0
_MAX_REDIRECTS = 5

# 明确拒绝的 host
_BLOCKED_HOSTS = {
    "localhost",
    "metadata.google.internal",
    "169.254.169.254",
}


def _is_blocked_ip(ip_str: str) -> bool:
    """判断 IP 是否属于禁止访问的范围。"""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _check_url_safety(url: str) -> str:
    """返回空字符串表示安全；否则返回拒绝原因。"""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"仅支持 http/https，当前: {parsed.scheme or '(空)'}"

    host = parsed.hostname
    if not host:
        return "URL 缺少 hostname"

    host_lower = host.lower().strip(".")
    if host_lower in _BLOCKED_HOSTS:
        return f"禁止访问: {host_lower}"

    # 直接是 IP
    try:
        ip = ipaddress.ip_address(host_lower)
        if _is_blocked_ip(str(ip)):
            return f"禁止访问内网/环回地址: {host_lower}"
        return ""
    except ValueError:
        pass

    # 域名 → 解析到 IP，检查所有 A/AAAA 记录
    try:
        infos = socket.getaddrinfo(host_lower, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        return f"无法解析域名: {host_lower} ({exc})"

    for info in infos:
        ip_str = info[4][0]
        if _is_blocked_ip(ip_str):
            return f"域名解析到内网地址，已拒绝: {host_lower} → {ip_str}"

    return ""


def _guess_filename(url: str, mime: str) -> str:
    path = urlparse(url).path
    name = path.rsplit("/", 1)[-1] or "download"
    # 无扩展名时用 MIME 补
    if "." not in name:
        ext = mimetypes.guess_extension(mime) or ""
        if ext:
            name = f"{name}{ext}"
    return name


@tool(
    name="fetch_url",
    description=(
        "从 URL 下载资源。"
        "**返回的二进制资源会被系统自动渲染给用户**——"
        "图片显示为图、PDF 显示为预览、文件显示为下载链接。"
        "适用于用户希望「看到」「展示」「渲染」「下载」某个 URL 内容的场景。"
        "支持图片、PDF、文件等；纯文本（非 HTML）直接返回文本。"
        "不要用于内网/本地地址。"
    ),
    parameters={
        "url": {"type": "str", "description": "完整 http/https URL"},
        "filename": {
            "type": "str",
            "description": "可选：覆盖默认文件名",
        },
    },
    examples=[
        "fetch_url(url='https://www.baidu.com/favicon.ico')",
        "fetch_url(url='https://example.com/report.pdf')",
    ],
    category="network",
    danger_level="safe",
    resource_kinds=["image", "pdf", "file"],
)
def fetch_url(url: str, filename: str = "") -> Dict[str, Any]:
    """从 URL 下载内容。"""
    if not url or not isinstance(url, str):
        return {"success": False, "error": "缺少 url"}

    url = url.strip()
    reason = _check_url_safety(url)
    if reason:
        return {"success": False, "error": f"URL 安全检查失败: {reason}"}

    try:
        with httpx.Client(
            timeout=_TIMEOUT,
            follow_redirects=True,
            max_redirects=_MAX_REDIRECTS,
            headers={"User-Agent": "Mozilla/5.0 (AgentWorkbench)"},
        ) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()

                mime = (resp.headers.get("Content-Type") or "").split(";")[0].strip()
                if not mime:
                    mime = mimetypes.guess_type(urlparse(url).path)[0] or "application/octet-stream"

                # 大小检查：先看 Content-Length
                cl = resp.headers.get("Content-Length")
                if cl and cl.isdigit() and int(cl) > _MAX_BYTES:
                    return {
                        "success": False,
                        "error": f"文件超过大小限制 ({int(cl)} > {_MAX_BYTES} bytes)",
                    }

                # 流式读取，边读边检查
                chunks = []
                total = 0
                for chunk in resp.iter_bytes(chunk_size=65536):
                    total += len(chunk)
                    if total > _MAX_BYTES:
                        return {
                            "success": False,
                            "error": f"下载超过大小限制 ({_MAX_BYTES} bytes)",
                        }
                    chunks.append(chunk)

                content = b"".join(chunks)
                name = filename or _guess_filename(url, mime)

    except httpx.HTTPStatusError as exc:
        return {
            "success": False,
            "error": f"HTTP {exc.response.status_code}: {exc.response.reason_phrase}",
        }
    except httpx.TimeoutException:
        return {"success": False, "error": f"请求超时（{_TIMEOUT}s）"}
    except httpx.HTTPError as exc:
        return {"success": False, "error": f"网络错误: {exc}"}
    except Exception as exc:
        logger.exception("[fetch_url] 未预期错误: %s", url)
        return {"success": False, "error": f"下载失败: {exc}"}

    # ── 分支 1：HTML 或纯文本 → 走 text ──
    if mime == "text/html":
        try:
            text = content.decode("utf-8", errors="replace")
        except Exception:
            text = ""
        return {
            "success": True,
            "message": f"已下载 HTML 页面（{len(content)} 字节）",
            "text": text[:50000],
            "url": url,
            "mime": mime,
            "size": len(content),
        }

    if mime.startswith("text/"):
        try:
            text = content.decode("utf-8", errors="replace")
        except Exception:
            text = ""
        return {
            "success": True,
            "message": f"已下载文本（{len(content)} 字节）",
            "text": text[:50000],
            "url": url,
            "mime": mime,
            "size": len(content),
        }

    # ── 分支 2：二进制 → 声明 resources ──
    return {
        "success": True,
        "message": f"已下载 {name}（{mime}, {len(content)} 字节）",
        "url": url,
        "mime": mime,
        "size": len(content),
        "resources": [
            {
                "kind": "file",          # resource_store 会按 mime 精确判 kind
                "filename": name,
                "mime": mime,
                "content": content,      # bytes
            },
        ],
    }