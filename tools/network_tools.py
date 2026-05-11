"""
网络接口请求工具
提供HTTP请求功能，支持GET/POST等方法
"""
import json
import urllib.request
import urllib.error
import urllib.parse
import ssl
import re
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse

from . import tool, get_registry


ALLOWED_PROTOCOLS = ['http', 'https']
BLOCKED_DOMAINS = [
    'localhost',
    '127.0.0.1',
    '0.0.0.0',
    '::1',
]

DANGEROUS_PORTS = [21, 22, 23, 25, 3306, 5432, 6379, 27017, 11211]


def is_url_safe(url: str) -> tuple[bool, str]:
    """检查URL是否安全"""
    try:
        parsed = urlparse(url)

        if parsed.scheme not in ALLOWED_PROTOCOLS:
            return False, f"不支持的协议: {parsed.scheme}"

        if not parsed.netloc:
            return False, "无效的URL"

        domain = parsed.netloc.split(':')[0].lower()

        if domain in BLOCKED_DOMAINS:
            return False, f"禁止访问: {domain}"

        if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', domain):
            if domain == '127.0.0.1' or domain == '0.0.0.0' or domain.startswith('192.168.') or domain.startswith('10.') or domain.startswith('172.'):
                return False, f"禁止访问内网IP: {domain}"

        if ':' in parsed.netloc:
            try:
                port = int(parsed.netloc.split(':')[1])
                if port in DANGEROUS_PORTS:
                    return False, f"禁止访问危险端口: {port}"
            except (ValueError, IndexError):
                pass

        return True, ""

    except Exception as e:
        return False, f"URL解析错误: {str(e)}"


def make_request(method: str, url: str, headers: Dict[str, str] = None,
                  data: Any = None, timeout: int = 30) -> Dict[str, Any]:
    """发起HTTP请求"""
    safe, error = is_url_safe(url)
    if not safe:
        return {"success": False, "error": error}

    try:
        context = ssl.create_default_context()

        if headers is None:
            headers = {}

        if 'User-Agent' not in headers:
            headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

        if data is not None:
            if isinstance(data, (dict, list)):
                data = json.dumps(data).encode('utf-8')
                if 'Content-Type' not in headers:
                    headers['Content-Type'] = 'application/json'
            elif isinstance(data, str):
                data = data.encode('utf-8')

        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        with urllib.request.urlopen(req, context=context, timeout=timeout) as response:
            status_code = response.status
            response_headers = dict(response.headers)

            content_type = response.headers.get('Content-Type', '')
            if 'application/json' in content_type:
                try:
                    body = json.loads(response.read().decode('utf-8'))
                except:
                    body = response.read().decode('utf-8')
            elif 'text/' in content_type:
                body = response.read().decode('utf-8')
            else:
                body = f"[二进制内容，长度: {response.length} bytes]"

            return {
                "success": True,
                "status_code": status_code,
                "status_message": response.reason,
                "headers": response_headers,
                "body": body,
                "url": url,
                "method": method
            }

    except urllib.error.HTTPError as e:
        return {
            "success": False,
            "error": f"HTTP错误 {e.code}: {e.reason}",
            "status_code": e.code,
            "response_headers": dict(e.headers),
            "response_body": e.read().decode('utf-8', errors='replace') if e.fp else None
        }
    except urllib.error.URLError as e:
        return {"success": False, "error": f"网络错误: {str(e.reason)}"}
    except Exception as e:
        return {"success": False, "error": f"请求失败: {str(e)}"}


@tool(
    name="http_get",
    description="发送HTTP GET请求，获取网页内容或API数据",
    parameters={
        "url": {"type": "str", "description": "请求URL"},
        "headers": {"type": "dict", "description": "请求头（可选）"},
        "timeout": {"type": "int", "description": "超时时间（秒），默认30秒"}
    },
    examples=[
        "http_get(url='https://api.github.com/users/octocat')",
        "http_get(url='https://httpbin.org/get', timeout=10)"
    ],
    category="network",
    danger_level="safe"
)
def http_get(url: str, headers: Dict[str, str] = None, timeout: int = 30) -> Dict[str, Any]:
    """发送GET请求"""
    return make_request('GET', url, headers, None, timeout)


@tool(
    name="http_post",
    description="发送HTTP POST请求，常用于提交表单或JSON数据",
    parameters={
        "url": {"type": "str", "description": "请求URL"},
        "data": {"type": "any", "description": "请求数据（dict/list或字符串）"},
        "headers": {"type": "dict", "description": "请求头（可选）"},
        "timeout": {"type": "int", "description": "超时时间（秒），默认30秒"}
    },
    examples=[
        "http_post(url='https://httpbin.org/post', data={'key': 'value'})",
        "http_post(url='https://api.example.com/submit', data={'name': 'test', 'age': 20})"
    ],
    category="network",
    danger_level="safe"
)
def http_post(url: str, data: Any, headers: Dict[str, str] = None, timeout: int = 30) -> Dict[str, Any]:
    """发送POST请求"""
    return make_request('POST', url, headers, data, timeout)


@tool(
    name="http_put",
    description="发送HTTP PUT请求，用于更新资源",
    parameters={
        "url": {"type": "str", "description": "请求URL"},
        "data": {"type": "any", "description": "请求数据"},
        "headers": {"type": "dict", "description": "请求头（可选）"},
        "timeout": {"type": "int", "description": "超时时间（秒）"}
    },
    examples=[
        "http_put(url='https://api.example.com/resource/1', data={'name': 'updated'})"
    ],
    category="network",
    danger_level="safe"
)
def http_put(url: str, data: Any, headers: Dict[str, str] = None, timeout: int = 30) -> Dict[str, Any]:
    """发送PUT请求"""
    return make_request('PUT', url, headers, data, timeout)


@tool(
    name="http_delete",
    description="发送HTTP DELETE请求，用于删除资源",
    parameters={
        "url": {"type": "str", "description": "请求URL"},
        "headers": {"type": "dict", "description": "请求头（可选）"},
        "timeout": {"type": "int", "description": "超时时间（秒）"}
    },
    examples=[
        "http_delete(url='https://api.example.com/resource/1')"
    ],
    category="network",
    danger_level="safe"
)
def http_delete(url: str, headers: Dict[str, str] = None, timeout: int = 30) -> Dict[str, Any]:
    """发送DELETE请求"""
    return make_request('DELETE', url, headers, None, timeout)


@tool(
    name="check_url_status",
    description="检查URL是否可访问，返回状态码和响应时间",
    parameters={
        "url": {"type": "str", "description": "要检查的URL"}
    },
    examples=[
        "check_url_status(url='https://www.baidu.com')"
    ],
    category="network",
    danger_level="safe"
)
def check_url_status(url: str) -> Dict[str, Any]:
    """检查URL状态"""
    import time
    start_time = time.time()

    safe, error = is_url_safe(url)
    if not safe:
        return {"success": False, "error": error}

    try:
        context = ssl.create_default_context()

        req = urllib.request.Request(url, method='HEAD')
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

        with urllib.request.urlopen(req, context=context, timeout=10) as response:
            elapsed_time = (time.time() - start_time) * 1000

            return {
                "success": True,
                "url": url,
                "status_code": response.status,
                "status_message": response.reason,
                "response_time_ms": round(elapsed_time, 2),
                "content_type": response.headers.get('Content-Type', ''),
                "content_length": response.headers.get('Content-Length', 'Unknown'),
                "server": response.headers.get('Server', 'Unknown'),
                "accessible": True
            }

    except urllib.error.HTTPError as e:
        elapsed_time = (time.time() - start_time) * 1000
        return {
            "success": True,
            "url": url,
            "status_code": e.code,
            "status_message": e.reason,
            "response_time_ms": round(elapsed_time, 2),
            "accessible": False,
            "error": f"HTTP {e.code}: {e.reason}"
        }
    except Exception as e:
        elapsed_time = (time.time() - start_time) * 1000
        return {
            "success": False,
            "url": url,
            "response_time_ms": round(elapsed_time, 2),
            "error": str(e)
        }


@tool(
    name="parse_html",
    description="解析HTML内容，提取链接、标题、文本等",
    parameters={
        "html": {"type": "str", "description": "HTML内容"},
        "extract": {"type": "str", "description": "提取类型（links/title/text/all），默认all"}
    },
    examples=[
        "parse_html(html='<html><body><a href=\"http://example.com\">Link</a></body></html>', extract='links')"
    ],
    category="network",
    danger_level="safe"
)
def parse_html(html: str, extract: str = "all") -> Dict[str, Any]:
    """解析HTML内容"""
    try:
        from html.parser import HTMLParser

        class HTMLLinkParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.links = []
                self.title = ""
                self.text_content = []
                self.current_tag = None
                self.current_attrs = {}

            def handle_starttag(self, tag, attrs):
                self.current_tag = tag
                self.current_attrs = dict(attrs)
                if tag == 'a' and 'href' in self.current_attrs:
                    self.links.append(self.current_attrs['href'])

            def handle_endtag(self, tag):
                self.current_tag = None
                self.current_attrs = {}

            def handle_data(self, data):
                if self.current_tag == 'title':
                    self.title += data
                elif self.current_tag in ['p', 'div', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th', 'article']:
                    self.text_content.append(data.strip())

        parser = HTMLLinkParser()
        parser.feed(html)

        result = {}

        if extract in ["all", "links"]:
            result["links"] = parser.links
        if extract in ["all", "title"]:
            result["title"] = parser.title.strip()
        if extract in ["all", "text"]:
            result["text"] = " ".join(parser.text_content)

        result["success"] = True
        result["link_count"] = len(parser.links)

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="fetch_json",
    description="获取并解析JSON API数据",
    parameters={
        "url": {"type": "str", "description": "JSON API URL"},
        "key_path": {"type": "str", "description": "JSON键路径（可选，如 data.users.0.name）"}
    },
    examples=[
        "fetch_json(url='https://api.github.com/users/octocat')",
        "fetch_json(url='https://api.example.com/data', key_path='data.0.id')"
    ],
    category="network",
    danger_level="safe"
)
def fetch_json(url: str, key_path: str = None) -> Dict[str, Any]:
    """获取并解析JSON"""
    result = http_get(url)

    if not result.get("success"):
        return result

    try:
        body = result.get("body")
        if isinstance(body, str):
            data = json.loads(body)
        else:
            data = body

        if key_path:
            keys = key_path.split('.')
            for k in keys:
                if k.isdigit():
                    k = int(k)
                data = data[k]

        return {
            "success": True,
            "url": url,
            "data": data,
            "key_path": key_path
        }

    except Exception as e:
        return {"success": False, "error": f"JSON解析失败: {str(e)}"}
