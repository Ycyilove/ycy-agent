"""
MySQL 数据库访问工具
支持自然语言配置数据库连接
"""
import os
import json
import re
from typing import Dict, Any, Optional

try:
    import pymysql
    from pymysql.cursors import DictCursor
except ImportError:
    pymysql = None

from .. import tool


def parse_natural_config(config_str: str) -> Dict[str, Any]:
    """解析自然语言配置或JSON配置"""
    # 尝试 JSON 解析
    try:
        return json.loads(config_str)
    except (json.JSONDecodeError, TypeError):
        pass

    # 解析自然语言配置
    config = {}

    # 主机地址
    host_patterns = [
        r'主机[是为：:]\s*(\S+)',
        r'host[是为：:]\s*(\S+)',
        r'地址[是为：:]\s*(\S+)',
        r'服务器[是为：:]\s*(\S+)',
    ]
    for pattern in host_patterns:
        match = re.search(pattern, config_str, re.IGNORECASE)
        if match:
            config['host'] = match.group(1).strip('"\'')
            break

    # 端口
    port_match = re.search(r'端口[是为：:\s]+(\d+)', config_str, re.IGNORECASE)
    if port_match:
        config['port'] = int(port_match.group(1))

    # 用户名
    user_patterns = [
        r'用户[名是为：:]\s*(\S+)',
        r'username[是为：:]\s*(\S+)',
        r'user[是为：:]\s*(\S+)',
        r'账号[是为：:]\s*(\S+)',
    ]
    for pattern in user_patterns:
        match = re.search(pattern, config_str, re.IGNORECASE)
        if match:
            config['user'] = match.group(1).strip('"\'')
            break

    # 密码
    pwd_patterns = [
        r'密码[是为：:]\s*(\S+)',
        r'password[是为：:]\s*(\S+)',
        r'passwd[是为：:]\s*(\S+)',
    ]
    for pattern in pwd_patterns:
        match = re.search(pattern, config_str, re.IGNORECASE)
        if match:
            config['password'] = match.group(1).strip('"\'')
            break

    # 数据库名
    db_patterns = [
        r'数据库[是为：:]\s*(\S+)',
        r'database[是为：:]\s*(\S+)',
        r'db[是为：:]\s*(\S+)',
    ]
    for pattern in db_patterns:
        match = re.search(pattern, config_str, re.IGNORECASE)
        if match:
            config['database'] = match.group(1).strip('"\'')
            break

    return config


def merge_with_env(config: Dict[str, Any]) -> Dict[str, Any]:
    """合并环境变量配置"""
    result = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", 3306)),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", ""),
        "charset": os.getenv("MYSQL_CHARSET", "utf8mb4")
    }
    result.update(config)
    return result


_connection_pool: Dict[str, Any] = {}


def get_connection(config: Dict[str, Any]) -> Any:
    """获取数据库连接"""
    if pymysql is None:
        raise ImportError("请先安装 pymysql: pip install pymysql")

    conn_key = f"{config.get('host', 'localhost')}:{config.get('port', 3306)}/{config.get('database', '')}"

    if conn_key not in _connection_pool or not _connection_pool[conn_key].open:
        try:
            conn = pymysql.connect(
                host=config["host"],
                port=config["port"],
                user=config["user"],
                password=config["password"],
                database=config.get("database") or None,
                charset=config.get("charset", "utf8mb4"),
                cursorclass=DictCursor,
                autocommit=True,
                connect_timeout=10
            )
            _connection_pool[conn_key] = conn
        except Exception as e:
            return {"error": f"连接失败: {str(e)}"}

    return _connection_pool[conn_key]


def close_all_connections():
    """关闭所有连接"""
    for conn in _connection_pool.values():
        try:
            conn.close()
        except Exception:
            pass
    _connection_pool.clear()


@tool(
    name="mysql_connect",
    description="连接到 MySQL 数据库，支持自然语言配置或JSON格式",
    category="database",
    danger_level="medium",
    parameters={
        "config": {"type": "str", "description": "数据库配置，支持三种格式：1) JSON字符串 {\"host\":\"localhost\",\"port\":3306,\"user\":\"root\",\"password\":\"xxx\",\"database\":\"mydb\"}  2) 自然语言如'主机localhost，用户root，密码123，数据库mydb'  3) 留空使用环境变量"},
        "host": {"type": "str", "description": "数据库主机地址（当 config 为空时使用）"},
        "port": {"type": "int", "description": "端口（当 config 为空时使用）"},
        "user": {"type": "str", "description": "用户名（当 config 为空时使用）"},
        "password": {"type": "str", "description": "密码（当 config 为空时使用）"},
        "database": {"type": "str", "description": "数据库名（当 config 为空时使用）"}
    },
    examples=[
        'mysql_connect(config=\'{"host":"localhost","user":"root","password":"123456","database":"mydb"}\')',
        'mysql_connect(config="主机localhost，用户root，密码123456，数据库test")',
        'mysql_connect(host="192.168.1.100", user="admin", password="pass", database="mydb")'
    ]
)
def mysql_connect(
    config: str = "",
    host: str = "",
    port: int = 0,
    user: str = "",
    password: str = "",
    database: str = ""
) -> Dict[str, Any]:
    """测试数据库连接"""
    if config:
        parsed = parse_natural_config(config)
    else:
        parsed = {}
        if host: parsed['host'] = host
        if port: parsed['port'] = port
        if user: parsed['user'] = user
        if password: parsed['password'] = password
        if database: parsed['database'] = database

    cfg = merge_with_env(parsed)

    try:
        if pymysql is None:
            return {"error": "pymysql 未安装，请运行: pip install pymysql"}

        conn = pymysql.connect(
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["user"],
            password=cfg["password"],
            database=cfg.get("database") or None,
            charset=cfg["charset"],
            connect_timeout=10
        )
        version = conn.get_server_info()
        conn.close()

        return {
            "success": True,
            "message": f"成功连接到 {cfg['host']}:{cfg['port']}",
            "server_version": version
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="mysql_query",
    description="执行 SQL SELECT 查询",
    category="database",
    danger_level="medium",
    parameters={
        "sql": {"type": "str", "description": "SQL 查询语句 (SELECT)"},
        "config": {"type": "str", "description": "数据库配置，支持自然语言或JSON，留空使用环境变量"},
        "limit": {"type": "int", "description": "结果条数限制，默认 100"}
    },
    examples=[
        'mysql_query(sql="SELECT * FROM users LIMIT 10")',
        'mysql_query(sql="SELECT name, email FROM users WHERE age > 18", config="数据库test，用户root，密码123")',
        'mysql_query(sql="SELECT COUNT(*) FROM orders", config=\'{"database":"shop"}\')'
    ]
)
def mysql_query(
    sql: str,
    config: str = "",
    limit: int = 100
) -> Dict[str, Any]:
    """执行查询语句"""
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        return {"error": "只允许执行 SELECT 查询"}

    if "LIMIT" not in sql_upper:
        sql = f"{sql.rstrip(';')} LIMIT {limit}"

    if config:
        parsed = parse_natural_config(config)
    else:
        parsed = {}

    cfg = merge_with_env(parsed)

    try:
        conn = get_connection(cfg)
        if isinstance(conn, dict) and "error" in conn:
            return conn

        with conn.cursor() as cursor:
            cursor.execute(sql)
            results = cursor.fetchall()

            # 序列化特殊类型
            serialized = []
            for row in results:
                r = {}
                for k, v in row.items():
                    if hasattr(v, 'isoformat'):
                        r[k] = v.isoformat()
                    elif isinstance(v, bytes):
                        r[k] = v.decode('utf-8', errors='replace')
                    else:
                        r[k] = v
                serialized.append(r)

            return {
                "success": True,
                "count": len(serialized),
                "columns": list(results[0].keys()) if results else [],
                "results": serialized
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="mysql_execute",
    description="执行 INSERT/UPDATE/DELETE 写操作",
    category="database",
    danger_level="high",
    parameters={
        "sql": {"type": "str", "description": "SQL 写操作语句"},
        "config": {"type": "str", "description": "数据库配置，支持自然语言或JSON，留空使用环境变量"}
    },
    examples=[
        'mysql_execute(sql="INSERT INTO users (name, email) VALUES (\'张三\', \'zhangsan@example.com\')")',
        'mysql_execute(sql="UPDATE users SET age=20 WHERE id=1", config=\'{"database":"mydb"}\')'
    ]
)
def mysql_execute(
    sql: str,
    config: str = ""
) -> Dict[str, Any]:
    """执行写操作"""
    sql_upper = sql.strip().upper()
    if sql_upper.startswith("SELECT"):
        return {"error": "查询请使用 mysql_query"}
    if sql_upper.startswith(("DROP", "TRUNCATE", "ALTER")):
        return {"error": "禁止执行危险操作"}

    if config:
        parsed = parse_natural_config(config)
    else:
        parsed = {}

    cfg = merge_with_env(parsed)

    try:
        conn = get_connection(cfg)
        if isinstance(conn, dict) and "error" in conn:
            return conn

        with conn.cursor() as cursor:
            affected = cursor.execute(sql)
            conn.commit()
            return {
                "success": True,
                "affected_rows": affected,
                "operation": sql_upper.split()[0]
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="mysql_show_tables",
    description="查看数据库中的所有表",
    category="database",
    danger_level="safe",
    parameters={
        "config": {"type": "str", "description": "数据库配置，支持自然语言或JSON"}
    },
    examples=[
        'mysql_show_tables(config=\'{"database":"mydb"}\')',
        'mysql_show_tables(config="数据库mydb")'
    ]
)
def mysql_show_tables(config: str = "") -> Dict[str, Any]:
    """获取表列表"""
    if config:
        parsed = parse_natural_config(config)
    else:
        parsed = {}

    cfg = merge_with_env(parsed)
    if not cfg.get("database"):
        return {"error": "需要指定数据库名"}

    try:
        conn = get_connection(cfg)
        if isinstance(conn, dict) and "error" in conn:
            return conn

        with conn.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            tables = [list(r.values())[0] for r in cursor.fetchall()]
            return {
                "success": True,
                "database": cfg["database"],
                "tables": tables,
                "count": len(tables)
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="mysql_describe_table",
    description="查看表结构（字段、类型、约束）",
    category="database",
    danger_level="safe",
    parameters={
        "table": {"type": "str", "description": "表名"},
        "config": {"type": "str", "description": "数据库配置，支持自然语言或JSON"}
    },
    examples=[
        'mysql_describe_table(table="users")',
        'mysql_describe_table(table="orders", config=\'{"database":"shop"}\')'
    ]
)
def mysql_describe_table(
    table: str,
    config: str = ""
) -> Dict[str, Any]:
    """查看表结构"""
    if config:
        parsed = parse_natural_config(config)
    else:
        parsed = {}

    cfg = merge_with_env(parsed)
    if not cfg.get("database"):
        return {"error": "需要指定数据库名"}

    try:
        conn = get_connection(cfg)
        if isinstance(conn, dict) and "error" in conn:
            return conn

        with conn.cursor() as cursor:
            cursor.execute(f"DESCRIBE `{table}`")
            columns = [{
                "field": r["Field"],
                "type": r["Type"],
                "null": r["Null"],
                "key": r["Key"],
                "default": r["Default"]
            } for r in cursor.fetchall()]

            return {
                "success": True,
                "table": table,
                "columns": columns
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="mysql_show_databases",
    description="查看 MySQL 服务器上所有数据库",
    category="database",
    danger_level="safe",
    parameters={
        "config": {"type": "str", "description": "数据库配置，支持自然语言或JSON"}
    },
    examples=[
        'mysql_show_databases()',
        'mysql_show_databases(config=\'{"host":"192.168.1.100","user":"admin","password":"pass"}\')'
    ]
)
def mysql_show_databases(config: str = "") -> Dict[str, Any]:
    """显示所有数据库"""
    if config:
        parsed = parse_natural_config(config)
    else:
        parsed = {}

    cfg = merge_with_env(parsed)
    cfg["database"] = None  # 不指定数据库

    try:
        conn = get_connection(cfg)
        if isinstance(conn, dict) and "error" in conn:
            return conn

        with conn.cursor() as cursor:
            cursor.execute("SHOW DATABASES")
            dbs = [list(r.values())[0] for r in cursor.fetchall()]
            return {
                "success": True,
                "databases": dbs,
                "count": len(dbs)
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="mysql_count",
    description="统计表中的记录数",
    category="database",
    danger_level="safe",
    parameters={
        "table": {"type": "str", "description": "表名"},
        "where": {"type": "str", "description": "WHERE 条件（可选）"},
        "config": {"type": "str", "description": "数据库配置，支持自然语言或JSON"}
    },
    examples=[
        'mysql_count(table="users")',
        'mysql_count(table="orders", where="status=\'completed\'")'
    ]
)
def mysql_count(
    table: str,
    where: str = "",
    config: str = ""
) -> Dict[str, Any]:
    """统计记录数"""
    if config:
        parsed = parse_natural_config(config)
    else:
        parsed = {}

    cfg = merge_with_env(parsed)
    if not cfg.get("database"):
        return {"error": "需要指定数据库名"}

    sql = f"SELECT COUNT(*) as total FROM `{table}`"
    if where:
        sql += f" WHERE {where}"

    try:
        conn = get_connection(cfg)
        if isinstance(conn, dict) and "error" in conn:
            return conn

        with conn.cursor() as cursor:
            cursor.execute(sql)
            result = cursor.fetchone()
            return {
                "success": True,
                "table": table,
                "count": result["total"]
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


__all__ = [
    "mysql_connect",
    "mysql_query",
    "mysql_execute",
    "mysql_show_tables",
    "mysql_describe_table",
    "mysql_show_databases",
    "mysql_count",
    "parse_natural_config",
    "close_all_connections"
]
