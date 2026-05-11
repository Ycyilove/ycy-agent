"""
时间计算工具
提供日期时间计算、时间差、时区转换等功能
"""
import re
from datetime import datetime, timedelta, date
from typing import Dict, Any, Optional, Union, List
from enum import Enum

from . import tool, get_registry


class TimeUnit(Enum):
    """时间单位"""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


def parse_date_string(date_str: str) -> Optional[datetime]:
    """解析日期字符串"""
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y%m%d",
        "%Y%m%d%H%M%S",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m-%d-%Y",
        "%m/%d/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%Y年%m月%d日",
        "%Y年%m月%d日 %H:%M:%S",
    ]

    date_str = date_str.strip()

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def parse_relative_time(date_str: str, base_date: datetime = None) -> Optional[datetime]:
    """解析相对时间描述"""
    if base_date is None:
        base_date = datetime.now()

    date_str = date_str.lower().strip()

    patterns = [
        (r'(\d+)\s*年\s*前', lambda m: base_date - timedelta(days=int(m.group(1)) * 365)),
        (r'(\d+)\s*个月\s*前', lambda m: base_date - timedelta(days=int(m.group(1)) * 30)),
        (r'(\d+)\s*周\s*前', lambda m: base_date - timedelta(weeks=int(m.group(1)))),
        (r'(\d+)\s*天\s*前', lambda m: base_date - timedelta(days=int(m.group(1)))),
        (r'(\d+)\s*小时\s*前', lambda m: base_date - timedelta(hours=int(m.group(1)))),
        (r'(\d+)\s*分钟\s*前', lambda m: base_date - timedelta(minutes=int(m.group(1)))),
        (r'(\d+)\s*秒\s*前', lambda m: base_date - timedelta(seconds=int(m.group(1)))),

        (r'(\d+)\s*年\s*后', lambda m: base_date + timedelta(days=int(m.group(1)) * 365)),
        (r'(\d+)\s*个月\s*后', lambda m: base_date + timedelta(days=int(m.group(1)) * 30)),
        (r'(\d+)\s*周\s*后', lambda m: base_date + timedelta(weeks=int(m.group(1)))),
        (r'(\d+)\s*天\s*后', lambda m: base_date + timedelta(days=int(m.group(1)))),
        (r'(\d+)\s*小时\s*后', lambda m: base_date + timedelta(hours=int(m.group(1)))),
        (r'(\d+)\s*分钟\s*后', lambda m: base_date + timedelta(minutes=int(m.group(1)))),
        (r'(\d+)\s*秒\s*后', lambda m: base_date + timedelta(seconds=int(m.group(1)))),

        (r'(\d+)\s*years?\s*ago', lambda m: base_date - timedelta(days=int(m.group(1)) * 365)),
        (r'(\d+)\s*months?\s*ago', lambda m: base_date - timedelta(days=int(m.group(1)) * 30)),
        (r'(\d+)\s*weeks?\s*ago', lambda m: base_date - timedelta(weeks=int(m.group(1)))),
        (r'(\d+)\s*days?\s*ago', lambda m: base_date - timedelta(days=int(m.group(1)))),
        (r'(\d+)\s*hours?\s*ago', lambda m: base_date - timedelta(hours=int(m.group(1)))),
        (r'(\d+)\s*minutes?\s*ago', lambda m: base_date - timedelta(minutes=int(m.group(1)))),
        (r'(\d+)\s*seconds?\s*ago', lambda m: base_date - timedelta(seconds=int(m.group(1)))),

        (r'(\d+)\s*years?\s*later', lambda m: base_date + timedelta(days=int(m.group(1)) * 365)),
        (r'(\d+)\s*months?\s*later', lambda m: base_date + timedelta(days=int(m.group(1)) * 30)),
        (r'(\d+)\s*weeks?\s*later', lambda m: base_date + timedelta(weeks=int(m.group(1)))),
        (r'(\d+)\s*days?\s*later', lambda m: base_date + timedelta(days=int(m.group(1)))),
        (r'(\d+)\s*hours?\s*later', lambda m: base_date + timedelta(hours=int(m.group(1)))),
        (r'(\d+)\s*minutes?\s*later', lambda m: base_date + timedelta(minutes=int(m.group(1)))),
        (r'(\d+)\s*seconds?\s*later', lambda m: base_date + timedelta(seconds=int(m.group(1)))),

        (r'yesterday', lambda m: base_date - timedelta(days=1)),
        (r'today', lambda m: base_date),
        (r'tomorrow', lambda m: base_date + timedelta(days=1)),
        (r'now', lambda m: base_date),
    ]

    for pattern, handler in patterns:
        match = re.search(pattern, date_str)
        if match:
            return handler(match)

    return None


@tool(
    name="calculate_time_difference",
    description="计算两个日期时间之间的时间差，支持多种单位输出",
    parameters={
        "start_time": {"type": "str", "description": "开始时间（支持多种格式，如 2024-01-01, 2024/01/01 10:00:00）"},
        "end_time": {"type": "str", "description": "结束时间"},
        "unit": {"type": "str", "description": "输出单位（auto/seconds/minutes/hours/days/weeks），默认auto"}
    },
    examples=[
        "calculate_time_difference(start_time='2024-01-01', end_time='2024-01-02')",
        "calculate_time_difference(start_time='2024-01-01 10:00:00', end_time='2024-01-01 11:30:00', unit='minutes')"
    ],
    category="time",
    danger_level="safe"
)
def calculate_time_difference(start_time: str, end_time: str, unit: str = "auto") -> Dict[str, Any]:
    """计算两个日期时间之间的时间差"""
    try:
        start_dt = parse_date_string(start_time) or parse_relative_time(start_time)
        end_dt = parse_date_string(end_time) or parse_relative_time(end_time)

        if start_dt is None:
            return {"success": False, "error": f"无法解析开始时间: {start_time}"}
        if end_dt is None:
            return {"success": False, "error": f"无法解析结束时间: {end_time}"}

        diff = end_dt - start_dt
        total_seconds = diff.total_seconds()

        if unit == "auto" or unit == "seconds":
            result = total_seconds
            unit_display = "秒"
        elif unit == "minutes":
            result = total_seconds / 60
            unit_display = "分钟"
        elif unit == "hours":
            result = total_seconds / 3600
            unit_display = "小时"
        elif unit == "days":
            result = total_seconds / 86400
            unit_display = "天"
        elif unit == "weeks":
            result = total_seconds / 604800
            unit_display = "周"
        else:
            return {"success": False, "error": f"不支持的时间单位: {unit}"}

        return {
            "success": True,
            "start_time": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "difference_seconds": total_seconds,
            "difference": {
                "value": round(result, 2),
                "unit": unit_display
            },
            "human_readable": str(diff),
            "start_timestamp": start_dt.timestamp(),
            "end_timestamp": end_dt.timestamp()
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="add_time",
    description="在指定时间上增加或减少一段时间",
    parameters={
        "base_time": {"type": "str", "description": "基准时间（默认当前时间）"},
        "value": {"type": "int", "description": "要增加/减少的数值"},
        "unit": {"type": "str", "description": "时间单位（seconds/minutes/hours/days/weeks/months/years）"}
    },
    examples=[
        "add_time(base_time='2024-01-01', value=10, unit='days')",
        "add_time(value=2, unit='weeks')"
    ],
    category="time",
    danger_level="safe"
)
def add_time(base_time: str = None, value: int = 0, unit: str = "days") -> Dict[str, Any]:
    """时间加减计算"""
    try:
        if base_time:
            dt = parse_date_string(base_time) or parse_relative_time(base_time)
            if dt is None:
                return {"success": False, "error": f"无法解析时间: {base_time}"}
        else:
            dt = datetime.now()

        if unit in ["second", "seconds"]:
            dt = dt + timedelta(seconds=value)
        elif unit in ["minute", "minutes"]:
            dt = dt + timedelta(minutes=value)
        elif unit in ["hour", "hours"]:
            dt = dt + timedelta(hours=value)
        elif unit in ["day", "days"]:
            dt = dt + timedelta(days=value)
        elif unit in ["week", "weeks"]:
            dt = dt + timedelta(weeks=value)
        elif unit in ["month", "months"]:
            from calendar import monthrange
            month = dt.month - 1 + value
            year = dt.year + month // 12
            month = month % 12 + 1
            day = min(dt.day, monthrange(year, month)[1])
            dt = dt.replace(year=year, month=month, day=day)
        elif unit in ["year", "years"]:
            dt = dt.replace(year=dt.year + value)
        else:
            return {"success": False, "error": f"不支持的时间单位: {unit}"}

        return {
            "success": True,
            "base_time": base_time or "当前时间",
            "operation": f"{'+' if value >= 0 else ''}{value} {unit}",
            "result": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "result_date": dt.strftime("%Y-%m-%d"),
            "result_time": dt.strftime("%H:%M:%S"),
            "timestamp": dt.timestamp(),
            "weekday": dt.strftime("%A"),
            "is_weekend": dt.weekday() >= 5
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="get_current_time",
    description="获取当前时间，支持不同时区和格式化输出",
    parameters={
        "timezone": {"type": "str", "description": "时区（如 Asia/Shanghai, UTC, US/Eastern），默认本地时间"},
        "format": {"type": "str", "description": "输出格式（如 %Y-%m-%d %H:%M:%S）"}
    },
    examples=[
        "get_current_time()",
        "get_current_time(timezone='UTC')",
        "get_current_time(format='%Y年%m月%d日 %H:%M')"
    ],
    category="time",
    danger_level="safe"
)
def get_current_time(timezone: str = None, format: str = None) -> Dict[str, Any]:
    """获取当前时间"""
    try:
        now = datetime.now()

        if timezone:
            try:
                import pytz
                tz = pytz.timezone(timezone)
                now = datetime.now(tz)
            except ImportError:
                return {"success": False, "error": "需要安装 pytz 库支持时区"}
            except Exception:
                return {"success": False, "error": f"无效的时区: {timezone}"}

        if format:
            try:
                formatted = now.strftime(format)
            except Exception:
                return {"success": False, "error": f"无效的格式: {format}"}
        else:
            formatted = now.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "success": True,
            "datetime": str(now),
            "formatted": formatted,
            "timestamp": now.timestamp(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "weekday": now.strftime("%A"),
            "weekday_cn": ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()],
            "is_weekend": now.weekday() >= 5,
            "timezone": str(now.tzinfo) if now.tzinfo else "本地时间"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="format_timestamp",
    description="将时间戳或日期转换为指定格式",
    parameters={
        "input_value": {"type": "str", "description": "输入值（时间戳或日期字符串）"},
        "output_format": {"type": "str", "description": "输出格式（如 %Y-%m-%d %H:%M:%S）"}
    },
    examples=[
        "format_timestamp(input_value='2024-01-01', output_format='%Y年%m月%d日')",
        "format_timestamp(input_value='1704067200', output_format='%Y-%m-%d')"
    ],
    category="time",
    danger_level="safe"
)
def format_timestamp(input_value: str, output_format: str = "%Y-%m-%d %H:%M:%S") -> Dict[str, Any]:
    """时间戳/日期格式转换"""
    try:
        dt = None

        if input_value.isdigit():
            timestamp = int(input_value)
            if timestamp > 1e11:
                timestamp = timestamp / 1000
            dt = datetime.fromtimestamp(timestamp)
        else:
            dt = parse_date_string(input_value)
            if dt is None:
                dt = parse_relative_time(input_value)

        if dt is None:
            return {"success": False, "error": f"无法解析输入: {input_value}"}

        formatted = dt.strftime(output_format)

        return {
            "success": True,
            "input": input_value,
            "output": formatted,
            "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": dt.timestamp(),
            "date": dt.strftime("%Y-%m-%d"),
            "time": dt.strftime("%H:%M:%S"),
            "weekday": dt.strftime("%A")
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="calculate_age",
    description="根据出生日期计算年龄",
    parameters={
        "birth_date": {"type": "str", "description": "出生日期（如 1990-01-01）"},
        "reference_date": {"type": "str", "description": "参考日期（默认当前日期）"}
    },
    examples=[
        "calculate_age(birth_date='1990-01-01')",
        "calculate_age(birth_date='2000-06-15', reference_date='2024-01-01')"
    ],
    category="time",
    danger_level="safe"
)
def calculate_age(birth_date: str, reference_date: str = None) -> Dict[str, Any]:
    """计算年龄"""
    try:
        birth = parse_date_string(birth_date)
        if birth is None:
            return {"success": False, "error": f"无法解析出生日期: {birth_date}"}

        if reference_date:
            ref = parse_date_string(reference_date)
            if ref is None:
                ref = parse_relative_time(reference_date)
            if ref is None:
                return {"success": False, "error": f"无法解析参考日期: {reference_date}"}
        else:
            ref = datetime.now()

        age = ref.year - birth.year
        if (ref.month, ref.day) < (birth.month, birth.day):
            age -= 1

        total_days = (ref - birth).days

        return {
            "success": True,
            "birth_date": birth.strftime("%Y-%m-%d"),
            "reference_date": ref.strftime("%Y-%m-%d"),
            "age": age,
            "age_detail": {
                "years": age,
                "months": (ref.year - birth.year) * 12 + ref.month - birth.month - (1 if ref.day < birth.day else 0),
                "days": total_days
            },
            "total_days": total_days,
            "next_birthday": _get_next_birthday(birth, ref),
            "zodiac": _get_zodiac(birth.month, birth.day)
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def _get_next_birthday(birth: datetime, ref: datetime) -> str:
    """计算下一个生日"""
    try:
        this_year_birth = birth.replace(year=ref.year)
        if this_year_birth < ref:
            next_birth = birth.replace(year=ref.year + 1)
        else:
            next_birth = this_year_birth
        days_until = (next_birth - ref).days
        return {
            "date": next_birth.strftime("%Y-%m-%d"),
            "days_until": days_until
        }
    except Exception:
        return {}


def _get_zodiac(month: int, day: int) -> Optional[str]:
    """获取星座"""
    zodiac_data = [
        ((1, 20), "水瓶座"), ((2, 18), "双鱼座"), ((3, 20), "白羊座"),
        ((4, 19), "金牛座"), ((5, 20), "双子座"), ((6, 21), "巨蟹座"),
        ((7, 22), "狮子座"), ((8, 22), "处女座"), ((9, 22), "天秤座"),
        ((10, 22), "天蝎座"), ((11, 21), "射手座"), ((12, 21), "摩羯座")
    ]
    for (m, d), zodiac in zodiac_data:
        if (month, day) < (m, d):
            return zodiac
    return "摩羯座"
