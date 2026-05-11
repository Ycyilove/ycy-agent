"""
CSV/Excel数据读取分析工具
提供数据文件读取、统计分析、数据预览等功能
"""
import os
import csv
import json
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from io import StringIO, BytesIO

from . import tool, get_registry


def detect_delimiter(content: str) -> str:
    """自动检测CSV分隔符"""
    sample = content[:1024]
    delimiters = [',', ';', '\t', '|']
    counts = {}

    for delim in delimiters:
        count = sample.count(delim)
        if count > 0:
            counts[delim] = count

    if counts:
        return max(counts, key=counts.get)
    return ','


@tool(
    name="read_csv",
    description="读取CSV文件并进行数据分析，支持自动检测分隔符、数据预览、统计汇总",
    parameters={
        "file_path": {"type": "str", "description": "CSV文件路径"},
        "encoding": {"type": "str", "description": "文件编码，默认utf-8"},
        "max_rows": {"type": "int", "description": "最大读取行数，默认100行用于预览"}
    },
    examples=[
        "read_csv(file_path='data.csv')",
        "read_csv(file_path='data.csv', max_rows=10)"
    ],
    category="data",
    danger_level="safe"
)
def read_csv(file_path: str, encoding: str = "utf-8", max_rows: int = 100) -> Dict[str, Any]:
    """读取CSV文件并进行基础分析"""
    try:
        if not os.path.exists(file_path):
            return {"success": False, "error": f"文件不存在: {file_path}"}

        file_size = os.path.getsize(file_path)

        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            sample = f.read(4096)
            f.seek(0)
            delimiter = detect_delimiter(sample)

            reader = csv.DictReader(f, delimiter=delimiter)
            headers = reader.fieldnames or []

            rows = []
            column_stats = {h: {"type": "unknown", "count": 0, "null_count": 0, "unique_values": set()} for h in headers}

            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                rows.append(row)

                for header in headers:
                    value = row.get(header, "")
                    if value:
                        column_stats[header]["count"] += 1
                        column_stats[header]["unique_values"].add(value)

            for header in headers:
                column_stats[header]["unique_count"] = len(column_stats[header]["unique_values"])
                del column_stats[header]["unique_values"]

            total_rows = sum(1 for _ in open(file_path, 'r', encoding=encoding)) - 1
            is_preview = total_rows > max_rows

        return {
            "success": True,
            "file_info": {
                "path": file_path,
                "name": os.path.basename(file_path),
                "size": f"{file_size / 1024:.2f} KB",
                "encoding": encoding,
                "delimiter": repr(delimiter),
                "total_rows": total_rows,
                "total_columns": len(headers),
                "is_preview": is_preview
            },
            "columns": headers,
            "preview": rows[:10],
            "preview_rows": len(rows),
            "column_stats": column_stats,
            "message": f"成功读取 {len(rows)} 行数据（共 {total_rows} 行）" if is_preview else f"成功读取全部 {total_rows} 行数据"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="analyze_csv",
    description="对CSV文件进行深度数据分析，计算各列的统计信息",
    parameters={
        "file_path": {"type": "str", "description": "CSV文件路径"},
        "encoding": {"type": "str", "description": "文件编码，默认utf-8"}
    },
    examples=[
        "analyze_csv(file_path='data.csv')"
    ],
    category="data",
    danger_level="safe"
)
def analyze_csv(file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
    """深度分析CSV文件"""
    try:
        if not os.path.exists(file_path):
            return {"success": False, "error": f"文件不存在: {file_path}"}

        numeric_columns = []
        categorical_columns = []

        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            sample = f.read(4096)
            f.seek(0)
            delimiter = detect_delimiter(sample)

            reader = csv.DictReader(f, delimiter=delimiter)
            headers = reader.fieldnames or []

            column_data = {h: [] for h in headers}

            for row in reader:
                for header in headers:
                    value = row.get(header, "")
                    column_data[header].append(value)

        analysis = {}
        for header, data in column_data.items():
            clean_data = [v for v in data if v.strip()]
            unique_count = len(set(clean_data))

            is_numeric = True
            numeric_values = []

            for v in clean_data:
                try:
                    numeric_values.append(float(v))
                except ValueError:
                    is_numeric = False
                    break

            col_info = {
                "total_count": len(data),
                "non_null_count": len(clean_data),
                "null_count": len(data) - len(clean_data),
                "unique_count": unique_count,
                "is_numeric": is_numeric,
                "sample_values": clean_data[:5]
            }

            if is_numeric and numeric_values:
                col_info.update({
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "sum": sum(numeric_values),
                    "avg": sum(numeric_values) / len(numeric_values)
                })
                numeric_columns.append(header)
            else:
                value_counts = {}
                for v in clean_data:
                    value_counts[v] = value_counts.get(v, 0) + 1
                col_info["top_values"] = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                categorical_columns.append(header)

            analysis[header] = col_info

        return {
            "success": True,
            "file": os.path.basename(file_path),
            "total_rows": len(column_data[headers[0]]) if headers else 0,
            "total_columns": len(headers),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "analysis": analysis,
            "summary": {
                "numeric_count": len(numeric_columns),
                "categorical_count": len(categorical_columns),
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns
            }
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="read_excel",
    description="读取Excel文件（.xlsx/.xls），支持多sheet、行列预览、基础统计",
    parameters={
        "file_path": {"type": "str", "description": "Excel文件路径"},
        "sheet_name": {"type": "str", "description": "Sheet名称，默认第一个sheet"},
        "max_rows": {"type": "int", "description": "最大读取行数，默认50行"}
    },
    examples=[
        "read_excel(file_path='data.xlsx')",
        "read_excel(file_path='data.xlsx', sheet_name='Sheet2', max_rows=20)"
    ],
    category="data",
    danger_level="safe"
)
def read_excel(file_path: str, sheet_name: str = None, max_rows: int = 50) -> Dict[str, Any]:
    """读取Excel文件"""
    try:
        import pandas as pd

        if not os.path.exists(file_path):
            return {"success": False, "error": f"文件不存在: {file_path}"}

        file_size = os.path.getsize(file_path)

        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names

        if sheet_name is None:
            sheet_name = sheet_names[0]
        elif sheet_name not in sheet_names:
            return {"success": False, "error": f"Sheet不存在: {sheet_name}"}

        df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=max_rows)

        total_rows = len(pd.read_excel(file_path, sheet_name=sheet_name))
        is_preview = total_rows > max_rows

        column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        preview_data = df.head(10).to_dict('records')

        stats = {}
        for col in df.columns:
            col_data = df[col].dropna()
            stats[col] = {
                "dtype": str(df[col].dtype),
                "count": len(col_data),
                "null_count": df[col].isna().sum(),
                "unique_count": df[col].nunique()
            }
            if col in numeric_cols:
                stats[col].update({
                    "min": float(df[col].min()) if not df[col].isna().all() else None,
                    "max": float(df[col].max()) if not df[col].isna().all() else None,
                    "mean": float(df[col].mean()) if not df[col].isna().all() else None
                })

        return {
            "success": True,
            "file_info": {
                "path": file_path,
                "name": os.path.basename(file_path),
                "size": f"{file_size / 1024:.2f} KB",
                "total_sheets": len(sheet_names),
                "sheet_names": sheet_names,
                "current_sheet": sheet_name,
                "total_rows": total_rows,
                "total_columns": len(df.columns),
                "is_preview": is_preview
            },
            "columns": list(df.columns),
            "column_types": column_types,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "preview": preview_data,
            "preview_rows": len(preview_data),
            "column_stats": stats,
            "message": f"成功读取Sheet '{sheet_name}' {len(preview_data)} 行数据（ 共 {total_rows} 行）" if is_preview else f"成功读取Sheet '{sheet_name}' 全部 {total_rows} 行数据"
        }

    except ImportError:
        return {"success": False, "error": "需要安装 openpyxl: pip install openpyxl"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="export_to_csv",
    description="将数据导出为CSV文件",
    parameters={
        "data": {"type": "list", "description": "要导出的数据（列表格式）"},
        "file_path": {"type": "str", "description": "输出CSV文件路径"},
        "columns": {"type": "list", "description": "列名列表"}
    },
    examples=[
        "export_to_csv(data=[{'name': 'Tom', 'age': 20}], file_path='output.csv', columns=['name', 'age'])"
    ],
    category="data",
    danger_level="medium"
)
def export_to_csv(data: List[Dict], file_path: str = "export.csv", columns: List[str] = None) -> Dict[str, Any]:
    """导出数据到CSV（不写入文件，仅返回数据，由API层处理下载）"""
    try:
        if not data:
            return {"success": False, "error": "数据为空"}

        if columns is None:
            columns = list(data[0].keys()) if data else []

        # 仅返回数据，不写入文件
        # 文件写入由后端 execute API 处理（触发浏览器下载）
        return {
            "success": True,
            "message": f"准备导出 {len(data)} 行数据到 CSV 文件",
            "data": data,
            "columns": columns,
            "rows": len(data)
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
