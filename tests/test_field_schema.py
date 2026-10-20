from rag.structure.field_classifier import FieldSchema

def test_empty_schema_returns_syntax_types_only():
    s = FieldSchema.empty()
    assert s.resolve("团队", "GZHU-3") == ("code", "value")
    assert s.resolve("姓名", "张三") == ("free_text", "default")
    assert s.resolve("分数", "195") == ("number", "value")

def test_schema_header_hint_wins():
    s = FieldSchema.from_dict({"types": {"team": {"header_hints": ["团队"], "is_anchor": True}}})
    assert s.resolve("团队", "GZHU-3") == ("team", "header")
    assert s.resolve("队伍", "GZHU-3") == ("code", "value")   # 未命中 hint

def test_schema_pattern_wins_over_syntax():
    s = FieldSchema.from_dict({"types": {"team": {"value_patterns": [r"[A-Z]{2,}-\d+"]}}})
    assert s.resolve("", "GZHU-3") == ("team", "value")
    assert s.resolve("", "T01") == ("code", "value")          # 不匹配 pattern，退 syntax