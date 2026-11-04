from rag.structure.field_classifier import FieldSchema

def test_english_name_not_misclassified():
    """默认 schema 不把 2-4 汉字硬判成人名，英文名也不猜成人名。

    _CODE_RE 会把纯字母短串判成 code（语法类别），这是设计行为。
    关键是：不判成 person（领域类型）。
    """
    s = FieldSchema.empty()
    t, _ = s.resolve("name", "Alice")
    assert t != "person"
    assert t in ("code", "free_text")

def test_team_number_not_hardcoded():
    """T01 不被默认判为 team。"""
    s = FieldSchema.empty()
    assert s.resolve("团队", "T01")[0] == "code"