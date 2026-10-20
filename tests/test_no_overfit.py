def test_english_name_not_misclassified():
    """默认 schema 不把 2-4 汉字硬判成人名，英文名也一样不猜。"""
    s = FieldSchema.empty()
    assert s.resolve("name", "Alice")[0] == "free_text"
    assert s.resolve("姓名", "张三")[0] == "free_text"

def test_team_number_not_hardcoded():
    """T01 不被默认判为 team。"""
    s = FieldSchema.empty()
    assert s.resolve("团队", "T01")[0] == "code"