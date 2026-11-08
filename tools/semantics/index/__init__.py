"""工具语义索引的统一编码 + 存储 + 三种检索策略。

设计：
    - encoder.py：唯一编码逻辑，所有策略共享
    - store.py：FAISS + records.json 原子读写
    - strategies/：shortlist / intent_search / hybrid 三种检索
    - facade.py：对外统一 API

B6-a：encoder.py + store.py
B6-b：strategies/ + facade.py
B6-c：改 tool_shortlister / mcp_intent_index / capability.index 为薄壳
"""