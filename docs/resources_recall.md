# 测试提问集

## A. 知识库图片召回（PDF）

| # | 提问 | 预期 | 失败看 |
|---|------|------|--------|
| A1 | 知识库里有哪些图片？列出前 5 个资源的文件名和页码。 | `list_knowledge_resources(kind="image", limit=5)` 返回 `count>0`，每项有 `id/url/page_no` | `knowledge_stats.total_resources` 是否为 0；`[dsh][resource] ingest` 的 `page_nos` |
| A2 | 把知识库里第 1 页的图片展示出来。 | `search_knowledge(mode="flat")` 命中 chunk，`chunks[0].resources` 非空且 `page_no=1` | `enrich_search_results` 按页过滤是否全过滤掉（应已放宽 ±1） |
| A3 | 知识库里的图片分别来自哪些文档？ | `list_knowledge_resources` 每项有 `source_filename` | `ResourceRecord.source_filename` 是否正确写入 |

## B. 知识库表格 / 记录召回

| # | 提问 | 预期 | 失败看 |
|---|------|------|--------|
| B1 | GZHU-3 获得了什么奖项？ | `search_knowledge(mode="auto")` 命中"全国团队二等奖"段落，不再依赖 `search_content` | 向量检索维度是否不匹配（需清空 `vector_store/` 重建） |
| B2 | 杨灿垣在哪个队？拿了什么奖？ | `mode="record"` 或 auto 命中记录，`chunks[].text` 含"杨灿垣"和"三等奖" | `field_schema.json` 是否为空；`record_store.stats().total_records` |
| B3 | 知识库里有哪些表格？把第一个表格的截图给我。 | `mode="record"` 命中记录，`resources` 非空（走新的 `_record_search` 按 `source_filename` 查） | `_record_search` 返回的 `resources` 是否为空 |
| B4 | GZHU-3 的队员都有谁？ | 命中成员列表 chunk，`resources` 若该页有图则非空 | `*.chunks.jsonl` 里 `headings: ["GZHU-3：..."]` 是否被索引 |

## C. URL 资源获取

| # | 提问 | 预期 | 失败看 |
|---|------|------|--------|
| C1 | 下载 https://www.python.org/static/img/python-logo.png 并展示。 | `fetch_url` 返回 `mime=image/png`，`resources[0].url` 可打开 | `_is_allowed` 是否拒绝；MIME 是否被 CDN 报成 `text/html`（应已纠正） |
| C2 | 抓取 https://example.com 的首页内容。 | `fetch_url` 返回 `text` 字段，不产生 resource | HTML 走 text 分支，正确 |
| C3 | 下载 https://arxiv.org/pdf/2005.11401 并总结。 | `resources[0].kind="pdf"`，`/api/resources/<id>` 返回 inline PDF | `_MAX_BYTES` 20MB 是否触发；`save_bytes` 是否拒绝 `application/pdf`（不会） |
| C4 | 下载 https://cdn.example.com/photo.jpg（服务器返回 text/html） | MIME 被纠正为 `image/jpeg`，资源落盘成功 | `fetch.py` 5a 的扩展名纠正是否生效；`_is_allowed` 图片例外是否生效 |
| C5 | 下载 http://127.0.0.1:8000/api/rag/stats | `fetch_url` 拒绝，返回"禁止访问" | `_is_blocked_ip("127.0.0.1")` |
| C6 | 下载 http://169.254.169.254/latest/meta-data/ | 拒绝 | `ip.is_link_local` 分支 |

## D. 跨来源资源混合

| # | 提问 | 预期 | 失败看 |
|---|------|------|--------|
| D1 | 知识库和刚才下载的 PDF 里，分别有哪些图片？对比列出。 | 知识库侧 `resources` 非空；下载的 PDF 若未入库，`resources` 为空（`fetch_url` 只落整个文件，不提内嵌图） | 这是设计边界，不是 bug |
| D2 | 把 https://example.com/report.pdf 加到知识库，然后查里面的图表。 | 入库后 `knowledge_stats.total_resources>0` | `add_file` 里 `ingest` 返回 0 → 看 `[dsh][rag] pypdf 无资源，Docling 补提` 是否触发 |
| D3 | 知识库里 PDF 的图片和 DOCX 的图片分开列。 | 每项有 `source_filename`，可按扩展名分组 | `_extract_zip_resources` 对 DOCX 是否提图 |

## E. 资源与 chunk 关联

| # | 提问 | 预期 | 失败看 |
|---|------|------|--------|
| E1 | 搜索"杨灿垣"，把相关段落和图片一起给我。 | `chunks[0].text` 含"杨灿垣"，`chunks[0].resources` 若该页有图则非空 | `enrich_search_results` 的 `chunk_pages` 过滤 |
| E2 | 搜索"天梯赛"，列出命中的 chunk 和它们关联的资源。 | 每个 chunk 的 `resources` 来自 `_links[chunk_id]` 或 `resource_ids` 兜底 | `_links` 是否在 `add_file` 时正确 `link_chunks` |
| E3 | 搜索"广州大学"，返回的资源和 chunk 的页码对得上吗？ | `resources[].page_no` 和 `chunks[].page_numbers` 相差不超过 1 | 放宽后的 ±1 容差 |

## F. 空库 / 无资源文档

| # | 提问 | 预期 | 失败看 |
|---|------|------|--------|
| F1 | 知识库里有哪些图片？（空库） | `total_resources=0`，`list_knowledge_resources` 返回 `count=0`，不报错 | `list_all()` 在空 dict 上是否抛异常 |
| F2 | 上传一个纯文本 TXT，然后查它的图片。 | `add_file` 成功，`total_resources=0`，`search_knowledge` 命中 chunk 但 `resources=[]` | `extract_resources` 对 `.txt` 返回空（正确，不是报错） |
| F3 | 上传一个纯扫描件 PDF，然后查它的图片和文本。 | 走 OCR 路线，`last_document` 有 `pictures`，`resources` 非空 | `_detect_need_ocr` 是否返回 True；`_parse_with_ocr` 是否设置 `last_document` |

## G. 安全边界

| # | 提问 | 预期 | 失败看 |
|---|------|------|--------|
| G1 | 下载 http://localhost:8000/api/rag/stats | 拒绝 | `_BLOCKED_HOSTS` 含 `localhost` |
| G2 | 下载一个 .svg 文件并展示 | `save_bytes` 拒绝（`image/svg+xml` 在黑名单），或 inline 时被 CSP 拦 | `_is_allowed` 的 SVG 分支；`inline_ok` 排除 SVG |
| G3 | 下载一个 Content-Type 为 text/html 的 .png | MIME 纠正为 `image/png`，资源落盘成功 | 5a 扩展名纠正 |
| G4 | 访问 /api/rag/resource/<不存在的id> | 404 | 路由是否正确抛 `HTTPException(404)` |

## H. 资源 URL 可访问性（手动）

| # | 操作 | 预期 |
|---|------|------|
| H1 | 拿到图片资源的 `url`（`/api/rag/resource/<id>`）在浏览器打开 | 显示图片，`Content-Disposition: inline`，响应头有 `X-Content-Type-Options: nosniff` |
| H2 | 打开 PDF 资源 URL | 内联预览 |
| H3 | 打开表格资源（`kind=table`） | 返回 JSON（`columns` + `rows`），不是文件 |
| H4 | 用 `/api/resources/<id>` 打开 `fetch_url` 下载的资源 | 正常返回；`inline_ok` 时内联，否则附件 |
| H5 | 打开一个 SVG 资源的 URL | 走 `attachment` 下载，不内联 |

## I. 删除与清理

| # | 提问 | 预期 | 失败看 |
|---|------|------|--------|
| I1 | 删除知识库里的 xxx.pdf，然后确认它的图片也没了。 | `delete_file` → `delete_source` + `record_store.delete_by_source` + `bm25_store.reset()`；`rag_resources/` 下该文档目录被删 | `delete_source` 返回的 `deleted_count` 是否等于原 `total_resources` |
| I2 | 清空知识库后，再查图片。 | `total_resources=0`，`rag_resources/` 为空 | `clear_all` 是否调了 `resource_manager.clear()` |
| I3 | 24 小时后再访问一个旧资源 URL。 | 404（TTL 清理） | `cleanup_expired` 是否在 lifespan 里被调 |

## J. 回归：上一轮的 RAG 链路

| # | 提问 | 预期 | 失败看 |
|---|------|------|--------|
| J1 | GZHU-3 获得了什么奖项？ | `search_knowledge` 直接命中，不依赖 `search_content` | 向量检索维度不匹配 → 清空 `vector_store/` 重建 |
| J2 | 2025 年天梯赛杨灿垣拿了多少分？ | 命中个人三等奖段落，`resources` 若该页有图则非空 | `*.chunks.jsonl` 索引是否完整 |
| J3 | 知识库里有哪些文档？ | `list_knowledge_sources` 返回 `count>0`，`file_index` 已从 metadata 重建 | `vector_store.get_stats().total_files` 是否为 0（应为实际文件数） |

## K. 性能边界（观察，非功能测试）

| # | 操作 | 观察点 |
|---|------|--------|
| K1 | 批量上传 20 个 PDF | `add_file` 是否每次都重建 BM25（P2-1 修复后应只在首次检索时构建一次） |
| K2 | 知识库有 1000+ chunk 时检索 | BM25 惰性构建耗时；`enrich_search_results` 的 O(n) 遍历是否可接受 |
| K3 | `rag_resources/` 有 1000+ 子目录时启动 | `_load_index` 读 `index.jsonl` 的耗时（P2-2 暂不优化，观察即可） |