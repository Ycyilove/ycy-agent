"""
LangChain 后端服务模块
集成LLM调用封装、RAG知识库、Agent工具调用、Chains工作流
支持 ModelScope API (兼容OpenAI格式) 和本地模型加载
"""
import os
# 配置 HuggingFace 国内镜像源（必须在任何导入之前设置）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HUB_CACHE'] = os.path.join(os.path.dirname(__file__), 'models')
import base64
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import anthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from session_memory import SessionMemory, get_session_memory

import tools.loader
from tools.agent import ToolAgent, IntentAnalysis, ToolCallRequest, ToolCallResult

from rag import DocumentParserFactory, TextChunker, FAISSVectorStore

from local_model_service import (
    LocalModelConfig, 
    LocalModelService, 
    get_local_model_service,
    download_model_from_modelscope,
    ModelSource
)


app = FastAPI(title="LLM API 服务")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODESCOPE_API_KEY = os.getenv("MODESCOPE_API_KEY", "ms-96302da9-c430-4be7-b972-91beb2966f43")
MODESCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1"

# 模型来源配置
MODEL_SOURCE = ModelSource.ONLINE  # 可选: ONLINE (在线API) 或 LOCAL (本地模型)

# 本地模型配置
LOCAL_MODEL_CONFIG = LocalModelConfig(
    model_name="deepseek-ai/DeepSeek-R1",
    model_path=os.getenv("LOCAL_MODEL_PATH", None),  # 本地模型路径，留空则从ModelScope下载
    device=os.getenv("DEVICE", "cpu"),  # 可选: cpu, cuda, auto
    max_tokens=2048,
    temperature=0.7,
    use_modelscope=True
)


tool_agent: Optional[ToolAgent] = None


def get_tool_agent() -> ToolAgent:
    """获取工具Agent实例（延迟初始化）"""
    global tool_agent
    if tool_agent is None:
        tools.loader.load_all_tools()
        tool_agent = ToolAgent()
    return tool_agent


def encode_file_to_base64(file_path: str) -> str:
    """将本地文件编码为base64"""
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def get_media_type(file_path: str) -> str:
    """根据文件扩展名获取media_type"""
    ext = os.path.splitext(file_path)[1].lower()
    media_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.pdf': 'application/pdf',
    }
    return media_types.get(ext, 'application/octet-stream')


class ModelScopeLLM:
    """ModelScope API LLM 封装类，支持在线API和本地模型"""

    QUICK_RESPONSE_PROMPT = """你是一个高效的AI助手，请直接、简洁地回答用户问题，不要过度思考，快速给出答案。"""
    DEEP_THINKING_PROMPT = """你是一个善于深度思考的AI助手，请仔细分析问题，一步步推理，先给出思考过程，再给出最终答案。请用"思考："开头来展示你的思考过程，然后用"答案："来给出最终结论。"""

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.7, max_tokens: int = 2048):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_local = MODEL_SOURCE == ModelSource.LOCAL
        
        if self.use_local:
            # 使用本地模型
            self.local_service = get_local_model_service(LOCAL_MODEL_CONFIG)
        else:
            # 使用在线API
            self.llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=MODESCOPE_API_KEY,
                base_url=MODESCOPE_BASE_URL,
            )

    def generate(self, messages: List[Dict[str, str]], mode: str = "quick") -> str:
        """同步生成回复"""
        processed_messages = self._add_mode_prompt(messages, mode)
        
        if self.use_local:
            if not self.local_service or not self.local_service.is_loaded:
                raise RuntimeError("本地模型未加载，请确保MODEL_SOURCE设置正确")
            return self.local_service.generate(processed_messages)
        else:
            langchain_messages = self._convert_to_langchain(processed_messages)
            return self.llm.invoke(langchain_messages).content

    def stream_generate(self, messages: List[Dict[str, str]], mode: str = "quick"):
        """流式生成回复"""
        processed_messages = self._add_mode_prompt(messages, mode)
        
        if self.use_local:
            if not self.local_service or not self.local_service.is_loaded:
                yield f"data: {json.dumps({'text': '本地模型未加载', 'error': True}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            yield from self.local_service.stream_generate(processed_messages)
        else:
            langchain_messages = self._convert_to_langchain(processed_messages)
            try:
                for chunk in self.llm.stream(langchain_messages):
                    if chunk.content:
                        yield f"data: {json.dumps({'text': chunk.content}, ensure_ascii=False)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'text': f'生成响应时发生错误: {str(e)}', 'error': True}, ensure_ascii=False)}\n\n"
            finally:
                yield "data: [DONE]\n\n"

    def _add_mode_prompt(self, messages: List[Dict[str, str]], mode: str = "quick") -> List[Dict[str, str]]:
        """添加模式提示词到消息列表"""
        system_prompt = self.QUICK_RESPONSE_PROMPT if mode == "quick" else self.DEEP_THINKING_PROMPT
        result = [{"role": "system", "content": system_prompt}]
        result.extend(messages)
        return result

    def _convert_to_langchain(self, messages: List[Dict[str, str]]):
        """将消息格式转换为LangChain格式"""
        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(HumanMessage(content=content))
            else:
                langchain_messages.append(HumanMessage(content=content))

        return langchain_messages


class MultimodalClient:
    """多模态客户端（支持图片/PDF分析）"""

    def __init__(self):
        self.client = anthropic.Anthropic(
            base_url='https://api-inference.modelscope.cn',
            api_key=MODESCOPE_API_KEY,
        )

    def stream_analyze(self, messages: List[Dict], max_tokens: int = 4096):
        """流式分析多模态内容，支持思考链"""
        thinking_content = ""
        text_content = ""
        
        with self.client.messages.stream(
            model='deepseek-ai/DeepSeek-R1',
            messages=messages,
            max_tokens=max_tokens,
            extra_headers={
                "anthropic-beta": "thinking-2025-01-21"
            }
        ) as stream:
            for event in stream:
                if hasattr(event, 'type') and event.type == 'content_block_delta':
                    if hasattr(event, 'delta'):
                        delta = event.delta
                        if hasattr(delta, 'type'):
                            if delta.type == 'thinking_delta':
                                thinking_content += delta.thinking
                                yield f"data: {json.dumps({'thinking': delta.thinking, 'type': 'thinking'})}\n\n"
                            elif delta.type == 'text_delta':
                                text_content += delta.text
                                yield f"data: {json.dumps({'text': delta.text, 'type': 'text'})}\n\n"
                        elif hasattr(delta, 'text'):
                            text_content += delta.text
                            yield f"data: {json.dumps({'text': delta.text, 'type': 'text'})}\n\n"
                elif hasattr(event, 'text'):
                    text_content += event.text
                    yield f"data: {json.dumps({'text': event.text, 'type': 'text'})}\n\n"

    def analyze(self, messages: List[Dict], max_tokens: int = 2048) -> str:
        """同步分析多模态内容"""
        message = self.client.messages.create(
            model='deepseek-ai/DeepSeek-R1',
            messages=messages,
            max_tokens=max_tokens
        )
        return message.content[0].text


class RAGService:
    """RAG 知识库服务"""

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def add_texts(self, texts: List[str]):
        """添加文本到知识库"""
        if not self.vector_store:
            self.vector_store = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings
            )
        else:
            self.vector_store.add_texts(texts)
        return f"已添加 {len(texts)} 条文本"

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """检索相关文档"""
        if not self.vector_store:
            return ["知识库未初始化，请先加载文档"]

        docs = self.vector_store.similarity_search(query, k=top_k)
        return [doc.page_content for doc in docs]

    def rag_answer(self, query: str, llm: ModelScopeLLM = None) -> str:
        """RAG问答"""
        if not llm:
            llm = ModelScopeLLM()

        relevant_docs = self.retrieve(query, top_k=3)
        context = "\n\n".join(relevant_docs)

        prompt = f"""基于以下参考信息回答用户问题。如果参考信息不足以回答，请说明无法回答。

参考信息:
{context}

用户问题: {query}

回答:"""

        result = llm.generate([
            {"role": "user", "content": prompt}
        ])
        return result


llm_instance = None
multimodal_client = None
rag_service = None


def get_llm():
    """延迟获取LLM实例"""
    global llm_instance
    if llm_instance is None:
        llm_instance = ModelScopeLLM()
    return llm_instance


def get_multimodal_client():
    """延迟获取多模态客户端"""
    global multimodal_client
    if multimodal_client is None:
        multimodal_client = MultimodalClient()
    return multimodal_client


def get_rag_service():
    """延迟获取RAG服务，避免启动时下载模型"""
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service


@app.get("/")
async def root():
    """健康检查"""
    return {"status": "ok", "message": "LLM API 服务运行中"}


@app.get("/api/tools")
async def list_tools():
    """获取所有可用工具列表"""
    try:
        agent = get_tool_agent()
        tools_desc = agent.registry.get_tool_descriptions()
        return {
            "status": "success",
            "tools": tools_desc,
            "count": len(tools_desc)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tools/{tool_name}")
async def get_tool_info(tool_name: str):
    """获取特定工具的详细信息"""
    try:
        agent = get_tool_agent()
        metadata = agent.registry.get_metadata(tool_name)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"工具不存在: {tool_name}")
        
        return {
            "status": "success",
            "tool": {
                "name": metadata.name,
                "description": metadata.description,
                "parameters": metadata.parameters,
                "category": metadata.category,
                "danger_level": metadata.danger_level,
                "examples": metadata.examples
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tools/analyze")
async def analyze_intent(message: str = Form(...)):
    """分析用户消息，识别是否需要调用工具"""
    try:
        agent = get_tool_agent()
        intent = agent.analyze_intent(message)

        result = {
            "needs_tool": intent.needs_tool,
            "intent_type": intent.intent_type,
            "confidence": intent.confidence,
            "reasoning": intent.reasoning
        }

        if intent.needs_tool and intent.suggested_tool:
            tool_call = agent.prepare_tool_call(intent)
            if tool_call is None:
                raise ValueError(f"prepare_tool_call returned None for tool: {intent.suggested_tool}")
            if not hasattr(tool_call, 'tool_name'):
                raise ValueError(f"tool_call is not a ToolCallRequest, got: {type(tool_call)} - {tool_call}")
            result["tool"] = {
                "name": tool_call.tool_name,
                "parameters": tool_call.parameters,
                "needs_confirmation": tool_call.user_confirmation_needed
            }

        return {"status": "success", "analysis": result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tools/execute")
async def execute_tool(
    tool_name: str = Form(...),
    parameters: str = Form("{}"),
    confirmed: bool = Form(False)
):
    """执行工具调用，导出类工具自动触发浏览器下载"""
    from fastapi.responses import Response
    import io
    import csv

    try:
        agent = get_tool_agent()

        metadata = agent.registry.get_metadata(tool_name)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"工具不存在: {tool_name}")

        if metadata.danger_level in ["medium", "high"] and not confirmed:
            return {
                "status": "confirmation_needed",
                "message": agent.generate_confirmation_message(
                    ToolCallRequest(
                        tool_name=tool_name,
                        parameters=json.loads(parameters),
                        confidence=1.0,
                        reasoning="",
                        user_confirmation_needed=True
                    )
                ),
                "danger_level": metadata.danger_level
            }

        params = json.loads(parameters)

        # 导出类工具 - 直接返回文件下载响应
        if tool_name == "export_to_csv":
            data = params.get("data", [])
            columns = params.get("columns")

            if not data:
                raise HTTPException(status_code=400, detail="数据为空")

            if columns is None:
                columns = list(data[0].keys()) if data else []

            # 生成 CSV 内容
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)
            csv_content = output.getvalue()

            # 返回下载响应
            filename = params.get("file_path", "export.csv")
            if not filename.endswith('.csv'):
                filename += '.csv'

            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "Content-Type": "text/csv; charset=utf-8"
                }
            )

        # 执行工具
        tool_call = ToolCallRequest(
            tool_name=tool_name,
            parameters=params,
            confidence=1.0,
            reasoning="",
            user_confirmation_needed=False
        )

        result = agent.execute_tool(tool_call)
        formatted_result = agent.format_tool_result(result)

        return {
            "status": "success",
            "result": formatted_result,
            "raw_result": result.result if isinstance(result.result, dict) else {"output": str(result.result)},
            "success": result.success,
            "execution_time": result.execution_time
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(message: str = Form(...), history: str = Form("[]")):
    """文本对话接口"""
    try:
        history_list = json.loads(history)
        messages = history_list + [{"role": "user", "content": message}]

        return StreamingResponse(
            get_llm().stream_generate(messages),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(messages: List[Dict[str, Any]]):
    """流式对话接口（直接接收消息数组）"""
    try:
        return StreamingResponse(
            get_llm().stream_generate(messages),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
async def analyze_image(
    message: str = Form(...),
    files: List[UploadFile] = File(None)
):
    """多模态分析接口"""
    try:
        content_list = []

        if files:
            for file in files:
                contents = await file.read()
                file_base64 = base64.b64encode(contents).decode("utf-8")

                filename = file.filename.lower()
                if filename.endswith('.pdf'):
                    media_type = "application/pdf"
                    content_type = "file"
                else:
                    media_type = f"image/{filename.split('.')[-1]}"
                    if media_type == "image/jpg":
                        media_type = "image/jpeg"
                    content_type = "image"

                content_list.append({
                    "type": content_type,
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": file_base64
                    }
                })

        content_list.append({
            "type": "text",
            "text": message
        })

        messages = [{"role": "user", "content": content_list}]

        return StreamingResponse(
            get_multimodal_client().stream_analyze(messages),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/add")
async def rag_add_texts(texts: List[str] = Form(...)):
    """添加文本到知识库"""
    try:
        service = get_rag_service()
        result = service.add_texts(texts)
        return {"status": "success", "message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/query")
async def rag_query(query: str = Form(...)):
    """RAG问答"""
    try:
        service = get_rag_service()
        result = service.rag_answer(query, get_llm())
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class LocalRAGService:
    """本地RAG知识库服务"""

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        self.vector_store = FAISSVectorStore(embedding_model=embedding_model)
        self.chunker = TextChunker(chunk_size=500, overlap=100)

    def add_file(self, file_content: bytes, filename: str) -> Dict:
        """添加文件到知识库"""
        try:
            if self.vector_store.file_exists(filename):
                return {"status": "warning", "message": f"文件 {filename} 已存在，跳过添加"}

            text = DocumentParserFactory.parse_file(file_content, filename)
            if not text or not text.strip():
                return {"status": "error", "message": "文件内容为空"}

            chunks = self.chunker.chunk_text(text)
            if not chunks:
                return {"status": "error", "message": "文本分块失败"}

            filenames = [filename] * len(chunks)
            result = self.vector_store.add_documents(chunks, filenames)

            return result
        except ValueError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": f"处理文件时出错: {str(e)}"}

    def add_files(self, files: List[tuple]) -> Dict:
        """批量添加文件"""
        results = []
        success_count = 0
        error_count = 0

        for file_content, filename in files:
            result = self.add_file(file_content, filename)
            results.append({"filename": filename, **result})
            if result["status"] == "success":
                success_count += 1
            else:
                error_count += 1

        return {
            "status": "success",
            "message": f"成功添加 {success_count} 个文件，失败 {error_count} 个",
            "details": results
        }

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索相关文档"""
        return self.vector_store.search(query, top_k)

    def rag_answer(self, query: str, top_k: int = 3) -> Dict:
        """RAG问答"""
        retrieved_docs = self.search(query, top_k)

        if not retrieved_docs:
            return {
                "answer": "知识库为空，请先上传文档",
                "retrieved_docs": [],
                "context": ""
            }

        context_parts = []
        for idx, doc in enumerate(retrieved_docs, 1):
            source = doc.get('filename', '未知来源')
            content = doc.get('text', '')
            score = doc.get('score', 0)
            context_parts.append(f"【文档 {idx}】来源: {source}\n{content}")

        context = "\n\n".join(context_parts)

        prompt = f"""基于以下参考信息回答用户问题。如果参考信息不足以回答，请基于你的知识回答，并说明情况。

参考信息:
{context}

用户问题: {query}

回答:"""

        llm = ModelScopeLLM()
        answer = llm.generate([{"role": "user", "content": prompt}])

        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "context": context
        }

    def stream_rag_answer(self, query: str, top_k: int = 3):
        """流式RAG问答"""
        retrieved_docs = self.search(query, top_k)

        if not retrieved_docs:
            yield f"data: {json.dumps({'text': '知识库为空，请先上传文档'})}\n\n"
            return

        context_parts = []
        for idx, doc in enumerate(retrieved_docs, 1):
            source = doc.get('filename', '未知来源')
            content = doc.get('text', '')
            context_parts.append(f"【文档 {idx}】来源: {source}\n{content}")

        context = "\n\n".join(context_parts)

        prompt = f"""基于以下参考信息回答用户问题。如果参考信息不足以回答，请基于你的知识回答，并说明情况。

参考信息:
{context}

用户问题: {query}

回答:"""

        llm = ModelScopeLLM()
        for chunk in llm.stream_generate([{"role": "user", "content": prompt}]):
            yield chunk

    def delete_file(self, filename: str) -> Dict:
        """删除指定文件的向量索引"""
        return self.vector_store.delete_file(filename)

    def clear_all(self) -> Dict:
        """清空全部知识库"""
        return self.vector_store.clear_all()

    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        return self.vector_store.get_stats()


local_rag_service = None


def get_local_rag_service():
    """获取本地RAG服务实例"""
    global local_rag_service
    if local_rag_service is None:
        local_rag_service = LocalRAGService()
    return local_rag_service


@app.post("/api/chat/rag")
async def chat_with_rag(
    message: str = Form(...),
    history: str = Form("[]"),
    use_rag: bool = Form(False),
    top_k: int = Form(3),
    mode: str = Form("quick")
):
    """
    结合知识库检索的对话接口
    - message: 用户消息
    - history: 对话历史JSON字符串
    - use_rag: 是否使用RAG模式
    - top_k: 检索文档数量，默认3
    - mode: 回答模式，"quick" 为快速回答，"deep" 为深度思考
    """
    try:
        history_list = json.loads(history)
        
        messages = history_list + [{"role": "user", "content": message}]
        
        if use_rag:
            local_rag = get_local_rag_service()
            retrieved_docs = local_rag.search(message, top_k)
            
            if retrieved_docs:
                context_parts = []
                for idx, doc in enumerate(retrieved_docs, 1):
                    source = doc.get('filename', '未知来源')
                    content = doc.get('text', '')
                    context_parts.append(f"【文档 {idx}】来源: {source}\n{content}")
                
                context = "\n\n".join(context_parts)
                
                system_prompt = f"""你是一个智能助手。请基于以下参考信息回答用户问题。

参考信息:
{context}

请根据以上参考信息来回答用户的问题。如果参考信息不足以回答，你可以基于自己的知识来回答。"""
                
                messages = [{"role": "system", "content": system_prompt}] + messages
        
        return StreamingResponse(
            get_llm().stream_generate(messages, mode=mode),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/add_file")
async def rag_add_file(file: UploadFile = File(...)):
    """添加文件到知识库"""
    try:
        file_content = await file.read()
        result = get_local_rag_service().add_file(file_content, file.filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/add_files")
async def rag_add_files(files: List[UploadFile] = File(...)):
    """批量添加文件到知识库"""
    try:
        file_list = []
        for file in files:
            file_content = await file.read()
            file_list.append((file_content, file.filename))

        result = get_local_rag_service().add_files(file_list)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rag/stats")
async def rag_stats():
    """获取知识库统计信息"""
    try:
        stats = get_local_rag_service().get_stats()
        return {"status": "success", **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/search")
async def rag_search(query: str = Form(...), top_k: int = Form(3)):
    """检索知识库"""
    try:
        results = get_local_rag_service().search(query, top_k)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/query/stream")
async def rag_query_stream(query: str = Form(...), top_k: int = Form(3)):
    """流式RAG问答"""
    try:
        return StreamingResponse(
            get_local_rag_service().stream_rag_answer(query, top_k),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/rag/file")
async def rag_delete_file(filename: str = Form(...)):
    """删除指定文件的向量索引"""
    try:
        result = get_local_rag_service().delete_file(filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/rag/clear")
async def rag_clear_all():
    """清空全部知识库"""
    try:
        result = get_local_rag_service().clear_all()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/create")
async def create_session(name: str = Form("")):
    """创建新会话"""
    try:
        session_id = get_session_memory().create_session(name)
        return {"status": "success", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/list")
async def list_sessions():
    """列出所有会话"""
    try:
        sessions = get_session_memory().list_sessions()
        return {"status": "success", "sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """获取会话信息"""
    try:
        info = get_session_memory().get_session_info(session_id)
        if info is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"status": "success", "session": info}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    try:
        success = get_session_memory().delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"status": "success", "message": "会话已删除"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/session/{session_id}/rename")
async def rename_session(session_id: str, name: str = Form(...)):
    """重命名会话"""
    try:
        success = get_session_memory().rename_session(session_id, name)
        if not success:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"status": "success", "message": "会话已重命名"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}/messages")
async def get_session_messages(session_id: str):
    """获取会话消息"""
    try:
        messages = get_session_memory().get_messages(session_id)
        return {"status": "success", "messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}/context")
async def get_session_context(session_id: str, max_messages: int = 20):
    """获取会话上下文（用于LLM对话）"""
    try:
        messages = get_session_memory().get_conversation_context(session_id, max_messages)
        return {"status": "success", "messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/{session_id}/message")
async def add_session_message(session_id: str, role: str = Form(...), content: str = Form(...)):
    """添加消息到会话"""
    try:
        msg_id = get_session_memory().add_message(session_id, role, content)
        return {"status": "success", "message_id": msg_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}/files")
async def get_session_files(session_id: str):
    """获取会话关联的文件"""
    try:
        files = get_session_memory().get_session_files(session_id)
        return {"status": "success", "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/{session_id}/file")
async def add_session_file(
    session_id: str,
    file_name: str = Form(...),
    file_path: str = Form(None),
    file_type: str = Form(None)
):
    """添加文件到会话"""
    try:
        file_id = get_session_memory().add_session_file(session_id, file_name, file_path, file_type)
        return {"status": "success", "file_id": file_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/session/file/{file_id}")
async def delete_session_file(file_id: str):
    """删除会话文件"""
    try:
        success = get_session_memory().delete_session_file(file_id)
        return {"status": "success", "deleted": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/search")
async def search_session_messages(session_id: str = Form(...), keyword: str = Form(...), limit: int = Form(10)):
    """搜索会话消息"""
    try:
        results = get_session_memory().search_messages(session_id, keyword, limit)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/info")
async def get_model_info():
    """获取当前模型配置信息"""
    try:
        return {
            "status": "success",
            "model_source": MODEL_SOURCE.value,
            "model_name": DEFAULT_MODEL,
            "display_name": DEFAULT_MODEL.split("/")[-1] if "/" in DEFAULT_MODEL else DEFAULT_MODEL,
            "use_local": MODEL_SOURCE == ModelSource.LOCAL
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/model/download")
async def download_model(model_name: str = Form(...), local_dir: str = Form(None)):
    """从ModelScope下载模型到本地"""
    try:
        success = download_model_from_modelscope(model_name, local_dir)
        if success:
            return {"status": "success", "message": "模型下载成功"}
        else:
            raise HTTPException(status_code=500, detail="模型下载失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/model/switch")
async def switch_model_source(source: str = Form(...)):
    """切换模型来源（在线/本地）"""
    global MODEL_SOURCE
    
    try:
        if source == "online":
            MODEL_SOURCE = ModelSource.ONLINE
            return {"status": "success", "message": "已切换到在线API模式"}
        elif source == "local":
            MODEL_SOURCE = ModelSource.LOCAL
            # 预加载本地模型
            service = get_local_model_service(LOCAL_MODEL_CONFIG)
            if service and service.is_loaded:
                return {"status": "success", "message": "已切换到本地模型模式，模型加载成功"}
            else:
                return {"status": "warning", "message": "已切换到本地模型模式，但模型加载失败，请检查配置"}
        else:
            raise HTTPException(status_code=400, detail="无效的来源类型，可选: online, local")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tools/download")
async def download_file(
    tool_name: str = Form(...),
    parameters: str = Form(...),
    confirmed: bool = Form(False)
):
    """
    执行工具并返回文件下载响应（触发浏览器下载）

    Args:
        tool_name: 工具名称
        parameters: JSON 格式的参数
        confirmed: 是否已确认危险操作
    """
    from fastapi.responses import Response
    import io
    import csv

    try:
        agent = get_tool_agent()

        metadata = agent.registry.get_metadata(tool_name)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"工具不存在: {tool_name}")

        params = json.loads(parameters)

        # 根据工具类型处理不同的导出逻辑
        if tool_name == "export_to_csv":
            data = params.get("data", [])
            columns = params.get("columns")

            if not data:
                raise HTTPException(status_code=400, detail="数据为空")

            if columns is None:
                columns = list(data[0].keys()) if data else []

            # 生成 CSV 内容
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)
            csv_content = output.getvalue()

            # 返回下载响应
            filename = params.get("file_path", "export.csv")
            if not filename.endswith('.csv'):
                filename += '.csv'

            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "Content-Type": "text/csv; charset=utf-8"
                }
            )

        elif tool_name == "run_python_code":
            # 执行代码并返回结果
            tool_call = ToolCallRequest(
                tool_name=tool_name,
                parameters=params,
                confidence=1.0,
                reasoning="",
                user_confirmation_needed=False
            )
            result = agent.execute_tool(tool_call)

            if result.success:
                output_text = str(result.result)
            else:
                output_text = f"Error: {result.error}"

            return Response(
                content=output_text,
                media_type="text/plain",
                headers={
                    "Content-Disposition": "attachment; filename=output.txt",
                    "Content-Type": "text/plain; charset=utf-8"
                }
            )

        else:
            # 其他工具，常规执行
            tool_call = ToolCallRequest(
                tool_name=tool_name,
                parameters=params,
                confidence=1.0,
                reasoning="",
                user_confirmation_needed=False
            )
            result = agent.execute_tool(tool_call)

            if result.success:
                result_text = json.dumps(result.result, ensure_ascii=False, indent=2)
            else:
                result_text = f"Error: {result.error}"

            return Response(
                content=result_text,
                media_type="application/json",
                headers={
                    "Content-Disposition": f'attachment; filename={tool_name}_result.json',
                    "Content-Type": "application/json; charset=utf-8"
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
