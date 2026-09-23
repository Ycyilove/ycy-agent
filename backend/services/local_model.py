"""
本地模型加载服务模块
支持从本地加载模型，避免在线下载
"""
import os
import json
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
from enum import Enum


class ModelSource(Enum):
    """模型来源"""
    ONLINE = "online"  # 在线API
    LOCAL = "local"   # 本地加载


@dataclass
class LocalModelConfig:
    """本地模型配置"""
    model_name: str
    model_path: Optional[str] = None
    device: str = "cpu"
    max_tokens: int = 2048
    temperature: float = 0.7
    use_modelscope: bool = True  # 是否使用ModelScope加载


class LocalModelService:
    """本地模型服务类"""
    
    def __init__(self, config: LocalModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
    
    def load_model(self) -> bool:
        """加载本地模型"""
        try:
            if self.config.use_modelscope:
                return self._load_modelscope_model()
            else:
                return self._load_huggingface_model()
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            return False
    
    def _load_modelscope_model(self) -> bool:
        """使用ModelScope加载模型"""
        try:
            from modelscope import AutoModelForCausalLM, AutoTokenizer
            
            print(f"正在从ModelScope加载模型: {self.config.model_name}")
            
            if self.config.model_path and os.path.exists(self.config.model_path):
                # 从本地路径加载
                print(f"从本地路径加载: {self.config.model_path}")
                model_dir = self.config.model_path
            else:
                # 从ModelScope下载到缓存
                model_dir = self.config.model_name
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                device_map=self.config.device,
                trust_remote_code=True
            )
            
            self._is_loaded = True
            print("模型加载成功！")
            return True
            
        except ImportError:
            print("请先安装ModelScope: pip install modelscope torch transformers")
            return False
        except Exception as e:
            print(f"ModelScope加载模型失败: {str(e)}")
            return False
    
    def _load_huggingface_model(self) -> bool:
        """使用HuggingFace加载模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"正在从HuggingFace加载模型: {self.config.model_name}")
            
            if self.config.model_path and os.path.exists(self.config.model_path):
                model_dir = self.config.model_path
            else:
                model_dir = self.config.model_name
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                device_map=self.config.device,
                trust_remote_code=True
            )
            
            self._is_loaded = True
            print("模型加载成功！")
            return True
            
        except ImportError:
            print("请先安装Transformers: pip install torch transformers")
            return False
        except Exception as e:
            print(f"HuggingFace加载模型失败: {str(e)}")
            return False
    
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """生成回复"""
        if not self._is_loaded or not self.model or not self.tokenizer:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        try:
            # 构建prompt
            prompt = self._build_prompt(messages)
            
            # 生成回复
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return response
            
        except Exception as e:
            print(f"生成失败: {str(e)}")
            raise
    
    def stream_generate(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """流式生成回复"""
        if not self._is_loaded or not self.model or not self.tokenizer:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        try:
            prompt = self._build_prompt(messages)
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # 注意：这是简化版本，实际的流式生成需要更复杂的实现
            # 这里先使用同步生成，然后逐字输出
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # 模拟流式输出
            for char in response:
                yield f"data: {json.dumps({'text': char}, ensure_ascii=False)}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            print(f"流式生成失败: {str(e)}")
            yield f"data: {json.dumps({'text': f'生成失败: {str(e)}', 'error': True}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
    
    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """构建prompt"""
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"系统: {content}")
            elif role == "user":
                prompt_parts.append(f"用户: {content}")
            elif role == "assistant":
                prompt_parts.append(f"助手: {content}")
        
        prompt_parts.append("助手: ")
        
        return "\n\n".join(prompt_parts)
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


# 全局单例
_local_model_service: Optional[LocalModelService] = None


def get_local_model_service(config: Optional[LocalModelConfig] = None) -> Optional[LocalModelService]:
    """获取本地模型服务实例"""
    global _local_model_service
    
    if _local_model_service is None and config is not None:
        _local_model_service = LocalModelService(config)
        _local_model_service.load_model()
    
    return _local_model_service


def download_model_from_modelscope(model_name: str, local_dir: Optional[str] = None) -> bool:
    """从ModelScope下载模型到本地"""
    try:
        from modelscope import snapshot_download
        
        print(f"正在下载模型: {model_name}")
        
        if local_dir is None:
            local_dir = os.path.join(os.path.dirname(__file__), "models", model_name.replace("/", "_"))
        
        os.makedirs(local_dir, exist_ok=True)
        
        print(f"下载到: {local_dir}")
        
        snapshot_download(
            model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"模型下载完成: {local_dir}")
        return True
        
    except ImportError:
        print("请先安装ModelScope: pip install modelscope")
        return False
    except Exception as e:
        print(f"下载模型失败: {str(e)}")
        return False
