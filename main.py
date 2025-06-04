# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext
)
from llama_index.llms import CustomLLM
from llama_index.llms.types import CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.embeddings import BaseEmbedding
from typing import Optional, List, Any, Generator
import dashscope
from dashscope import Generation, TextEmbedding
from pydantic import Field
import numpy as np
import json

# 加载环境变量
load_dotenv()

class QwenEmbedding(BaseEmbedding):
    """千问嵌入模型包装器"""
    
    def __init__(self):
        super().__init__()
        # 设置 API key
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    
    def _get_query_embedding(self, query: str) -> List[float]:
        response = TextEmbedding.call(
            model="text-embedding-v2",
            input=query
        )
        if response.status_code == 200:
            # 从 API 响应中提取嵌入向量
            embedding = response.output['embeddings'][0]['embedding']
            return embedding
        else:
            raise Exception(f"API调用失败: {response.message}")
    
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_query_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = TextEmbedding.call(
            model="text-embedding-v2",
            input=texts
        )
        if response.status_code == 200:
            # 从 API 响应中提取嵌入向量
            embeddings = [item['embedding'] for item in response.output['embeddings']]
            return embeddings
        else:
            raise Exception(f"API调用失败: {response.message}")
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)

class QwenLLM(CustomLLM):
    """千问模型包装器"""
    
    model_name: str = Field(default="qwen-max")
    temperature: float = Field(default=0.7)
    
    def __init__(self, model: str = "qwen-max", temperature: float = 0.7):
        super().__init__(
            model_name=model,
            temperature=temperature
        )
        # 设置 API key
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_name,
            temperature=self.temperature,
            context_window=8192,
            num_output=2048,
        )
    
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = Generation.call(
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
            **kwargs
        )
        if response.status_code == 200:
            return CompletionResponse(text=response.output.text)
        else:
            raise Exception(f"API调用失败: {response.message}")
    
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """实现流式输出"""
        response = Generation.call(
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
            stream=True,
            **kwargs
        )
        
        def generate() -> Generator[CompletionResponse, None, None]:
            try:
                for chunk in response:
                    if hasattr(chunk, 'output'):
                        #print(f"Debug - Output type: {type(chunk.output)}")
                        print(f"Debug - Output content: {chunk.output}")
                        if isinstance(chunk.output, dict):
                            if 'text' in chunk.output:
                                yield CompletionResponse(text=chunk.output['text'])
                        elif hasattr(chunk.output, 'text'):
                            yield CompletionResponse(text=chunk.output.text)
            except Exception as e:
                print(f"Debug - Error in stream: {str(e)}")
                raise Exception(f"流式输出失败: {str(e)}")
        
        return generate()

def create_index():
    """创建文档索引"""
    # 加载文档
    documents = SimpleDirectoryReader('data').load_data()
    
    # 创建 LLM
    llm = QwenLLM(temperature=0.7)
    
    # 创建千问嵌入模型
    embed_model = QwenEmbedding()
    
    # 创建服务上下文
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model
    )
    
    # 创建索引
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context
    )
    return index

def main():
    # 检查 API key
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("请设置 DASHSCOPE_API_KEY 环境变量")
        return

    print("正在创建文档索引...")
    index = create_index()
    
    print("\n文档索引创建完成！现在你可以开始提问了。")
    print("输入 'quit' 退出程序")
    
    while True:
        query = input("\n请输入你的问题: ")
        if query.lower() == 'quit':
            break
            
        try:
            # 创建查询引擎
            query_engine = index.as_query_engine(streaming=True)
            # 执行查询
            print("\n回答: ", end="", flush=True)
            response = query_engine.query(query)
            print(f"\nDebug - Response type: {type(response)}")
            print(f"Debug - Response content: {response}")
            if hasattr(response, 'response_gen'):
                print("Debug - Using response_gen")
                for text_chunk in response.response_gen:
                    print(f"Debug - Text chunk type: {type(text_chunk)}")
                    print(f"Debug - Text chunk content: {text_chunk}")
                    if isinstance(text_chunk, str):
                        print(text_chunk, end="", flush=True)
                    elif hasattr(text_chunk, 'text'):
                        print(text_chunk.text, end="", flush=True)
            else:
                print("Debug - Using response.response")
                print(response.response, end="", flush=True)
            print()  # 打印换行
        except Exception as e:
            print("发生错误: {}".format(str(e)))

if __name__ == "__main__":
    main() 