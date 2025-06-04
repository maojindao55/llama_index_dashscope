# LlamaIndex Demo

这是一个使用 LlamaIndex 构建的简单文档问答系统示例。

## 功能特点

- 支持加载本地文档
- 使用 LlamaIndex 进行文档索引
- 支持自然语言问答

## 安装

1. 克隆此仓库
2. 安装依赖：
```bash
pip install -r requirements.txt
```
3. 在项目根目录创建 `.env` 文件并添加你的 OpenAI API key：
```
OPENAI_API_KEY=your_api_key_here
```

## 使用方法

1. 将你的文档放在 `data` 目录下
2. 运行主程序：
```bash
python main.py
```

## 示例

程序会加载示例文档并允许你进行问答。示例问题：
- "文档的主要内容是什么？"
- "请总结一下文档的要点" # llama_index_dashscope
