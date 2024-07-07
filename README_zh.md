## 概要

- 这是一个基本的知识库类型的大语言模型应用。可以帮助你理解这种类型应用的基本工作流程和功能。
- 你可以让大语言模型从你提供的上下文里回答你的问题。
- 主要步骤：向量化（内嵌），RAG（检索增强生成）。

## 环境

- 操作系统: Windows。
- 硬件: 不用GPU。
- 软件: Python和相关包。

## 开始

1. 运行“Vectorization.py”来向量化你的上下文并把结果保存到本地的“embeddingsBert.npy”和“chunksBert.txt”。
2. 运行“RAG.py”来加载“embeddingsBert.npy”和“chunksBert.txt”到内存，并生对其成索引，向量化问题，从索引中检索问题相关内容，生成并输出答案到命令行终端。
3. 你可以在“context.py”中指定你自己的上下文，在“RAG.py”中的“question”处设置你自己的问题。

## 许可

参见[README.md](README.md)和[LICENSE](LICENSE)。
