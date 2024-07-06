# Copyright 2024-, RGBYang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from colorama import Fore, Style

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 加载向量化的文本和原文
# load the vectorization and the text.
embeddings = np.load("embeddingsBert.npy")
with open("chunksBert.txt", "r", encoding='utf-8') as f:
    chunks = f.readlines()

# 初始化Faiss索引
# init the Faiss index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# 加载BERT模型和分词器（用于生成问题的嵌入）
# init model and tokenizaer
model_name = "bert-base-chinese"
bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)

# 加载生成模型和分词器（用于生成答案）
# init the model and the tokenizaer for generating the answer.
gen_model_name = "qwen/Qwen2-0.5B"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name)

# 定义embed_text函数
# define vectorization function
def embed_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

# 用户输入的问题
# input the question
question = "巴黎奥运会最贵的票多少钱？"

# 向量化问题
# vectorize the question
question_embedding = embed_text(question, bert_tokenizer, bert_model)

# 检索相关内容
# serch form the index, and select the best k chunks to retrieved_chunks.
D, I = index.search(question_embedding.reshape(1, -1), k=3) #k=5
retrieved_chunks = [chunks[i] for i in I[0]]

# 将检索到的内容拼接
# concatenate retrieved_chunks to a context string
context = " ".join(retrieved_chunks)
print(Fore.GREEN+"从向量库中检索出的上下文:", Style.RESET_ALL+context)
print(Fore.GREEN+"问题:", Style.RESET_ALL+question)

# 使用哪种关键字生成prompt取决于你使用的模型支持哪些语言
# you can make a prompt with English keywords like "context: {context}\n\nquestion: {question}\nanswer:xxx".
# depends on the supported languages by your model.
prompt = f"上下文: {context}\n\n问题: {question}\n答案:"

# 生成答案
# generate the answer
inputs = gen_tokenizer(prompt, return_tensors="pt")
outputs = gen_model.generate(
    inputs["input_ids"],
    max_length=800,
    do_sample=True,
    temperature=0.5,
    top_k=5,
    top_p=0.75,
    pad_token_id=gen_tokenizer.eos_token_id
)

# 解码并打印答案
# cecode and print the real answer
# 期望的fullAnswer里应该是"上下文: {context}\n\n问题: {question}\n答案:xxx"这种格式，但有时会被截断或不对。
# expected fullAnswer here should like "context: {context}\n\nquestion: {question}\nanswer:xxx".
fullAnswer = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
keyword = '答案:'
answer = ''
if keyword in fullAnswer: 
    answer = fullAnswer.split(keyword)[-1].strip()
else:
    answer = fullAnswer.strip()
print(Fore.RED+"最相关的答案是:", Style.RESET_ALL + answer)