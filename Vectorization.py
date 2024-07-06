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

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
import torch
from Context import context # 示例大文本资料

# 加载BERT模型和分词器
# Init the model and tokenizer
model_name = "bert-base-chinese" # 支持中文
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 将文本分割成较小的段落
# split the context to chunks
def split_text(text, max_length=128):
    sentences = text.split('\n')
    chunks = []
    chunk = []
    current_length = 0
    for sentence in sentences:
        if current_length + len(sentence) <= max_length:
            chunk.append(sentence)
            current_length += len(sentence)
        else:
            chunks.append(''.join(chunk))
            chunk = [sentence]
            current_length = len(sentence)
    if chunk:
        chunks.append(''.join(chunk))
    return chunks

chunks = split_text(context)

# 向量化文本
# vectorize the context
def embed_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

embeddings = [embed_text(chunk, tokenizer, model) for chunk in chunks]
embeddings = np.vstack(embeddings)

# 保存向量化文本和原文
# save the vectorization result and the text.
np.save("embeddingsBert.npy", embeddings)
with open("chunksBert.txt", "w", encoding='utf-8') as f:
    for chunk in chunks:
        f.write(chunk + "\n")
