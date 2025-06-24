# 📁 项目结构
# nlp-vis-app/
# ├── app.py                  # 主入口：Streamlit UI + 功能逻辑
# ├── models/
# │   └── model_config.json   # 可选模型配置（名称 -> huggingface模型名）
# ├── data/
# │   └── sst2_dev.json       # 示例数据集
# ├── utils.py                # 功能函数（加载模型、绘图等）
# └── requirements.txt        # 依赖

# 🔹 Step 1: requirements.txt（依赖文件）
# ------------------------------
# streamlit
# torch
# transformers
# scikit-learn
# matplotlib
# umap-learn
# pandas

# 🔹 Step 2: models/model_config.json
# ------------------------------
# {
#   "sst2-tiny": "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english",
#   "sst2-base": "distilbert-base-uncased-finetuned-sst-2-english"
# }

# 🔹 Step 3: data/sst2_dev.json（示例）
# ------------------------------
# [
#   {"sentence": "it's a charming and often affecting journey.", "label": 1},
#   {"sentence": "unflinchingly bleak and desperate", "label": 0},
#   {"sentence": "a major career as a commercial yet inventive filmmaker", "label": 1}
# ]

# 🔹 Step 4: utils.py
# ------------------------------
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.manifold import TSNE
import umap
import numpy as np

@torch.no_grad()
def load_model_and_tokenizer(hf_model):
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    model = AutoModelForSequenceClassification.from_pretrained(hf_model)
    model.eval()
    return tokenizer, model

@torch.no_grad()
def compute_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs, output_hidden_states=True)
    cls_embeddings = outputs.hidden_states[-1][:, 0, :].numpy()
    reducer = umap.UMAP()
    reduced = reducer.fit_transform(cls_embeddings)
    return reduced


# 🔹 Step 5: app.py（主入口）
# ------------------------------
import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from utils import load_model_and_tokenizer, compute_embeddings

# ✅ 就放在这里

# 下面是你的页面设置、模型加载等代码
st.set_page_config(page_title="NLP 可视化平台", layout="wide")
st.title("🌟 NLP 模型可视化系统")

# 页面设置
st.set_page_config("🧠 NLP 模型解释平台", layout="wide")
st.title("🧠 NLP 模型可视化与解释平台 (Lite)")

# 模型配置
with open("models/model_config.json") as f:
    model_dict = json.load(f)

model_name = st.sidebar.selectbox("选择模型", list(model_dict.keys()))
tokenizer, model = load_model_and_tokenizer(model_dict[model_name])

# 数据加载
uploaded_file = st.sidebar.file_uploader("上传数据 (JSON, 包含 sentence 和 label 字段)", type="json")
if uploaded_file:
    data = pd.read_json(uploaded_file)
else:
    data = pd.read_json("data/sst2_dev.json")

# 可视化嵌入（降维）
embeddings = compute_embeddings(data["sentence"].tolist(), tokenizer, model)

# 分类预测
inputs = tokenizer(data["sentence"].tolist(), return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()
preds = probs.argmax(axis=1)

# 主界面布局
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📄 数据表")
    st.dataframe(data)
    selected_idx = st.number_input("选择索引以查看详细信息", min_value=0, max_value=len(data)-1, value=0)

    st.markdown("#### 🧾 模型分类结果")
    st.write(f"**Sentence:** {data['sentence'][selected_idx]}")
    st.write(f"**True Label:** {data['label'][selected_idx]}")
    st.write(f"**Predicted:** {preds[selected_idx]} | Confidence: {probs[selected_idx].max():.2f}")

with col2:
    st.subheader("📊 嵌入投影")
    fig, ax = plt.subplots()
    colors = ["blue" if p==1 else "orange" for p in preds]
    ax.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, alpha=0.7)
    ax.scatter(embeddings[selected_idx, 0], embeddings[selected_idx, 1], c="red", s=100, label="Selected")
    ax.legend()
    st.pyplot(fig)

st.markdown("---")
st.caption("🔗 本页面基于 HuggingFace + UMAP + Streamlit 构建，支持嵌入投影、分类解释、文本分析等功能。")

# 🔚


