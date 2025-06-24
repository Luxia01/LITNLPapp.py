# app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt
import pandas as pd
import random

# -----------------------------
# 初始化页面设置
# -----------------------------
st.set_page_config(page_title="NLP 模型可视化系统", layout="wide")
st.title("🔍 基于 DistilBERT 的 NLP 模型可视化 Demo")

# -----------------------------
# 示例数据集
# -----------------------------
def load_sample_data():
    return pd.DataFrame({
        "text": [
            "I really loved the movie!",
            "The product was terrible and disappointing.",
            "It was okay, not great.",
            "Fantastic experience, would recommend!",
            "Worst decision ever.",
            "Nothing special, average performance.",
            "Highly enjoyable, brilliant acting.",
            "Awful, just awful.",
            "A masterpiece of cinema!",
            "Wouldn’t watch it again."
        ]
    })

# -----------------------------
# 模型加载
# -----------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# 文本输入或样本选择
# -----------------------------
st.sidebar.header("📌 数据输入")
mode = st.sidebar.radio("选择输入方式：", ["自由输入", "示例样本"])

if mode == "自由输入":
    text = st.text_area("请输入英文文本：", "I really enjoyed the film!")
else:
    df = load_sample_data()
    idx = st.sidebar.number_input("选择样本编号", min_value=0, max_value=len(df)-1, step=1)
    text = df.iloc[idx]["text"]
    st.info(f"选中文本：{text}")

# -----------------------------
# 运行模型并显示结果
# -----------------------------
if st.button("开始分析"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()

    labels = ["Negative", "Positive"]
    pred_label = labels[probs.argmax().item()]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧾 模型预测结果")
        st.write(f"**预测标签：** {pred_label}")
        st.write(f"**分类概率：** {dict(zip(labels, [round(float(p), 3) for p in probs]))}")

        fig, ax = plt.subplots()
        ax.bar(labels, probs.tolist(), color=["red", "green"])
        ax.set_ylim([0, 1])
        st.pyplot(fig)

    with col2:
        st.subheader("🧠 模拟 Attention 高亮")
        tokens = tokenizer.tokenize(text)
        importance = torch.rand(len(tokens))  # 随机模拟注意力值
        highlighted = ""
        for token, score in zip(tokens, importance):
            token_clean = token.replace("##", "")
            opacity = round(float(score), 2)
            highlighted += f"<span style='background-color: rgba(255,255,0,{opacity}); padding:2px'>{token_clean}</span> "
        st.markdown(highlighted, unsafe_allow_html=True)

# -----------------------------
# 底部信息
# -----------------------------
st.markdown("---")
st.caption("🎯 本平台由 Streamlit + HuggingFace Transformers 构建，当前为 Demo 版本，仅用于教学用途。")

