# app.py

import streamlit as st
import os
import json
import requests
from datetime import datetime

# 设置 Streamlit 页面
st.set_page_config(page_title="用户 NLP 可解释系统", layout="centered")

# 用户管理
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

# 登录功能
def login():
    st.title("🔐 NLP 可解释分析系统")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    if st.button("登录"):
        if username == "liming" and password == "123456":
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("登录成功！")
            st.experimental_rerun()
        else:
            st.error("用户名或密码错误")

if not st.session_state.authenticated:
    login()
    st.stop()

# 日志目录
user_dir = f"user_data/{st.session_state.username}"
os.makedirs(user_dir, exist_ok=True)

st.title("🌟 NLP 模型分析 + LIT 解释 ")

# 文本输入
text_input = st.text_area("请输入需分析的文本：", "I love this movie so much!")

# 文本保存
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if text_input:
    with open(f"{user_dir}/text_{timestamp}.txt", "w") as f:
        f.write(text_input)

# 分析按钮
if st.button("推送给 LIT 解析"):
    st.info("正在通过 API 推送文本给 LIT...")

    # 转成 JSON
    input_json = {
        "text": text_input,
        "user": st.session_state.username
    }

    with open(f"{user_dir}/input_{timestamp}.json", "w") as f:
        json.dump(input_json, f)

    # 假设本地 LIT 服务器运行在 127.0.0.1:7777
    try:
        response = requests.post("http://localhost:7777/api/analyze", json=input_json)
        result = response.json()

        st.subheader("📉 分析结果")
        st.write("**类别概率**:", result.get("probs"))
        st.write("**预测结果**:", result.get("label"))

        st.subheader("💡 Attention 高亮")
        st.markdown(result.get("highlighted"), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"无法连接 LIT 服务器：{e}")
