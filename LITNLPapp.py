import streamlit as st
from modules.model_loader import load_model
from modules.dataset_loader import load_dataset
from modules.embedding_visualizer import plot_embeddings
from modules.datapoint_selector import select_datapoints
from modules.scalars import scalar_analysis

# 侧边栏
st.sidebar.title("NLP LIT 工具")
model_name = st.sidebar.selectbox("选择模型", ["sst2-tiny", "sst2-base"])
dataset_name = st.sidebar.selectbox("选择数据集", ["sst_dev"])
selection_mode = st.sidebar.selectbox("数据点选择方式", ["随机", "相关", "父项", "子项"])
color_mode = st.sidebar.selectbox("数据点着色", ["类别", "预测误差"])
slice_action = st.sidebar.radio("数据点切片", ["创建", "添加", "删除"])
fixed_datapoint = st.sidebar.checkbox("固定数据点")
if st.sidebar.button("Copy Link"):
    st.write("已复制当前视图链接！")

# 加载模型和数据
model, tokenizer = load_model(model_name)
dataset = load_dataset(dataset_name)

# 数据点选择
selected_indices = select_datapoints(dataset, mode=selection_mode)

# 主页面
st.title("NLP 模型可视化与解释工具")

# Datapoint Editor
st.header("Datapoint Editor")
# ...显示和编辑数据点...

# Classification Results
st.header("Classification Results")
# ...显示分类结果和置信度...

# Scalars
st.header("Scalars")
scalar_analysis(dataset, selected_indices)

# Embeddings
st.header("Embeddings")
plot_embeddings(model, tokenizer, dataset, selected_indices, color_mode)

# Data Table
st.header("Data Table")
st.dataframe(dataset)



