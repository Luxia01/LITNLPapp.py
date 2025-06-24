import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import umap

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
