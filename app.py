import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MT5Tokenizer, MT5ForConditionalGeneration

# ---- Load RAG models ----
@st.cache_resource
def load_rag_models():
    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return embed_model, gen_tokenizer, gen_model

# ---- Load fine-tuned model ----
@st.cache_resource
def load_finetuned_model():
    model = MT5ForConditionalGeneration.from_pretrained("./flan_t5_finetuned_output1/checkpoint-1350", local_files_only=True)
    tokenizer = MT5Tokenizer.from_pretrained("./flan_t5_finetuned_output1/checkpoint-1350", local_files_only=True)
    return model, tokenizer

# ---- Load RAG data ----
@st.cache_resource
def load_rag_data():
    index = faiss.read_index("rag_index_clean.faiss")
    with open("rag_texts_clean.pkl", "rb") as f:
        texts = pickle.load(f)
    return index, texts

# Load everything
embed_model, gen_tokenizer, gen_model = load_rag_models()
fine_model, fine_tokenizer = load_finetuned_model()
index, texts = load_rag_data()

# ---- Streamlit UI ----
st.title("🚗 Car Sentiment Chatbot (RAG vs Fine-Tuned mT5)")

query = st.text_area("Enter your question about AMG GT XX or related car topics:")
model_choice = st.radio("Select model:", ["RAG", "Fine-tuned LLM"])

if st.button("Generate Answer"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        if model_choice == "RAG":
            # Encode + retrieve
            query_vec = embed_model.encode([query]).astype("float32")
            D, I = index.search(query_vec, k=5)
            retrieved_texts = [texts[i] for i in I[0]]

            # Generate
            context = " ".join(retrieved_texts)
            prompt = f"Question: {query}\nContext: {context}\nAnswer:"
            inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = gen_model.generate(**inputs, max_length=100)
            answer = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Show
            st.markdown(f"*RAG Answer:* {answer}")
            st.markdown("*Retrieved snippets:*")
            for r in retrieved_texts:
                st.write(f"- {r[:200]}...")

        else:  # Fine-tuned mT5
            inputs = fine_tokenizer(query, return_tensors="pt", truncation=True, padding=True)
            outputs = fine_model.generate(**inputs, max_length=50)
            answer = fine_tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.markdown(f"*Fine-tuned mT5 Answer:* {answer}")
