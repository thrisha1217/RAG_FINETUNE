**Multilingual Sentiment-Aware AI Assistant**
This project implements a multilingual AI assistant designed to generate sentiment-aware responses using data from public social media posts (English and German). It offers two modes:

**Retrieval-Augmented Generation (RAG)** using sentence-transformer embeddings, FAISS vector search, and an open-source instruct model.

**Fine-tuned generative model (mT5-small)** trained directly on labeled data.

A **Streamlit UI** allows users to interact with both systems and compare their performance (A/B testing). The assistant is designed for use cases like customer feedback analysis, community moderation, and sentiment monitoring.

**🚀 Features**
Collects and structures multilingual (English + German) social media posts (Reddit, YouTube, forums).

Adds sentiment labels (positive, negative, neutral) to ~2,000 posts.

**RAG system:**

Sentence-transformer embeddings (e.g. paraphrase-multilingual-MiniLM)

FAISS vector database for fast retrieval

Open-source instruct model (e.g. LLaMA, Mistral) for response generation

**Fine-tuned model:
**
mT5-small fine-tuned on labeled posts to generate direct responses

Streamlit UI with model toggle for A/B testing

Evaluation of accuracy, latency, runtime cost, transparency, and maintainability

🏗️ Architecture
 ┌───────────────────────────────┐
 │      Data Collection           │
 │ (Reddit, YouTube, forums)      │
 └─────────────┬─────────────────┘
               │
               ▼
 ┌───────────────────────────────┐
 │  Preprocessing + Sentiment     │
 │        Labeling (~2k)          │
 └─────────────┬─────────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
 ┌──────────────┐ ┌──────────────┐
 │  RAG system  │ │ Fine-tuned    │
 │              │ │ mT5-small     │
 └──────┬───────┘ └──────┬───────┘
        │                │
        ▼                ▼
 ┌───────────────────────────────┐
 │        Streamlit UI            │
 │   (Textbox + Model Toggle)     │
 └─────────────┬─────────────────┘
               │
               ▼
 ┌───────────────────────────────┐
 │         Evaluation             │
 │ (accuracy, latency, cost, etc) │
 └───────────────────────────────┘

**⚙️ Installation
Clone the repo:**
git clone https://github.com/yourusername/your-repo.git
cd your-repo
**
Install dependencies:**
pip install -r requirements.txt
Download required models:

Sentence-transformer (e.g. paraphrase-multilingual-MiniLM)

mT5-small from HuggingFace

FAISS for vector search

**📝 Usage
Run the app:**

bash
Copy
Edit
streamlit run app.py
You can enter a query and toggle between:

RAG (retrieves relevant posts, generates response + sources)

Fine-tuned (direct generation)

**📊 Data Collection**
Sources:

Reddit comments

YouTube comments

Public forums

**Dataset format:**

text	language	source	label (if labeled)
Post text	en/de	Reddit / YouTube / etc	positive / negative / neutral

Collected: ~3,000–5,000 posts
Labeled: ~2,000 posts for training/testing

**🤖 Model Details**
RAG System
Embedding: paraphrase-multilingual-MiniLM

Vector DB: FAISS

Instruct model: Open-source (e.g. LLaMA, Mistral)

Top-k retrieval → generate response with reference snippets

Fine-tuned Model
Base: mT5-small

Fine-tuned on ~2,000 labeled posts

Directly generates sentiment-aware responses

**📈 Evaluation**
Metric	Description
Accuracy	F1-score on ~300 labeled posts
Latency	Avg. response time per query
Runtime cost	Tokens processed per 1K → estimated compute cost
Transparency (RAG)	% responses including source snippets
Maintainability	Time to update DB or retrain model

**🔧 Configuration
Parameters:
**
top_k: number of retrieved posts in RAG mode

Model toggle: switch between RAG and fine-tuned mode in UI

🤝 Contributing
Contributions welcome!
Please:

Fork the repo

Create a feature branch (git checkout -b feature-name)

Submit a pull request

📄 License
MIT License — see LICENSE file for details.

**🙌 Acknowledgments
HuggingFace Transformers**

FAISS

Sentence-Transformers

Streamlit

🔮 Future Work
Add more languages (e.g. French, Spanish)

Improve sentiment labeling automation

Add explainability layer (highlight key evidence)
