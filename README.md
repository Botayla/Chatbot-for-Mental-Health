# Chatbot-for-Mental-Health
# 🧠 Mental Health Chatbot with RAG

This project is a **Mental Health Chatbot** built using **Retrieval-Augmented Generation (RAG)** combined with a **fine-tuned GPT-2 model**.  
The idea is to enhance a trained language model on mental health FAQs with external **PDF knowledge sources**, enabling more accurate and grounded answers.

---
🛠️ Steps
1️⃣ Data Preprocessing

    - Original file: Mental_Health_FAQ.csv

    - Dropped column: Question_ID

  -  Combined each question and answer into:

`Q: ...

A: ...`


Saved as: processed_Mental_Health_FAQ.csv

Run:

`python scripts/data_preprocessing.py`

2️⃣ Model Training

Base model: GPT-2

    - Added a special padding token for batching

    - Train/test split: 80/20

    - Training handled with HuggingFace Trainer API

Run:

`python scripts/train.py`

3️⃣ RAG Pipeline

 1- Load PDFs from resources/

2- Split documents into chunks using RecursiveCharacterTextSplitter

3- Convert chunks to embeddings with:

4 - sentence-transformers/all-MiniLM-L6-v2

5- Store and search embeddings using FAISS

6- Define GPT-2 as the LLM using HuggingFace pipeline

 7- Build RetrievalQA chain with LangChain

Run:

`python scripts/rag_pipeline.py`

🔍 Example Usage

After running rag_pipeline.py, you can query the system:

`query = "What are the main symptoms of depression?"

response = qa_chain.run(query)

print(response)`
