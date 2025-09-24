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

python scripts/data_preprocessing.py






colab Notebook :https://colab.research.google.com/drive/1Er9GaxDcS3YSu4lRm6V8pGdvOR7OekmL?usp=sharing
