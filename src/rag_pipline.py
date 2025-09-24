from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import GPT2Tokenizer , GPT2LMHeadModel
from transformers import pipeline


# Load Documents
def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def rag_pipeline(documents , model , tokenizer ):
    # split documents to chunks 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Number of chunks {len(texts)}")

    # embedding each chunk 
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # indexing with FAISS

    db = FAISS.from_documents(texts, embeddings)
    retriver = db.as_retriever(search_kwargs={'k': 2})

    # define LLM
    hf_pipeline = pipeline(
    "text-generation", model=model,tokenizer=tokenizer,
      max_new_tokens=300)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # qa_chain = RetrievalQA( llm = llm,retriever = retriver )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,                # Pass your LLM here
        chain_type="map_reduce",      # This is the default chain type (stuffs all documents into the prompt)
        retriever=retriver      # Pass your retriever instance here
    )

    return qa_chain



pdf1 = load_documents(r"resources\Anxiety-Disorders-Explained-NIMH-1.pdf")
pdf2 = load_documents(r"resources\bipolar-disorder_0.pdf")
pdf3 = load_documents(r"resources\NIMH_Depression.pdf")
pdf4 = load_documents(r"resources\post-traumatic-stress-disorder.pdf")
pdf5 = load_documents(r"resources\schizophrenia.pdf")

documents = pdf1 + pdf2 + pdf3 + pdf4 + pdf5
# get pre_trained model & tokenizer 
model = GPT2LMHeadModel.from_pretrained(r"Models\mental_health_gpt2")
tokenizer = GPT2Tokenizer.from_pretrained(r"Models\mental_health_gpt2")
model.config.pad_token_id = tokenizer.pad_token_id

qa_chain = rag_pipeline(documents , model , tokenizer)