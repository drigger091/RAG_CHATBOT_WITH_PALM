import os

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from langchain_community.llms import GooglePalm
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY is not set")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY



def get_pdf_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap= 20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store =FAISS.from_texts(text_chunks,embedding= embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    
    llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=GOOGLE_API_KEY, temperature=0.1)

    memory = ConversationBufferMemory(memory_key = "chat_history",return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vector_store.as_retriever(),memory=memory)
    return conversation_chain


            
    
    