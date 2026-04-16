from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

# LLM

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load and chunk docs
with open("docs.txt", "r") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.create_documents([text])

# Vector store
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Memory
conversation = [
    SystemMessage(content="You are a helpful assistant. Answer from the context provided.")
]

def chat(user_input):
    docs = retriever.invoke(user_input)
    context = "\n".join([d.page_content for d in docs])
    augmented = f"Context:\n{context}\n\nQuestion: {user_input}"
    
    conversation.append(HumanMessage(content=augmented))
    response = llm.invoke(conversation)
    conversation.append(AIMessage(content=response.content))
    return response.content

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    print(f"Assistant: {chat(user_input)}\n")