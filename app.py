from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

conversation = [
    SystemMessage(content="You are a helpful assistant.")
]

def chat(user_input):
    conversation.append(HumanMessage(content=user_input))
    response = llm.invoke(conversation)
    conversation.append(AIMessage(content=response.content))
    return response.content

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    print(f"Assistant: {chat(user_input)}\n")