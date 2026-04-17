from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

search = DuckDuckGoSearchRun()
tools = [search]

agent = create_react_agent(llm, tools)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    print(f"Assistant: {response['messages'][-1].content}\n")