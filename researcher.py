from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
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

SYSTEM_PROMPT = """You are an expert research assistant. When given a topic:
1. Search the web for latest information
2. Summarize findings in a structured format
3. Include: Overview, Key Points, Latest Developments, and Conclusion
Be concise but comprehensive."""

def research(topic):
    print(f"\nResearching: {topic}\n")
    print("-" * 50)
    
    response = agent.invoke({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Research this topic thoroughly: {topic}"}
        ]
    })
    
    return response["messages"][-1].content

while True:
    topic = input("\nEnter research topic (or 'exit'): ")
    if topic.lower() == "exit":
        break
    result = research(topic)
    print(f"\n{result}\n")
    print("=" * 50)