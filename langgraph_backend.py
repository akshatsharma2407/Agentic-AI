from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

summarizer = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chatnode(state: ChatState):
    messages = state['messages']
    response = model.invoke(messages)
    return {'messages' : [response]}

graph = StateGraph(ChatState)
graph.add_node('chatnode', chatnode)

graph.add_edge(START, "chatnode")
graph.add_edge("chatnode", END)
checkpointer = InMemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)