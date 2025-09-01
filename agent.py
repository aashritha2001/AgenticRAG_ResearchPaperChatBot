from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from vectorstore import vector_store
import streamlit as st

# LLM + prompt
llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = hub.pull("hwchase17/openai-functions-agent")

# Retriever tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve info from uploaded research paper."""
    retrieved_docs = st.session_state.vector_store.similarity_search(query, k=3)

    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve]

# Agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
