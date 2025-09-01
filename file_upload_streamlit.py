import os
from dotenv import load_dotenv
import streamlit as st

# langchain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

# supabase
from supabase.client import Client, create_client

# load env
load_dotenv()
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

global vector_store

# clear DB
def clear_supabase_table():
    # Delete all rows safely
    supabase.table("documents").delete().neq(
        "id", "00000000-0000-0000-0000-000000000000"
    ).execute()

# LLM + agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = hub.pull("hwchase17/openai-functions-agent")

# retriever tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve info from the uploaded research paper."""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Streamlit App ---
st.set_page_config(page_title="Research Paper Chatbot", page_icon="📚", layout="wide")
st.title("📚 Research Paper Assistant")

# 2-column layout
left_col, right_col = st.columns([1,2])

with left_col:
    st.header("Step 1: Upload your PDF")
    uploaded_file = st.file_uploader("Upload a research PDF", type=["pdf"])

    if uploaded_file is not None:
        # clear previous DB
        clear_supabase_table()

        # save locally
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # load + chunk
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(documents)

        # insert into Supabase
        vector_store = SupabaseVectorStore.from_documents(
            docs,
            embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents",
            chunk_size=1000,
        )

        st.success("✅ Research paper embedded into knowledge base!")

# --- Right column: Chatbot ---
with right_col:
    st.header("Chat with your research paper")

     # chat input at the bottom
    user_question = st.chat_input("Ask a question about the uploaded paper...")

    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # render all previous messages
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            st.chat_message("user").markdown(message.content)
        else:
            st.chat_message("assistant").markdown(message.content)

   
    if user_question:
        # append user message
        st.session_state.messages.append(HumanMessage(user_question))
        st.chat_message("user").markdown(user_question)

        # invoke agent
        result = agent_executor.invoke({
            "input": user_question,
            "chat_history": st.session_state.messages
        })
        ai_message = result["output"]

        # append AI response
        st.session_state.messages.append(AIMessage(ai_message))
        st.chat_message("assistant").markdown(ai_message)
