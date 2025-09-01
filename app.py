import streamlit as st
from vectorstore import clear_supabase_table, init_vector_store
from pdf_processsor import process_pdf
from agent import agent_executor
from langchain_core.messages import HumanMessage, AIMessage
from vectorstore import vector_store

st.set_page_config(page_title="Research Paper Chatbot", page_icon="📚", layout="wide")
st.title("📚 Research Paper Assistant")

left_col, right_col = st.columns([1,2])

# --- LEFT COLUMN: PDF Upload ---
with left_col:
    st.header("Step 1: Upload your PDF")
    uploaded_file = st.file_uploader("Upload a research PDF", type=["pdf"])

    if uploaded_file is not None:
        # clear previous DB
        clear_supabase_table()

        # save locally
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # process PDF into document chunks
        docs = process_pdf("temp.pdf")

        # insert into Supabase vector store
        init_vector_store(docs)

        st.session_state.vector_store = vector_store

        st.success("✅ Research paper embedded into knowledge base!")

# --- RIGHT COLUMN: Chatbot ---
with right_col:
    st.header("Chat with your research paper")

    # check if vector_store exists in session_state
    if "vector_store" not in st.session_state:
        st.info("Upload a PDF to start chatting with it.")
    else:
        # initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # chat input at bottom
        user_question = st.chat_input("Ask a question about the uploaded paper...")

        # show previous messages (latest at top)
        for message in (st.session_state.messages):
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