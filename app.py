from llama_index.llms.ollama import Ollama
from llama_index.core.chat_engine.types import ChatMode
import streamlit as st
from pathlib import Path
from rag_functions import (
    Create_Vector,
    Create_Graph,
    init_llm_ollama,
    init_index,
)
import shutil
import os

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="./llama.png",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)


def Clear_Chat() -> None:
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)


def main() -> None:
    menu = ["Document Ingestion", "Chatbot"]
    # st.sidebar.image("./llama.png", use_column_width=True)
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Chatbot":
        st.header('ECS SD System Testing RAG', divider='rainbow')
        init_llm_ollama()
        index = init_index()
        if "chat_engine" not in st.session_state.keys():
            st.session_state.chat_engine = index.as_chat_engine(
                chat_mode="condense_question", streaming=True, verbose=True
            )
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Ask me anything",
                }
            ]
        if prompt := st.chat_input("Ask me a question"):
            st.session_state.messages.append({"role": "user", "content": prompt})
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    st.empty()
                    response = st.session_state.chat_engine.chat(message=prompt)
                    st.write(response.response)
                    message = {"role": "assistant", "content": response.response}
                    st.session_state.messages.append(message)

    elif choice == "Document Ingestion":
        st.subheader("Document Ingestion")
        with st.form("Cleanup"):
            clean = st.form_submit_button("Reinitialize")
            st.write("This helps remove all previous data & chat history")
            if clean:
                try:
                    shutil.rmtree("./data")
                    os.makedirs("data")
                    Clear_Chat()
                    # load_client = chromadb.PersistentClient(path="./chroma_db")
                    # load_client.clear_system_cache
                    st.success("Successfully Reinitialized")
                except:
                    os.makedirs("data")
                    st.error("Nothing to initialize")
        with st.form(key="FileUpload", clear_on_submit=True):
            uploaded_file = st.file_uploader(
                "Drop your files here", type=["txt", "csv", "pdf", "py"]
            )
            Submit = st.form_submit_button(label="Save")
            if Submit:
                st.markdown("**The file is sucessfully Saved.**")
                # Save uploaded file to './data' folder.
                save_folder = "./data"
                save_path = Path(save_folder, uploaded_file.name)
                with open(save_path, mode="wb") as f:
                    f.write(uploaded_file.getvalue())
        if uploaded_file is not None:
            select_rag = st.radio(
                "Select RAG Type",
                ["Vector Database", "Graph Database"],
                horizontal=True,
            )
            if select_rag == "Graph RAG  (Advanced RAG using Graph Database)":
                st.info("Not Implemented Yet")
            vectorized = st.button("Embed & Vectorize")
            if vectorized:
                Clear_Chat()
                if select_rag == "Vector Database":
                    with st.spinner("Please wait while I vectorize..."):
                        Create_Vector()
                        st.success("Done...You can launch the Chatbot now")
                elif select_rag == "Graph Database":
                    Create_Graph()


if __name__ == "__main__":
    main()
