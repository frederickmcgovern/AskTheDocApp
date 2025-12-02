# Updated imports
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def generate_response(uploaded_file, openai_api_key, query_text: str) -> str:
    if uploaded_file is None:
        return "No file uploaded."

    # Make sure we read from the beginning of the file each time
    uploaded_file.seek(0)
    document_text = uploaded_file.read().decode("utf-8")

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )
    texts = text_splitter.create_documents([document_text])

    # Embeddings
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    # Vector store
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    # LLM
    llm = ChatOpenAI(api_key=openai_api_key, temperature=0)

    # RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )

    # Run the chain
    return qa.run(query_text)


# ----- Streamlit UI -----

st.set_page_config(page_title="🦜🔗 Ask the Doc App")
st.title("🦜🔗 Ask the Doc App")

# File upload
uploaded_file = st.file_uploader("Upload an article", type="txt")

# Query text
query_text = st.text_input(
    "Enter your question:",
    placeholder="Please provide a short summary.",
    disabled=not uploaded_file,
)

result = []

with st.form("myform", clear_on_submit=True):
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        disabled=not (uploaded_file and query_text),
    )
    submitted = st.form_submit_button(
        "Submit",
        disabled=not (uploaded_file and query_text),
    )

    # Just check that a key was provided
    if submitted and openai_api_key:
        with st.spinner("Calculating..."):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
            del openai_api_key  # don’t keep it in memory

if result:
    st.info(result[-1])
