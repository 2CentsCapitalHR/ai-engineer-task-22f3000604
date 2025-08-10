import langchain_helper as lch
import streamlit as st
from langchain_unstructured import UnstructuredLoader

st.title("Corporate Agent")

uploaded_file = st.file_uploader("Upload a .docx file", type=["docx"])  # DOCX only
query = st.text_input("Enter your query about the document")

if st.button("SUBMIT"):
    if uploaded_file and query:
        with st.spinner("Processing..."):

            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                loader = UnstructuredLoader(file_path=temp_path, partition_via_api=False)
                documents = loader.load()
                db = lch.vector_db_from_documents(documents)
                answer = lch.get_response_from_db(db, query)
                st.success(answer)
            except Exception as e:
                st.error(f"Processing failed: {e}")
    else:
        st.error("Please upload a document and enter a query.")