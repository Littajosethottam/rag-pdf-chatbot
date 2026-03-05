import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
from openai import OpenAI

st.set_page_config(page_title="AI PDF Assistant", page_icon="📄", layout="wide")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("📄 AI PDF Assistant")

st.sidebar.header("Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if st.sidebar.button("Reset Chat"):
    st.session_state.messages = []

if uploaded_files:

    pages = []
    texts = []

    with st.spinner("Reading PDFs..."):
        for file in uploaded_files:
            reader = PdfReader(file)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages.append((file.name, i + 1))
                    texts.append(text)

    with st.spinner("Creating embeddings..."):
        embeddings = []
        for text in texts:
            emb = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            ).data[0].embedding
            embeddings.append(emb)

    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask something about your documents...")

    if question:

        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        q_embed = client.embeddings.create(
            model="text-embedding-3-small",
            input=question
        ).data[0].embedding

        D, I = index.search(np.array([q_embed]).astype("float32"), k=3)

        context = "\n\n".join([texts[i] for i in I[0]])
        source = pages[I[0][0]]

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer using the document context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:{question}"}
            ],
            stream=True
        )

        with st.chat_message("assistant"):

            response_text = ""
            placeholder = st.empty()

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
                    placeholder.write(response_text + "▌")

            placeholder.write(response_text)

            file_name, page_num = source
            st.caption(f"📄 Source: {file_name} — Page {page_num}")

        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )

else:
    st.info("Upload one or more PDFs in the sidebar to start.")