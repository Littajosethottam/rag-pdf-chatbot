import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
from openai import OpenAI

# Page config
st.set_page_config(page_title="AI PDF Assistant", page_icon="📄", layout="wide")

# OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("📄 AI PDF Assistant")

# Sidebar
st.sidebar.header("Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if st.sidebar.button("Reset Chat"):
    st.session_state.messages = []

# If PDFs uploaded
if uploaded_files:

    pages = []
    texts = []

    # Read PDFs
    with st.spinner("Reading PDFs..."):
        for file in uploaded_files:
            reader = PdfReader(file)

            for i, page in enumerate(reader.pages):
                text = page.extract_text()

                if text:
                    pages.append((file.name, i + 1))
                    texts.append(text)

    # Create embeddings
    with st.spinner("Creating embeddings..."):

        embeddings = []

        for text in texts:

            emb = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            ).data[0].embedding

            embeddings.append(emb)

    embeddings = np.array(embeddings).astype("float32")

    # FAISS vector index
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    question = st.chat_input("Ask something about your documents...")

    if question:

        # Show user message
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.write(question)

        # Create question embedding
        q_embed = client.embeddings.create(
            model="text-embedding-3-small",
            input=question
        ).data[0].embedding

        # Search vector DB
        D, I = index.search(np.array([q_embed]).astype("float32"), k=3)

        context = "\n\n".join([texts[i] for i in I[0]])
        source = pages[I[0][0]]

        # Ask LLM (non-streaming)
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "Answer using the provided document context."
                        },
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion: {question}"
                        }
                    ]
                )

                answer = response.choices[0].message.content

                st.write(answer)

                file_name, page_num = source

                st.caption(
                    f"📄 Source: {file_name} — Page {page_num}"
                )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

else:
    st.info("Upload one or more PDFs in the sidebar to start chatting.")