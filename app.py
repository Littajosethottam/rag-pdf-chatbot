import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("📄 Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    reader = PdfReader(uploaded_file)

    pages = []
    texts = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(i + 1)
            texts.append(text)

    # Create embeddings
    embeddings = []

    for text in texts:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embeddings.append(response.data[0].embedding)

    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    question = st.chat_input("Ask about the document")

    if question:

        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        q_embed = client.embeddings.create(
            model="text-embedding-3-small",
            input=question
        ).data[0].embedding

        D, I = index.search(np.array([q_embed]).astype("float32"), k=1)

        context = texts[I[0][0]]
        page = pages[I[0][0]]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer using the document context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:{question}"}
            ]
        )

        answer = response.choices[0].message.content

        answer_with_source = f"{answer}\n\n📄 Source: Page {page}"

        st.session_state.messages.append(
            {"role": "assistant", "content": answer_with_source}
        )

        st.chat_message("assistant").write(answer_with_source)