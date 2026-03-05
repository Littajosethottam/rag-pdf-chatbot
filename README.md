# 📄 AI PDF Assistant

An AI-powered application that allows users to upload PDF documents and ask questions about their content.

The app uses Retrieval-Augmented Generation (RAG) to search document text and generate answers using a language model.

🔗 **Live App:** [https://rag-pdf-chatbot.streamlit.app  ](https://rag-pdf-chatbot-8npmfnvpj3mcebbvfcueus.streamlit.app/)

🔗 **GitHub Repository:** https://github.com/Littajosethottam/rag-pdf-chatbot

---
## Demo

![AI PDF Assistant Interface](RAG_chatbot.png)
## Features

- Upload one or multiple PDF documents
- Ask questions about document content
- Semantic search using embeddings
- Vector similarity search with FAISS
- AI-generated answers based on document context
- Source citation showing the document and page number
- Interactive chat interface built with Streamlit

---

## How It Works

The application follows a Retrieval-Augmented Generation (RAG) pipeline:

Instead of sending the entire document to the language model, the system retrieves **only the most relevant sections** and uses them as context to produce accurate answers.


## Project Motivation

Many organizations and researchers work with large collections of documents such as reports, research papers, manuals, and policy documents. Searching through these files manually can be inefficient and time-consuming.

This project explores how **AI-powered document assistants** can transform static PDFs into interactive knowledge systems. By combining vector search with language models, users can quickly access insights from complex documents through simple natural language queries.

The project demonstrates how modern AI systems can support **knowledge discovery, document understanding, and decision support**.


## Example Use Cases

### Academic Research
- Summarizing research papers
- Extracting key concepts from study materials
- Supporting exam preparation

### Business Intelligence
- Searching internal reports and documentation
- Extracting insights from strategy or financial documents
- Supporting data-driven decision making

### Knowledge Management
- Turning document collections into searchable AI assistants
- Supporting internal documentation systems

### Legal & Compliance
- Quickly locating information in regulatory or policy documents
- Extracting relevant clauses from large legal texts


## Tech Stack

- **Python**
- **Streamlit**
- **OpenAI API**
- **FAISS (Vector Similarity Search)**
- **PyPDF**
- **NumPy**


## Installation

Clone the repository:
git clone https://github.com/Littajosethottam/rag-pdf-chatbot.git

cd rag-pdf-chatbot


Install dependencies:


pip install -r requirements.txt


Run the application:


streamlit run app.py



## Environment Setup

Create a `.streamlit/secrets.toml` file and add your OpenAI API key:


OPENAI_API_KEY = "your-api-key"



## Example Questions

You can ask questions such as:

- Summarize this document
- What are the key concepts discussed?
- Explain the methodology used
- What conclusions does the paper draw?
- List the main findings of this report


## What This Project Demonstrates

This project showcases several key AI engineering concepts:

- Retrieval-Augmented Generation (RAG)
- Vector embeddings for semantic search
- Vector databases using FAISS
- AI-powered document analysis
- Interactive AI applications using Streamlit


## Author

**Litta Jose Thottam**

LinkedIn  
https://linkedin.com/in/litta-thottam-12a223193

GitHub  
https://github.com/Littajosethottam


## License

This project is intended for educational and demonstration purposes.




