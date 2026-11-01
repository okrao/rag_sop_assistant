# HSA Policy RAG SOP Assistant

## Overview
A Retrieval-Augmented Generation (RAG) assistant designed to eliminate manual research time for customer support representatives. This tool ingests complex Health Savings Account (HSA) policy PDFs and allows agents to query them via semantic search, returning instant, accurate policy answers.

## Business Impact
* **Reduced Average Handling Time (AHT):** Replaces manual Ctrl+F document searches with instant semantic retrieval.
* **Improved First Call Resolution (FCR):** Ensures reps have immediate access to accurate, context-aware policy guidelines while on the phone with customers.

## Tech Stack
* **Language:** Python
* **LLM & Inference:** Groq API (High-speed inference)
* **Orchestration:** LangChain
* **Vector Database:** ChromaDB
* **Frontend:** Streamlit

## Core Capabilities
* **PDF Ingestion & Chunking:** Processes unstructured standard operating procedure (SOP) documents.
* **Semantic Search:** Retrieves highly relevant document chunks based on user intent.
* **Interactive UI:** Provides a clean, chat-like interface for agents to ask policy questions.