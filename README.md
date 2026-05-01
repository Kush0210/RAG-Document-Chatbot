# 🧠 Enterprise RAG Architecture: Document Intelligence Engine

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Native-green.svg)](https://python.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-LPU_Inference-orange.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Live Demo:** [Insert your Streamlit Cloud URL here]

A high-performance, fault-tolerant Retrieval-Augmented Generation (RAG) system designed to securely ingest, vectorize, and query PDF documents. This project demonstrates modern AI engineering principles, specifically focusing on data privacy, low-latency inference, and system reliability.

## 🚀 Architectural Highlights

*   **Zero-Cost, Privacy-First Embeddings:** Replaced traditional cloud embedding APIs with local Hugging Face CPU models (`all-MiniLM-L6-v2`). This ensures document text is vectorized locally, drastically reducing cloud latency and keeping document data secure prior to the LLM inference stage.
*   **High-Speed Inference:** Utilizes **Groq's LPU architecture** to run Meta's `Llama-3.3-70b-versatile`, delivering near-instant conversational responses.
*   **Automated Fault Tolerance:** Implemented LangChain's `.with_fallbacks()` mechanism. If the primary 70B model hits rate limits (HTTP 429) or experiences downtime, the system automatically and silently routes the prompt to a secondary, high-availability 8B model, ensuring zero downtime for the end user.
*   **Optimized Vector Search:** Leverages FAISS (Facebook AI Similarity Search) with optimized `k` retrieval constraints to protect LLM Token-Per-Minute (TPM) limits while maintaining highly relevant context windows.

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend/UI** | Streamlit | Lightweight, Pythonic web application framework. |
| **Orchestration** | LangChain (LCEL) | Pure LangChain Expression Language for pipeline routing. |
| **Primary LLM** | Groq (`Llama-3.3-70B`) | Cloud inference utilizing Language Processing Units (LPUs). |
| **Fallback LLM** | Groq (`Llama-3.1-8B`) | High-availability secondary inference model. |
| **Embeddings** | HuggingFace CPU | `all-MiniLM-L6-v2` running entirely locally. |
| **Vector Database** | FAISS | In-memory similarity search and dense vector clustering. |
