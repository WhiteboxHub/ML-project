# Generative AI and Machine Learning 14-Hour Workflow Plan

This guide outlines a structured 14-hour workflow plan to cover essential topics in Generative AI, retrieval systems, text-to-SQL agents, fraud detection, deployment, and monitoring. The plan balances learning, coding, and implementation for optimal efficiency.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Hourly Breakdown](#hourly-breakdown)
3. [Folder Structure](#folder-structure)
4. [Getting Started](#getting-started)
5. [Dependencies](#dependencies)
6. [How to Run](#how-to-run)

---

## Introduction

This project includes:
- Generative AI foundations.
- Retrieval-Augmented Generation (RAG) pipelines.
- Fraud detection system implementation.
- Deployment and monitoring of machine learning projects.

The goal is to efficiently design, train, evaluate, and deploy systems while ensuring modularity and reusability.

---

## Hourly Breakdown

### **Hour 1: Generative AI Foundations**
- Topics: Transformer architectures (GPT, Llama2, Claude 2), prompt engineering, chaining methods.
- Tasks:
  - Design simple prompts for Llama2 and Claude 2.

---

### **Hour 2: Web Scraping and Chunking**
- Topics: Data collection and preprocessing.
- Tasks:
  - Scrape data using BeautifulSoup/Scrapy.
  - Implement chunking methods:
    - Fixed-length chunks.
    - Overlapping sliding windows.
    - Semantic chunking with embeddings.

---

### **Hour 3: Vector Embeddings (Methods)**
- Topics: Embedding generation.
- Tasks:
  - Generate embeddings using:
    - Sentence Transformers.
    - Llama2 or OpenAI API.
  - Compare embedding quality.

---

### **Hour 4: Vector Embedding Storage**
- Topics: Embedding storage.
- Tasks:
  - Use pgvector (PostgreSQL), ChromaDB, and FAISS for storage.

---

### **Hour 5: Retrieval Agents**
- Topics: Retrieval methods.
- Tasks:
  - Implement sparse, dense, and hybrid retrieval.
  - Evaluate precision, recall, and latency.

---

### **Hour 6: Retrieval-Augmented Generation (RAG)**
- Topics: Building a RAG system.
- Tasks:
  - Implement pipelines using Llama2 and Claude 2.

---

### **Hour 7: Text-to-SQL Agent**
- Topics: Query reconstruction.
- Tasks:
  - Build a text-to-SQL agent using LangChain.
  - Add query orchestration.

---

### **Hour 8: Fine-Tuning LLMs**
- Topics: Enhancing model performance.
- Tasks:
  - Fine-tune Llama2 and Claude 2.
  - Compare performance improvements.

---

### **Hour 9: Evaluation and Hallucination**
- Topics: LLM response evaluation.
- Tasks:
  - Use BLEU, ROUGE, and human feedback for evaluation.
  - Align outputs with retrieval grounding.

---

### **Hour 10: Fraud Detection System – Data Pipeline**
- Topics: Data pipeline creation.
- Tasks:
  - Handle noisy data, including outlier removal and normalization.

---

### **Hour 11: Fraud Detection System – Training**
- Topics: Model training.
- Tasks:
  - Train an FCNN with PyTorch.
  - Train a LightGBM/XGBoost model.
  - Compare metrics: Precision, recall, F1-score, ROC-AUC.

---

### **Hour 12: Handling Data Drift**
- Topics: Adapting to data drift.
- Tasks:
  - Implement drift detection using statistical tests or anomaly detection methods.
  - Adjust model pipeline for incremental learning.

---

### **Hour 13: Deployment of Gen AI and ML Projects**
- Topics: Model deployment.
- Tasks:
  - Use FastAPI for RESTful APIs.
  - Deploy on SageMaker and RDS (pgvector).

---

### **Hour 14: Monitoring and CI/CD**
- Topics: Automating monitoring and deployment.
- Tasks:
  - Use CloudWatch for model performance tracking.
  - Set up a CI/CD pipeline on AWS.

---

## Folder Structure

```plaintext
project/
├── generative_ai/
│   ├── transformers/
│   │   ├── llama2.py
│   │   ├── claude2.py
│   │   └── prompt_engineering/
│   └── embeddings/
│       ├── sentence_transformers.py
│       ├── llama2_embeddings.py
│       └── openai_embeddings.py
├── data_preprocessing/
│   ├── scraping/
│   │   ├── beautifulsoup_scraper.py
│   │   └── scrapy_pipeline.py
│   └── chunking/
│       ├── fixed_length.py
│       ├── sliding_window.py
│       └── semantic_chunking.py
├── vector_storage/
│   ├── pgvector_setup.sql
│   ├── chromadb_setup.py
│   └── faiss_storage.py
├── retrieval_agents/
│   ├── sparse_retrieval.py
│   ├── dense_retrieval.py
│   └── hybrid_retrieval.py
├── rag_pipeline/
│   ├── llama2_rag.py
│   └── claude2_rag.py
├── text_to_sql_agent/
│   └── langchain_sql.py
├── fraud_detection/
│   ├── data_pipeline/
│   │   ├── outlier_removal.py
│   │   └── normalization.py
│   ├── training/
│   │   ├── fcnn_training.py
│   │   ├── lightgbm_training.py
│   │   └── metrics_comparison.py
│   └── data_drift/
│       ├── statistical_tests.py
│       └── anomaly_detection.py
├── deployment/
│   ├── fastapi/
│   │   └── app.py
│   ├── sagemaker/
│   │   └── deployment_script.py
│   └── ci_cd/
│       ├── aws_pipeline.yml
│       └── cloudwatch_monitoring.py
└── README.md
