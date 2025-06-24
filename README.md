Section Name Retriever with FAISS and Transformers

This project implements three different strategies to retrieve relevant section names for a given product using semantic similarity and category classification. The core idea is to embed product descriptions and section titles using sentence-transformers, and use FAISS for fast vector similarity search.

You explore three levels of complexity:
1. FAISS-based section retrieval given a known category
2. FAISS-based category inference + section retrieval
3. Zero-shot category classification + FAISS-based section retrieval

Project Structure
-----------------
 
``` . ├── Section_List.csv # Dataset with columns: Category 1, Section Name 
      ├── section_retriever_faiss.py # Version 1: Requires known category 
      ├── section_retriever_faiss_infer.py # Version 2: Infers category using FAISS 
      ├── section_retriever_zero_shot_faiss.py # Version 3: Zero-shot category classification 
      ├── README.md # You are here 
      └── *.faiss.index / *.pkl # Auto-saved FAISS indices and mappings per category 
``` 


Use Cases & Methods
-------------------

Problem Statement:
Input: A product description (and optionally a category)
Output: Top N matching section names for placing this product in a content template / web layout / e-commerce site.

Strategy 1: FAISS-Based Retrieval (Known Category)
--------------------------------------------------
File: section_retriever_faiss.py

- You input both the product info and the known category.
- The section names are embedded using SentenceTransformer.
- A FAISS IndexFlatIP index is built per category.
- It finds top-k sections closest to the product embedding.

Strategy 2: Category Inference using FAISS Embedding Similarity
---------------------------------------------------------------
File: section_retriever_faiss_infer.py

- You only provide the product info.
- It first embeds all known categories and builds a FAISS index over them.
- The closest category is selected based on cosine similarity.
- Then falls back to the Strategy 1 method for retrieving top sections.

Strategy 3: Category Inference using Zero-Shot Classification
-------------------------------------------------------------
File: section_retriever_zero_shot_faiss.py

- Uses HuggingFace's facebook/bart-large-mnli zero-shot pipeline.
- Classifies the product description against all categories via NLI.
- Once the best-matching category is found, retrieves section names using FAISS like before.

Installation
------------
pip install -r requirements.txt

Requirements:
- pandas
- numpy
- faiss-cpu
- sentence-transformers
- transformers
- torch

Notes
-----
- FAISS indices and section mappings are cached to disk per category.
- Embeddings are normalized to L2 for cosine similarity.
- You can expand this to support multilingual or hierarchical section retrieval.

Ideas for Improvement
---------------------
- Integrate a hybrid approach: Zero-shot for high-level category → FAISS for subcategories.
- Add spell correction or fuzzy matching pre-embedding.
- Cache the zero-shot classifier with ONNX for faster inference.


Author Notes
------------
This project was created while experimenting with different retrieval strategies for dynamic product-section matching in eCommerce or content layout pipelines. It balances speed (FAISS) with semantic reasoning (NLI-based classification).