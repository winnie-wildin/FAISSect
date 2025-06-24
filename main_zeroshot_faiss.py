import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import re
import os, pickle
def sanitize_filename(s):
    # Replace all non-word characters (everything except a-z, A-Z, 0-9 and _) with _
    return re.sub(r'[^\w\-]', '_', s)
class SectionRetriever:
    def __init__(self, df):
        self.df = df
        self.df['Category 1'] = self.df['Category 1'].str.lower().str.strip()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.category_indices = {}
        self.category_section_map = {}
        self._load_or_build_indices() 
        self._build_category_index()

    def _build_category_index(self):
        unique_categories = self.df['Category 1'].unique().tolist()
        self.category_list = unique_categories
        embeddings = self.model.encode(unique_categories, convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(embeddings)
        self.category_faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.category_faiss_index.add(embeddings)


    def _load_or_build_indices(self):
        for category, group in self.df.groupby("Category 1"):
            safe_cat = sanitize_filename(category)
            idx_path = f'{safe_cat}_faiss.index'
            map_path = f'{safe_cat}_section_map.pkl'
            if os.path.exists(idx_path) and os.path.exists(map_path):
                print(f'Loading index for {category}')
                self.category_indices[category] = faiss.read_index(idx_path)
                with open(map_path, 'rb') as f:
                    self.category_section_map[category] = pickle.load(f)
            else:
                print(f'Creating index for {category}')
                sections = group["Section Name"].dropna().unique().tolist()
                embeddings = self.model.encode(sections, convert_to_numpy=True).astype('float32')
                faiss.normalize_L2(embeddings)
                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings)
                self.category_indices[category] = index
                self.category_section_map[category] = sections
                faiss.write_index(index, idx_path)
                with open(map_path, 'wb') as f:
                    pickle.dump(sections, f)
    def infer_category(self, product_info):
        candidate_labels = self.df['Category 1'].unique().tolist()
        result = self.zero_shot(product_info, candidate_labels)
        best = result['labels'][0]
        print(f"\n🔎 Zero-Shot Inferred Category: '{best}' (Score: {result['scores'][0]:.4f})")
        return best




    def get_top_sections(self, product_info, category, top_k=5):
        category = category.lower().strip()  # Normalize lookup
        print(f"\n--- Query for category '{category}' ---")
        if category not in self.category_indices:
            return []
        
        print(f"Product info: {product_info[:60]}...")  # Print beginning of product info
         # Get the candidate section names and their embeddings from saved map
        sections = self.category_section_map[category]
        print(f"Section names for this category ({len(sections)}): {sections}")


        query_embedding = self.model.encode([product_info], convert_to_numpy=True).astype('float32')
        print(f"Query embedding shape: {query_embedding.shape}")
        print(f"Query embedding (first 10 dims): {query_embedding[0][:10]}")

        faiss.normalize_L2(query_embedding)
        print("Query embedding L2 norm (should be 1):", np.linalg.norm(query_embedding))
        # print("Query embedding shape:", query_embedding.shape)
        # print("Query embedding norm:", np.linalg.norm(query_embedding))

        D, I = self.category_indices[category].search(query_embedding, top_k)
        print("\nFAISS search results:")
        for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
            section_name = sections[idx]
            print(f"Rank {rank}: {section_name:<35} Score: {score:.4f}   [Index: {idx}]")
        print("Similarity scores:", D)
        print("Indices:", I)


        return [self.category_section_map[category][i] for i in I[0]]
# Load your DataFrame
df = pd.read_csv("Section_List.csv")  # or from Google Sheet

# Initialize retriever
retriever = SectionRetriever(df)

# Query
product = """Bacca Bucci Men Lace Up Sneaker Shoes
Street Style Design : The Fashion Sneakers Upper And Sole Are Design With Lightweight And Eliminate The Excess Lining Material, Using The Pu To Overlay With Micro Fiber Pieces, The Piercing Dots As Embellishments, Solid Color For Visual Impact, Showing The Trend Of Street Style.

"""



category = retriever.infer_category(product)

top_sections = retriever.get_top_sections(product, category)
print("Category:", category)
print("Top Sections:", top_sections)
