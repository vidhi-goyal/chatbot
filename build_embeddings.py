# build_embeddings.py

import os
from rag_utils import build_vector_store
# Path to data folder containing all PDFs
# pdf_folder = r"C:\Users\prasukh.jain\Desktop\personalchat\backend\data"

# # Gather all PDF file paths
# pdf_files = [
#     os.path.join(pdf_folder, f)
#     for f in os.listdir(pdf_folder)
#     if f.lower().endswith(".pdf")
# ]
 
# # Build a single vector DB from all PDFs
# build_vector_store(pdf_files, "policy_vector_db")
# print("✅ Vector DB created from all PDFs.")


build_vector_store(["faq1.pdf"], "small_vector_db")
print("✅ Small vector DB created.")