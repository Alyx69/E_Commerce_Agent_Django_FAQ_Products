import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings


# -----------------------
# Config
# -----------------------
INDEX_DIR = "products_faiss_index"
TOP_K = 5

# -----------------------
# Embeddings
# -----------------------
def load_embeddings():
    return OllamaEmbeddings(
        model="embeddinggemma:300m",
        base_url="http://localhost:11434",
    )

# Initialize embeddings BEFORE using them
embeddings = load_embeddings()

# -----------------------
# Load Vector Store
# -----------------------
def load_vectorstore():
    if not os.path.exists(INDEX_DIR):
        raise FileNotFoundError(f"Vector store directory '{INDEX_DIR}' not found. Please ensure the index has been created first.")
    
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore

try:
    vectorstore = load_vectorstore()
except Exception as e:
    print(f"Failed to load vector store: {e}")
    sys.exit(1)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K},
)

# -----------------------
# FAISS statistics helper
# -----------------------
def get_faiss_stats(vectorstore):
    index = vectorstore.index
    stats = {
        "Documents ": index.ntotal,
        "Embedding dimension ": index.d,
        "Index type ": type(index).__name__,
        "Top-K ": TOP_K,
        "Index path ": os.path.abspath(INDEX_DIR),
    }

    # Disk size
    try:
        size = sum(
            os.path.getsize(os.path.join(INDEX_DIR, f))
            for f in os.listdir(INDEX_DIR)
        )
        stats["Index size (MB) "] = f"{size / 1024 / 1024:.2f} "
    except Exception:
        stats["Index size (MB) "] = "N/A "

    return stats

faiss_stats = get_faiss_stats(vectorstore)

# -----------------------
# Tkinter GUI
# -----------------------
root = tk.Tk()
root.title("FAISS Vector Search (LangChain)")
root.geometry("820x560")

# =======================
# Statistics Frame
# =======================
stats_frame = ttk.LabelFrame(root, text="FAISS Index Statistics", padding=10)
stats_frame.pack(fill="x", padx=10, pady=(10, 0))

for i, (key, value) in enumerate(faiss_stats.items()):
    ttk.Label(stats_frame, text=f"{key}:", width=22).grid(row=i, column=0, sticky="w")
    ttk.Label(stats_frame, text=str(value)).grid(row=i, column=1, sticky="w")

# =======================
# Search Frame
# =======================
search_frame = ttk.Frame(root, padding=10)
search_frame.pack(fill="x")

query_var = tk.StringVar()
query_entry = ttk.Entry(search_frame, textvariable=query_var, width=55)
query_entry.pack(side="left", padx=(0, 10))

# -----------------------
# Search handler
# -----------------------
def on_search():
    query = query_var.get().strip()
    if not query:
        return

    try:
        results = retriever.invoke(query)
    except Exception as e:
        messagebox.showerror("Search error", str(e))
        return

    results_box.config(state="normal")
    results_box.delete("1.0", tk.END)

    for i, doc in enumerate(results, start=1):
        results_box.insert(
            tk.END,
            f"#{i}\n{doc.page_content}\n\n"
        )

    results_box.config(state="disabled")

search_button = ttk.Button(search_frame, text="Search", command=on_search)
search_button.pack(side="left")

# =======================
# Results Frame
# =======================
results_frame = ttk.Frame(root, padding=10)
results_frame.pack(fill="both", expand=True)

scrollbar = ttk.Scrollbar(results_frame)
scrollbar.pack(side="right", fill="y")

results_box = tk.Text(
    results_frame,
    wrap="word",
    yscrollcommand=scrollbar.set,
    state="disabled",
    font=("Segoe UI", 10),
)
results_box.pack(fill="both", expand=True)
scrollbar.config(command=results_box.yview)

# Enter key triggers search
root.bind("<Return>", lambda event: on_search())

root.mainloop()