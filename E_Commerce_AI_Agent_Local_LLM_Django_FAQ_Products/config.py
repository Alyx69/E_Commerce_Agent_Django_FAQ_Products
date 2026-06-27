## LLM Configurations
LLM_MODEL = "llama3.1:8b"
LLM_CONTEXT_WINDOW = 4096
LLM_TEMPERATURE = 0

## System Prompt Configuration
SYSTEM_PROMPT_FILE = "E_Commerce_Agent_System_Prompt.txt"

## FAISS Vector Database Configuration
PRODUCTS_INDEX_DIR = "products_faiss_index"
FAQ_INDEX_DIR = "faq_faiss_index"
EMBEDDING_MODEL = "embeddinggemma:300m"

## Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"