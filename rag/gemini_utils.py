from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
from dotenv import load_dotenv
load_dotenv("./.env")

def set_retrieval_document_model():
    Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001", task_type="retrieval_document")

def set_retrieval_query_model():
    Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001", task_type="retrieval_query")
    
Settings.llm = Gemini(model_name="models/gemini-pro", temperature=0.5)
