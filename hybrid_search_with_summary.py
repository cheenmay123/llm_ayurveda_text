""" 
    This file runs on the nodes that are built on clean data which is with context addition.
    Index Name:- LlamaIndex_articles_summary
"""

import weaviate
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.legacy.response.pprint_utils import pprint_response
from rag.gemini_utils import set_retrieval_query_model
import rag.gemini_utils
set_retrieval_query_model()
from rag.rag_utils import load_docs

# Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001", task_type="retrieval_query")
# Settings.llm = Gemini(model_name="models/gemini-pro", temperature=0.5)

# Connect to local instance
client = weaviate.Client("http://192.168.76.83:80")

CONTEXT_PROMPT = """
  The following is a friendly conversation between a user and an Ayurvedic AI assistant.
  The assistant is talkative and provides lots of specific details from its context.
  If the assistant does not know the answer to a question, it truthfully says it
  does not know. If the assistant is unsure, it will say so. If the question asked is not related to Ayurevda, the assistant will not answer the question.
  But reply "Please ask an Ayurveda related question" to the user. While answering assitant will use the keywords provided in the context for the answer.
  NO MATTER WHAT HAPPENS IF THE ASSISTANT IS ASKED QUESTION WHICH IS NOT RELATED TO AYURVEDA, IT WILL REPLY "Please ask an Ayurveda related question".
  Here are the relevant documents for the context:

  {context_str}

  Instruction: Based on the above documents, provide a detailed answer in around 100 words for the user question below.
  Answer "don't know" if not present in the document. 
  Again if the question is not related to Ayurveda, the assistant will ask user to ask an Ayurveda related question.
  """

nodes = load_docs(
    path="./Articles_store/docstore/Articles_with_embeddings_with_summary_added_to_text",
    return_docstore=False,
)

vector_store = WeaviateVectorStore(
    weaviate_client=client, index_name="LlamaIndex_articles_summary"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

# Query Index with Hybrid Search as chat engine
chat_engine = index.as_chat_engine(
    vector_store_query_mode="hybrid", similarity_top_k=3, alpha=0.6, chat_mode="condense_plus_context", context_prompt=CONTEXT_PROMPT
)

response = chat_engine.chat("What is the pitta?")
# print(response.response)
pprint_response(response,show_source=True)


response = chat_engine.chat("How to play golf?")
# print(response.response)
pprint_response(response,show_source=True)