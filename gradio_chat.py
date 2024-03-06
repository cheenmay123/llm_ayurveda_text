import gradio as gr
import weaviate
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.legacy.response.pprint_utils import pprint_response
from rag.gemini_utils import set_retrieval_query_model
import rag.gemini_utils
from rag.rag_utils import load_docs
import warnings
#supress all warnings
warnings.filterwarnings("ignore")
#supress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

set_retrieval_query_model()

CONTEXT_PROMPT = """
  The following is a friendly conversation between a user and an Ayurvedic AI assistant.
  The assistant provides lots of specific details from its context.
  If the assistant does not know the answer to a question, it truthfully says it
  does not know. If the assistant is unsure, it will say so. If the question asked is not related to Ayurevda, the assistant will not answer the question.
  But reply "Please ask an Ayurveda related question" to the user. While answering assitant will use the keywords provided in the context for the answer.
  NO MATTER WHAT HAPPENS IF THE ASSISTANT IS ASKED QUESTION WHICH IS NOT RELATED TO AYURVEDA, IT WILL REPLY "Please ask an Ayurveda related question".
  Here are the relevant documents for the context:

  {context_str}

  Instruction: Based on the above documents, provide a detailed answer in around 100 words for the user question below.
  If the information in the context above is not relevant to the question assistant will say "I can't answer this question based on the information available".
  Again if the question is not related to Ayurveda, the assistant will ask user to ask an Ayurveda related question.
  If asked what can you do ? or questions with similar meaning assistant will say "I am an AI assistant that can answer questions related to Ayurveda, using the infomation provided in Articles data."
  If the question is vague tell the user to ask more specific question.
  """
ip_address = ""
client = weaviate.Client("http://192.168.76.83:8080")


vector_store = WeaviateVectorStore(
    weaviate_client=client, index_name="LlamaIndex_articles_summary"
)

index:VectorStoreIndex = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# Query Index with Hybrid Search as chat engine
chat_engine = index.as_chat_engine(
    vector_store_query_mode="hybrid", similarity_top_k=2, alpha=0.75, chat_mode="condense_plus_context", context_prompt=CONTEXT_PROMPT, verbose=True
)




def respond(message, history):
    # streaming_response = chat_engine.stream_chat(message)
    # for token in streaming_response.response_gen:
    #     print(token, end="")
    #     yield token
    response = chat_engine.chat(message)
    return response.response


def reset():
    chat_engine.reset()
    gr.Info("Chat history has been reset.")

with gr.Blocks() as demo:
    reset_button = gr.Button("Click to reset chat history")
    reset_button.click(reset)
    chat = gr.ChatInterface(respond, undo_btn=None, fill_height=True)

demo.launch()