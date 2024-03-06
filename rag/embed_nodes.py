from typing import List

from llama_index.core.schema import TextNode
from tqdm import tqdm

from rag.rag_utils import save_to_docstore
from rag.discord_utils import send_msg
import rag.gemini_utils
from rag.gemini_utils import set_retrieval_document_model
from llama_index.core import Settings


set_retrieval_document_model()

def embed_nodes_on_summary(
    nodes: List[TextNode], save_dir: str = "./storage/docstore", docstore_name: str = None, save_docstore: bool = False 
):
    for node in tqdm(nodes, desc="Processing nodes", total=len(nodes), unit="node"):
        node.embedding = Settings.embed_model.get_text_embedding(node.metadata["summary"])
    if save_docstore is True and docstore_name is None:
        raise ValueError("docstore_name is not provided")
    # save nodes to docstore
    if save_docstore is True:
        save_to_docstore(nodes, save_dir, docstore_name)
    send_msg("Nodes embedded successfully!")
    return nodes

def embed_nodes(
    nodes: List[TextNode], save_dir: str = "./storage/docstore", docstore_name: str = None, save_docstore: bool = False 
):
    for node in tqdm(nodes, desc="Processing nodes", total=len(nodes), unit="node"):
        node.embedding = Settings.embed_model.get_text_embedding(node.text)
    if save_docstore is True and docstore_name is None:
        raise ValueError("docstore_name is not provided")
    # save nodes to docstore
    if save_docstore is True:
        save_to_docstore(nodes, save_dir, docstore_name)
    send_msg("Nodes embedded successfully!")
    return nodes