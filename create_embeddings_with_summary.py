from rag.embed_nodes import embed_nodes_on_summary
from rag.rag_utils import load_docs


nodes = load_docs("./Articles_store/docstore/Articles_with_summary_revised")


nodes = nodes = embed_nodes_on_summary(nodes=nodes, save_dir="./Articles_store/docstore", docstore_name="Articles_with_embeddings_with_summary", save_docstore=True)