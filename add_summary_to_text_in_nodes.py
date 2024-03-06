from rag.context_addition.context_addition import combine_summary_text

from rag.rag_utils import load_docs, save_to_docstore

nodes = load_docs('./Articles_store/docstore/Articles_with_embeddings_with_summary')

save_to_docstore(nodes, "Articles_store/docstore", "Articles_with_embeddings_with_summary_added_to_text")