from rag.embed_nodes import embed_nodes

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

sentence_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
dir_path = "./data/text_data/Articles"
reader = SimpleDirectoryReader(dir_path, recursive=True)
docs = reader.load_data(show_progress=True)
nodes = sentence_splitter.get_nodes_from_documents(docs, show_progress=True)

# count = 0
# for node in nodes:
#     if len(node.text) > 10000:
#         count += 1
# print(count)
nodes = embed_nodes(nodes=nodes, save_dir="./Articles_store/docstore", docstore_name="Articles_with_embeddings_on_chunks", save_docstore=True)
