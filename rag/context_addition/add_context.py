from context_addition import process_nodes,process_nodes_with_id
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

sentence_splitter = SentenceSplitter(chunk_overlap=200, chunk_size=1024)
dir_path = "./data/text_data/Articles"
reader = SimpleDirectoryReader(dir_path, recursive=True)
docs = reader.load_data(show_progress=True)
nodes = sentence_splitter.get_nodes_from_documents(docs, show_progress=True)

if __name__ =="__main__":
    process_nodes(nodes, "Articles_with_summary_v2")