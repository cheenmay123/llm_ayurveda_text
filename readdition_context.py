# from rag.rag_utils import load_docs

# nodes = load_docs("./storage/Articles_with_summary", return_docstore=False)

# #check how many nodes do not containe metadata['summary']
# count = 0
# for node in nodes:
#     if 'summary' not in node.metadata:
#         count += 1
# print(count)

# #load pickle file
# import pickle

# with open('failed_nodes.pkl', 'rb') as f:
#     nodes = pickle.load(f)
# print(len(nodes))


from rag.rag_utils import load_docs
from rag.context_addition.context_addition import process_nodes_with_id

nodes = load_docs("./storage/Articles_with_summary", return_docstore=False)

node_ids = []

for node in nodes:
    # Check if 'summary' key exists in metadata and if it contains the word "apolog" or word "sorry"
    if 'summary' in node.metadata and ('apolog' in node.metadata['summary'] or 'sorry' in node.metadata['summary']):
        # If it does, add the node._id to the list
        node_ids.append(node.id_)

nodes = process_nodes_with_id(nodes, "Articles_with_summary_revised", node_ids)