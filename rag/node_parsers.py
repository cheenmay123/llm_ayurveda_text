from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import HTMLNodeParser
from typing import List
import llama_index.core.schema as schema


def extract_htmltag_nodes(
    input_dir: str, tag_list: list = ["p", "li", "b", "i", "u", "section", "text"]
) -> List[schema.TextNode]:
    """
    Extracts nodes for ["p", "li", "b", "i", "u", "section", "text"] tags \
        from HTML files in the specified directory.

    Args:
        input_dir (str): The directory path containing HTML files.

    Returns:
        List[schema.TextNode]: A list of HTML tag nodes.

    Example:
        >>> nodes = extract_htmltag_nodes('/path/to/html_files')
    """
    reader = SimpleDirectoryReader(
        input_dir=input_dir,
        recursive=True,
    )
    documents = reader.load_data(show_progress=True)
    node_parser = HTMLNodeParser(tags=tag_list, include_prev_next_rel=False)
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
    nodes = [node for node in nodes if len(node.get_content()) > 0]
    for node in nodes:
        node.text = node.text.replace("\n", " ").replace("\t", " ")
        node.excluded_embed_metadata_keys = [
            "tag",
            "file_path",
            "file_name",
            "file_type",
            "file_size",
            "creation_date",
            "last_modified_date",
            "last_accessed_date",
        ]
    return nodes
