from pathlib import Path

from llama_index.core.storage.docstore import SimpleDocumentStore

def save_to_docstore(nodes, save_dir: str, store_name: str) -> None:
    """
    Create a document store and save it to the specified directory.

    Args:
        nodes (List[str]): List of nodes to be added to the document store.
        save_dir (str): Directory path where the document store will be saved.
        store_name (str): Name of the document store file.

    Returns:
        None
    """
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    docstore.add_documents(nodes)
    docstore.persist(persist_path=save_dir / store_name)
    return docstore


def load_docs(path, return_docstore: bool = False):
    """
    Load documents from the specified path.

    Args:
        path (str): The path to the document store.
        return_docstore (bool, optional): Whether to return the document store object instead of the loaded documents. Defaults to False.

    Returns:
        list or SimpleDocumentStore: A list of loaded documents if return_docstore is False, otherwise the document store object.
    """
    docstore = SimpleDocumentStore.from_persist_path(persist_path=path)
    if return_docstore:
        return docstore
    else:
        return list(docstore.docs.values())