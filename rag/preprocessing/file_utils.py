import os
import glob
from pathlib import Path


def delete_empty_dirs(path: str) -> None:
    """
    Deletes empty directories recursively starting from the given path.

    Args:
        path (str): The path to start deleting empty directories from.

    Returns:
        None
    """
    for dirpath, dirnames, files in os.walk(path, topdown=False):
        if not dirnames and not files:
            os.rmdir(dirpath)


def delete_files_except_html(path: str) -> None:
    """
    Delete all files in the given directory, except for HTML files.

    Args:
        directory (str): The directory path.

    Returns:
        None
    """
    # Use glob to match .htm and .html files
    html_files = glob.glob(path + "/**/*.htm*", recursive=True)

    # Iterate over all files in the directory
    for filename in glob.glob(path + "/**", recursive=True):
        if os.path.isfile(filename) and filename not in html_files:
            os.remove(filename)  # Remove the file


def get_all_paths_from_dir(dir: str) -> list:
    """
    Get all file paths in a directory.

    Args:
        dir (str): The directory path.

    Returns:
        list: A list of file paths.
    """
    return [f for f in glob.glob(dir + "/**", recursive=True) if os.path.isfile(f)]


def replace_directory_in_path(filepath: str, old_dir: str, new_dir: str) -> str:
    """
    Replace a specific directory in a file path and create the new directory if it doesn't exist.

    Args:
        filepath (str): The original file path.
        old_dir (str): The old directory to be replaced.
        new_dir (str): The new directory to replace the old one.

    Returns:
        str: The new file path.
    """
    # Convert the filepath to a Path object
    path = Path(filepath)

    # Convert the old directory to a Path object
    old_dir_path = Path(old_dir)

    # Replace the old directory with the new directory
    new_path = Path(str(path).replace(str(old_dir_path), new_dir))

    return str(new_path)

def save_file_with_text(text, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open (filepath, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    paths = get_all_paths_from_dir("./data/raw_data/Articles")
    delete_files_except_html("./data/raw_data/Articles")
    delete_empty_dirs("./data/raw_data/Articles")
