from html_processing import (
    clean_html,
    convert_ul_to_p,
    delete_empty_tags_regex,
    process_html_table,
    replace_newlines_in_soup,
    replace_spaces_in_soup,
)
import re

from tqdm import tqdm

from file_utils import get_all_paths_from_dir, save_file_with_text, replace_directory_in_path

def create_text_files_from_html(dir_path):
    files = get_all_paths_from_dir(dir_path)
    for file in tqdm(files, total=len(files), desc="Processing files"):
        try:
            soup = clean_html(file)
            soup = replace_newlines_in_soup(soup)
            soup = convert_ul_to_p(soup)
            soup = delete_empty_tags_regex(soup)
            soup = replace_spaces_in_soup(soup)
            soup = process_html_table(soup, "csv")
            soup = replace_spaces_in_soup(soup)
            text = soup.get_text()
            save_file_with_text(text, re.sub(r'\.htm[l]?', '.txt', replace_directory_in_path(file, "raw_data", "text_data")))
        except Exception as e:
            print(f"An error occurred while processing {file}: {str(e)}")

if __name__ == "__main__":
    create_text_files_from_html("./data/raw_data/Articles")
    create_text_files_from_html("./data/raw_data/Therapeutic_Procedures/Procedure_Files")