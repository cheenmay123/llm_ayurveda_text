import io
import re
from pathlib import Path

import chardet
import pandas as pd
from bs4 import BeautifulSoup


def detect_encoding(filename: str) -> str:
    """
    Detects the encoding of a file.

    Args:
        filename (str): The path to the file.

    Returns:
        str: The encoding of the file.
    """
    with open(filename, "rb") as f:
        result = chardet.detect(f.read())
    return result["encoding"]


def make_soup(filepath: str) -> BeautifulSoup:
    """
    Create a BeautifulSoup object from an HTML file.

    Args:
        filepath (str): The path to the HTML file.

    Returns:
        BeautifulSoup: The BeautifulSoup object representing the HTML file.
    """
    encoding = detect_encoding(filename=filepath)
    with open(filepath, "r", encoding=encoding) as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    return soup


def save_soup(soup: BeautifulSoup, filepath: str):
    """
    Save a BeautifulSoup object to an HTML file.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object to save.
        filepath (str): The path to the HTML file.
    """
    filepath = Path(filepath)
    # Create the new directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(soup.prettify())


def delete_tags_with_surekh(file_path: str):
    soup = make_soup(file_path)
    # Find all <span> tags with 'font-family:SD01-TTSurekh'
    try:
        spans = soup.find_all(
            "span", style=lambda value: "font-family:SD01-TTSurekh" in value
        )
        if spans:
            for span in spans:
                span.decompose()
    except Exception as e:
        pass

    try:
        for element in soup.select(".MsoPlainText"):
            element.decompose()
    except Exception as e:
        pass
    return soup


def clean_html(file_path: str):
    soup = delete_tags_with_surekh(file_path)
    _tags_to_decopose = ["script", "style", "img", "nobr", "meta", "link", "title"]

    _tags_to_unwrap = ["font", "span", "o:p", "i", "b", "u", "h1", "h2", "h3"]

    _attributes_to_remove = [
        "class",
        "style",
        "bgcolor",
        "lang",
        "onclick",
        "onload",
        "align",
        "font",
        "xmlns",
        "xmlns:o",
        "xmlns:v",
        "xmlns:w",
        "link",
        "id",
        "vlink",
        "border",
        "bordercolordark",
        "bordercolorlight",
        "cellpadding",
        "cellspacing",
        "size",
        "font-size",
        "width"
    ]

    tags_to_decompose = soup(_tags_to_decopose)
    tags_to_unwrap = soup(_tags_to_unwrap)

    if tags_to_decompose:
        for tag in soup(_tags_to_decopose):
            tag.decompose()
    if tags_to_unwrap:
        for _tag in soup(_tags_to_unwrap):
            _tag.unwrap()

    for tag in soup():
        for attribute in _attributes_to_remove:
            del tag[attribute]
    # TODO: Implement deletion of empty tags here

    raw_html = str(soup)
    raw_html = re.sub(r"\xa0|†|`|&amp;", "", raw_html) #just add whatever characters need replacing seperated by |
    cleaned_html = re.sub(r"<\?if.*?<\?endif\?>", "", raw_html, flags=re.DOTALL)
    cleaned_html = re.sub(r"<!--.*?-->", "", cleaned_html, flags=re.DOTALL)

    return BeautifulSoup(cleaned_html, "html.parser")


def process_html_table(soup, table_parser: str) -> BeautifulSoup:
    """
    Process the HTML table in the given file path and convert it to the specified format.

    Args:
        file_path (str): The path to the HTML file.
        table_parser (str): The format to convert the table to. Valid options are 'md' for Markdown and 'tsv' for TSV.

    Returns:
        BeautifulSoup: The modified BeautifulSoup object with the converted table.

    """
    tables = soup.find_all("table")
    for table in tables:
        try:
            df = pd.read_html(io.StringIO(str(table)), header=0, index_col=0)
            md = df[0].to_markdown(tablefmt="grid")
            tsv = df[0].to_csv(sep="\t")
            csv = df[0].to_csv()
            # convert table tag to text tag
            table.name = "text"
            if table_parser == "md":
                table.string = "\n<table_start>\n"+md+"<table_end>\n"
            elif table_parser == "tsv":
                table.string = "\n<tsv_start>\n"+tsv+"<tsv_end>\n"
            elif table_parser == "csv":
                table.string = "\n<csv_start>\n"+csv+"<csv_end>\n"
            elif table_parser == "blank":
                table.string = ""
        except Exception as e:
            table.decompose()  # Delete the table tag
    return soup


# def delete_empty_tags(soup: BeautifulSoup) -> BeautifulSoup:
#     """
#     Delete all empty tags in the given BeautifulSoup object.

#     Args:
#         soup (BeautifulSoup): The BeautifulSoup object to modify.

#     Returns:
#         BeautifulSoup: The modified BeautifulSoup object with the empty tags removed.

#     """
#     for tag in soup.find_all():
#         if not tag.text.strip():
#             tag.decompose()
#     return soup


def delete_empty_tags_regex(soup: BeautifulSoup) -> BeautifulSoup:
    html_str = str(soup)
    clean_html = re.sub(r"<[^/>]*>\s*</[^>]*>", "", html_str)
    soup = BeautifulSoup(clean_html, "html.parser")
    return soup



# def replace_newlines_in_leaves(soup: BeautifulSoup) -> BeautifulSoup:
#     for tag in soup.find_all():
#         if len(tag.find_all()) == 0:  # Check if tag is a leaf
#             if tag.name != "text":  # Don't replace in <text> tags
#                 if tag.string:  # Check if tag has a string
#                     tag.string.replace_with(re.sub(r"\n\s{2,}", "", tag.string))
#     return soup


def replace_newlines_in_soup(soup: BeautifulSoup) -> BeautifulSoup:
    soup_str = str(soup)
    cleaned_soup_str = re.sub(r"\n\s*", " ", soup_str)
    return BeautifulSoup(cleaned_soup_str, "html.parser")


def replace_spaces_in_soup(soup: BeautifulSoup) -> BeautifulSoup:
    soup_str = str(soup)
    cleaned_soup_str = re.sub(r" {2,}", " ", soup_str)
    return BeautifulSoup(cleaned_soup_str, "html.parser")


def convert_ul_to_p(soup: BeautifulSoup) -> BeautifulSoup:
    for ul_tag in soup.find_all("ul"):
        p_tag = soup.new_tag("p")
        p_tag.string = ul_tag.text
        ul_tag.replace_with(p_tag)
    return soup

