import os
import pickle
import textwrap
import time
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv
from google.generativeai import GenerationConfig
from llama_index.core.schema import TextNode
from llama_index.embeddings.gemini import GeminiEmbedding
from tqdm import tqdm
#TODO: Implement discord send message
from rag.discord_utils import send_msg
from rag.rag_utils import save_to_docstore


def make_prompt(relevant_passage):
    """Creates prompt for summarising the passage provided. 

    Args:
        relevant_passage (string): passage to be provided for summarising.

    Returns:
        prompt: _description_
    """ 
    load_dotenv("./.env") 

    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))  

    # model = genai.GenerativeModel('gemini-pro')

    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = textwrap.dedent("""You are an informative person well versed in Ayurveda medicine and the content provided in the passage. \ \
    There are a lot of complicated ayurveda terminlogies used in the reference passage provided in format PASSAGE:"Actual passage here". \
    Be sure to respond in a complete sentence, being comprehensive, including all relevant information. \
    Make sure to respond in plain text only DO NOT USE MARKDOWN and use simple english language without using any complicated words. \
    While answering be sure to break down complicated concepts and \
    strike a friendly and converstional tone while being to the point with the queries. \
    For technical jargons and make sure to provide meaning in parantheses. \
    You may also ignore any csv data provided. \
    The answer to the query is mostly present in the reference passage provided. Go through the passage thoroughly and fetch relevant information. \
    Strictly limit your responses to 300 words. \
    Summarise the information provided in the reference passage comprehensively. \
    IF YOU CAN'T SUMMARISE THE PASSAGE, TRY TO PROVIDE A BRIEF DESCRIPTION OF THE PASSAGE. \
    PASSAGE: '{relevant_passage}'

        ANSWER:
    """).format(relevant_passage=escaped)

    return prompt


def process_nodes(nodes, store_name):
    # nodes = extract_htmltag_nodes('../../data/clean_html/Articles', tag_list=["p","section"])
    config = GenerationConfig(
    # max_output_tokens = 600,
    temperature= 0.6,
    top_p = 1,
    top_k = 1 ,
    )
    safety_settings = [{'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'}, {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'}]

    model = genai.GenerativeModel('gemini-pro', generation_config=config, safety_settings=safety_settings)

    total_nodes = len(nodes)
    processed_nodes = 0
    failed_nodes = []
    try:

        for node in tqdm(nodes, desc="Processing nodes", unit="node", total=total_nodes):
            prompt = make_prompt(node.text)
            try:
                answer = model.generate_content(prompt)
            except Exception as e:
                node.metadata['summary'] = "NO ANSWER"
                failed_nodes.append(node.id_)
                print(answer.candidates)

            node.metadata['summary'] = answer.text
            processed_nodes += 1
            if processed_nodes % 100 == 0:
                send_msg("Processed " + str(processed_nodes) + " out of " + str(total_nodes) + " nodes")
            time.sleep(3)
    except Exception as e:
        with open("gemini_response.pkl", "wb") as pickle_file:
            pickle.dump(answer, pickle_file)

#save all failed nodes list to pickle
    with open("failed_nodes.pkl", "wb") as pickle_file:
        pickle.dump(failed_nodes, pickle_file)
    print(failed_nodes)
    send_msg("Summary addition complete, Failed nodes: " + str(len(failed_nodes)) + " out of " + str(total_nodes) + " nodes")
    save_to_docstore(nodes, save_dir='./storage', store_name=store_name)



def process_nodes_with_id(nodes, store_name,node_ids):
    # nodes = extract_htmltag_nodes('../../data/clean_html/Articles', tag_list=["p","section"])
    config = GenerationConfig(
    # max_output_tokens = 600,
    temperature= 0.6,
    top_p = 1,
    top_k = 1 ,
    )
    safety_settings = [{'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'}, {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'}]

    model = genai.GenerativeModel('gemini-pro', generation_config=config, safety_settings=safety_settings)

    total_nodes = len(nodes)
    processed_nodes = 0
    failed_nodes = []
    try:

        for node in tqdm(nodes, desc="Processing nodes", unit="node", total=total_nodes):
            if node.id_ in node_ids:
                prompt = make_prompt(node.text)
                try:
                    answer = model.generate_content(prompt)
                except Exception as e:
                    node.metadata['summary'] = "NO ANSWER"
                    failed_nodes.append(node.id_)
                    print(answer.candidates)

                node.metadata['summary'] = answer.text
                processed_nodes += 1
                time.sleep(3)
                
    except Exception as e:
        with open("gemini_response.pkl", "wb") as pickle_file:
            pickle.dump(answer, pickle_file)

#save all failed nodes list to pickle
    with open("failed_nodes.pkl", "wb") as pickle_file:
        pickle.dump(failed_nodes, pickle_file)
    send_msg("Summary addition complete, Failed nodes: " + str(len(failed_nodes)) + " out of " + str(total_nodes) + " nodes")
    save_to_docstore(nodes, save_dir='./storage', store_name=store_name)

def combine_summary_text(nodes: List[TextNode]):
    for node in nodes:
        node.text = node.metadata["summary"] + " " + node.text
    return nodes

if __name__ == "__main__":
    send_msg("Summary addition started")