import os
import shutil
import json
import random

def get_language_code(language_name):
    languages = {
        "English": "en",
        "Korean": "ko",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Chinese": "zh",
        "Japanese": "ja",
        "Russian": "ru",
        "Portuguese": "pt",
        "Italian": "it",
        "Arabic": "ar",
        "Hindi": "hi",
        "Bengali": "bn",
        "Punjabi": "pa",
        "Javanese": "jv",
        "Turkish": "tr",
        "Vietnamese": "vi",
        "Persian": "fa",
        "Polish": "pl",
        "Dutch": "nl",
    }
    return languages.get(language_name, "Unknown")

def delete_folder_and_make_folder(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"The folder '{output_dir}' and its contents have been deleted.")

    os.makedirs(output_dir, exist_ok=True)


def convert_to_oai_format(qa_pair, system_prompt_msg="You're an AI assistant that guides a user to the location of your CS center"):
    """
    Convert the QA pair to the jsonl format required by the OpenAI API.

    Args:
        qa_pair: list of dictionaries or list of lists containing the QA pairs
        system_prompt_msg: message to be displayed as the system prompt

    Returns:
        formatted_data: jsonl format data for OpenAI API
    """
    if isinstance(qa_pair, list):
        formatted_data = []
        for qa in qa_pair:

            sample = [{"role": "system", "content": system_prompt_msg}]

            if isinstance(qa, list): # multi-turn
                for qa_ in qa:
                    if isinstance(qa_, dict):
                        user_message = {"role": "user", "content": qa_["QUESTION"]}
                        assistant_message = {"role": "assistant", "content": qa_["ANSWER"]}
                    else:
                        user_message = {"role": "user", "content": qa_[0]}
                        assistant_message = {"role": "assistant", "content": qa_[1]}
                    sample.append(user_message)
                    sample.append(assistant_message)
            else:  # single turn
                if isinstance(qa, dict):                
                    user_message = {"role": "user", "content": qa["QUESTION"]}
                    assistant_message = {"role": "assistant", "content": qa["ANSWER"]}
                else:
                    user_message = {"role": "user", "content": qa[0]}
                    assistant_message = {"role": "assistant", "content": qa[1]} 
                sample.append(user_message)
                sample.append(assistant_message)

            msg = {"messages": sample} 
            formatted_data.append(msg)
        random.shuffle(formatted_data) 
        return formatted_data

    else:
        print("Argument is not a list")
        return None


def save_jsonl(dictionary_data, file_path):
    with open(file_path, 'w', encoding='UTF-8-sig') as f:
        for entry in dictionary_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")