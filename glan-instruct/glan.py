######################################################################################################
# [Description] 
# Open Source implementation of GLAN (Generalized Instruction Tuning) using the Azure OpenAI API.
#
# Author: Daekeun Kim (daekeun.kim@microsoft.com)
# Reference: Synthetic Data (Almost) from Scratch: Generalized Instruction Tuning for Language Models
#            https://arxiv.org/pdf/2402.13064
######################################################################################################

import os
import json
import time
import uuid
import random
import openai
import markdown
import textwrap
import jsonlines
from tqdm import tqdm
from bs4 import BeautifulSoup
from datasets import load_dataset
from dotenv import load_dotenv
from openai import AzureOpenAI, RateLimitError

import logging
logger = logging.getLogger('GLAN_logger')
MAX_RETRIES = 3
DELAY_INCREMENT = 30

load_dotenv()  # take environment variables from .env.

client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key        = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version    = os.getenv("AZURE_OPENAI_API_VERSION")
)

def format_timespan(seconds):
    hours = seconds // 3600
    minutes = (seconds - hours*3600) // 60
    remaining_seconds = seconds - hours*3600 - minutes*60
    timespan = f"{hours} hours {minutes} minutes {remaining_seconds:.4f} seconds."
    return timespan


def read_jsonl(filepath):
    data = []
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            data.append(obj)
    return data


def read_text_to_list(file_path):
    data_list = []
    
    with open(file_path, 'r') as file:
        for line in file:
            cleaned_line = line.strip()
            if cleaned_line:
                data_list.append(cleaned_line)
    
    return data_list


def save_list_to_text(file_path, data_list):
    with open(file_path, 'w') as file:
        for item in data_list:
            file.write(f"{item}\n")


def generate_taxonomy(max_number_of_fields=10, model_name="gpt-4o", **kwargs):
    """
    Generate a taxonomy of human knowledge and capabilities.
    """

    prompt = f"""
    Create a taxonomy of human knowledge and capabilities. Break it down into fields, sub-fields, and disciplines.
    Limit the number of fields to a maximum of {max_number_of_fields}.

    Provide the result in JSON format with the following structure:
    {{
        "fields": [
            {{
                "field_name": "Field Name",
                "sub_fields": [
                    {{
                        "sub_field_name": "Sub-field Name",
                        "disciplines": ["Discipline 1", "Discipline 2", ...]
                    }},
                    ...
                ]
            }},
            ...
        ]
    }}

    Examples of `field_name` are Natural Sciences, Humanities or Service.
    Examples of `sub_field_name` are Chemistry, Sociology or Retailing.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_format = {'type': "json_object"},
        **kwargs    
    )
    taxonomy = response.choices[0].message.content
    try:
        taxonomy_json = json.loads(taxonomy)
    except json.JSONDecodeError:
        taxonomy_json = {"error": "Failed to parse JSON"}

    key = next(iter(taxonomy_json))
    disciplines = [discipline for field in taxonomy_json[key] for sub_field in field['sub_fields'] for discipline in sub_field['disciplines']]
    
    return taxonomy_json, disciplines

def validate_subjects_json_structure(data):
    """
    Check if the JSON data has the correct structure for the subjects.
    """
    # Check if the top-level key "subjects" exists and is a list
    if "subjects" not in data or not isinstance(data["subjects"], list):
        return False
    
    # Iterate through each subject to validate its structure
    for subject in data["subjects"]:
        # Check if each subject is a dictionary
        if not isinstance(subject, dict):
            return False
        # Check if required keys exist in each subject and have the correct types
        if "subject" not in subject or not isinstance(subject["subject"], str):
            return False
        if "level" not in subject or not isinstance(subject["level"], int):
            return False
        if "subtopics" not in subject or not isinstance(subject["subtopics"], list):
            return False
        # Check if each item in "subtopics" is a string
        if not all(isinstance(subtopic, str) for subtopic in subject["subtopics"]):
            return False
    
    return True


def generate_subjects(discipline, max_number_of_subjects=2, max_number_of_subtopics=5, model_name="gpt-4o", **kwargs):
    """
    Generate a list of subjects for a given discipline. Please refer to section 2.2 of the paper.
    """

    prompt = f"""
    You are an expert in {discipline}. Create a comprehensive list of subjects a student should learn under this discipline. 
    For each subject, provide the level (e.g., 100, 200, 300, 400, 500, 600, 700, 800, 900) and include key subtopics in JSON format.
    {{    
        "subjects": [
            {{
                'subject': 'Introduction to Computer Science',
                'level': 100,
                'subtopics': [
                    'Basic Programming',
                    'Software Development Fundamentals',
                    'Computer Organization'
                ]
            }}, 
            ...
        ]
    }}
    Limit the number of `subjects` to a maximum of {max_number_of_subjects}.    
    Limit the number of `subtopics` to a maximum of {max_number_of_subtopics} for each `subject`.    
    """

    t0 = time.time()    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_format = {'type': "json_object"},
        **kwargs    
    )
    subjects = response.choices[0].message.content

    subjects_json = json.loads(subjects)
    if not validate_subjects_json_structure(subjects_json):
        logger.info("Failed to parse JSON. Trying again.")
        subjects_json = generate_subjects(discipline, max_number_of_subjects, max_number_of_subtopics, model_name, **kwargs)

    t1 = time.time()
    logger.info(f"Generating subjects took {t1 - t0:.4f} seconds.")
    
    return subjects_json


def generate_syllabus(subject, level, subtopics, max_number_of_session_name=5, model_name="gpt-4o", **kwargs):
    """
    Generate a syllabus for a given subject at a specific level. Please refer to section 2.3 of the paper.
    """
    prompt = f"""
    You are an expert in creating educational syllabi. Create a detailed syllabus for the subject "{subject}" at the {level} level. 
    The syllabus should be broken down into multiple class sessions, each covering different key concepts. 
    The subtopics for this subject include: {subtopics}. Provide the syllabus in JSON format with the following structure in JSON format:

    {{    
        "syllabus": [
            {{
                "session_name": "Session 1 Name",
                "description": "Brief description of the session",
                "key_concepts": ["Key concept 1", "Key concept 2", ...]
            }},
            ...
        ]
    }} 
    Limit the number of `session_name` to a maximum of {max_number_of_session_name}.      
    """
    t0 = time.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_format = {'type': "json_object"},
        **kwargs    
    )

    output = response.choices[0].message.content.strip()
    #logger.info(textwrap.indent(output, '\t'))
    
    try:
        syllabus_json = json.loads(output)
        key = next(iter(syllabus_json))
        syllabus = syllabus_json[key]
    except json.JSONDecodeError:
        logger.error("Failed to parse JSON")
        return None, None

    # Extract class details
    class_sessions = [session['session_name'] for session in syllabus]
    key_concepts = [session['key_concepts'] for session in syllabus]
    t1 = time.time()
    logger.info(f"\tGenerating syllabus took {t1 - t0:.4f} seconds.")

    return class_sessions, key_concepts


def sample_class_sessions_and_key_concepts(class_sessions, key_concepts, single_session=True):
    """
    Sample class sessions and key concepts to generate questions of varying difficulty.

    class_sessions: List of class sessions
    key_concepts: List of key concepts for each session.
    single_session: Whether to sample from a single session or multiple sessions.
    :return: Combination of sampled class sessions and core concepts
    """
    if single_session:
        session_index = random.randint(0, len(class_sessions) - 1)
        selected_session = class_sessions[session_index]
        num_concepts = min(5, len(key_concepts[session_index]))
        selected_key_concepts = random.sample(key_concepts[session_index], k=random.randint(1, num_concepts))
    else:
        if len(class_sessions) < 2:
            raise ValueError("Not enough sessions for multi-session sampling")
        session_indices = random.sample(range(len(class_sessions)), k=2)
        selected_sessions = [class_sessions[i] for i in session_indices]
        combined_key_concepts = key_concepts[session_indices[0]] + key_concepts[session_indices[1]]
        num_concepts = min(5, len(combined_key_concepts))
        selected_key_concepts = random.sample(combined_key_concepts, k=random.randint(2, num_concepts))
    
    return selected_session if single_session else selected_sessions, selected_key_concepts


def generate_questions(
        class_sessions, key_concepts, subject, level, subtopics, model_name="gpt-4o", 
        num_iterations=2, num_questions_per_iteration=5, max_tokens=2048, batch_size=4, language="Korean", **kwargs
    ):
    """
    Generate questions based on class sessions and key concepts using LangChain pipeline. Please refer to section 2.4 of the paper.
    """

    from langchain_openai import AzureChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
    from langchain_core.pydantic_v1 import BaseModel, Field

    llm = AzureChatOpenAI(
        max_tokens=max_tokens,
        openai_api_version="2024-05-01-preview",
        azure_deployment=model_name,
        **kwargs                    
    )

    class CustomOutputParser(BaseOutputParser):
        def parse(self, text: str):
            cleaned_text = text.strip()
            return {"question": cleaned_text}


    prompt = PromptTemplate.from_template(
    """Based on the class session(s) {selected_class_sessions} and key concepts {selected_key_concepts}, generate a homework question.
    A question must be less than {max_tokens} tokens.
    Write in {language}.
    """
    )
    chain = prompt | llm | CustomOutputParser()    

    questions = []
    for idx in range(num_iterations):
        t0 = time.time()
        logger.info(f"\t\t===== Generating Questions: Iteration {idx}")
        selected_class_sessions, selected_key_concepts = sample_class_sessions_and_key_concepts(class_sessions, key_concepts, single_session=True)

        batch_inputs = [{
            "selected_class_sessions": selected_class_sessions,
            "selected_key_concepts": selected_key_concepts,
            "max_tokens": max_tokens,
            "language": language
        } for _ in range(num_questions_per_iteration)]

        metadata = {"subject": subject, "level": level, "subtopics": subtopics}

        with tqdm(total=len(batch_inputs), desc="\t\tProcessing Questions") as pbar:
            for i in range(0, len(batch_inputs), batch_size):
                minibatch = batch_inputs[i:i+batch_size]

                retries = 0
                while retries <= MAX_RETRIES:
                    try:
                        questions_ = chain.batch(minibatch, {"max_concurrency": batch_size})
                        break  # Exit the retry loop once successful
                    except RateLimitError as rate_limit_error:
                        delay = (retries + 1) * DELAY_INCREMENT
                        logger.warning(f"{rate_limit_error}. Retrying in {delay} seconds...")
                        time.sleep(delay)
                        retries += 1

                        if retries > MAX_RETRIES:
                            logger.error(f"Max retries reached this batch. Skipping to next batch.")
                            break
                    except Exception as e:
                        logger.error(f"Error in process_inputs: {e}")
                        break  
                
                for q in questions_:
                    q.update(metadata)
                questions.extend(questions_)
                pbar.set_postfix({"current_batch": f"{i//batch_size + 1}/{(len(batch_inputs) + (batch_size-1))//batch_size}"})

                pbar.update(len(minibatch))
        
        t1 = time.time()
        logger.info(f"\t\tIteration {idx} took {t1 - t0:.4f} seconds.")

    return questions


def generate_answers(all_questions, model_name="gpt-4o", max_tokens=1024, batch_size=5, **kwargs):
    """
    Generate answers to the questions using LangChain pipeline. Please refer to section 2.4 of the paper.
    """
    from langchain.schema.output_parser import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
    from langchain_openai import AzureChatOpenAI

    llm = AzureChatOpenAI(
        temperature=0, 
        max_tokens=max_tokens,
        openai_api_version="2024-05-01-preview",
        azure_deployment=model_name,                   
    )

    system_prompt = """Answer the question. Keep the answer short and concise. The topic, level, and subtopic of this question are as follows.

    ## Subject: {subject}
    ## Level: {level}
    ## Subtopics: {subtopics}

    Respond "DO NOT KNOW" if not sure about the answer.
    """
    system_prompt += f"Answer must be less than {max_tokens} token length."

    system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)
    human_prompt = [
        {
            "type": "text",
            "text": "{question}"
        },
    ]
    human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

    prompt = ChatPromptTemplate.from_messages(
        [
            system_message_template,
            human_message_template
        ]
    )

    chain = prompt | llm | StrOutputParser()
    logger.info(f"===== Generating Answers")
    t0 = time.time()
    all_answers = []
    
    with tqdm(total=len(all_questions), desc="Processing Answers") as pbar:
        for i in range(0, len(all_questions), batch_size):
            minibatch = all_questions[i:i+batch_size]
   
            retries = 0
            while retries <= MAX_RETRIES:
                try:
                    answers = chain.batch(minibatch, {"max_concurrency": batch_size})
                    break  # Exit the retry loop once successful
                except RateLimitError as rate_limit_error:
                    delay = (retries + 1) * DELAY_INCREMENT
                    logger.warning(f"{rate_limit_error}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    retries += 1

                    if retries > MAX_RETRIES:
                        logger.error(f"Max retries reached this batch. Skipping to next batch.")
                        break
                except Exception as e:
                    logger.error(f"Error in process_inputs: {e}")
                    break            
            
            all_answers.extend(answers)
            pbar.set_postfix({"current_batch": f"{i//batch_size + 1}/{(len(all_questions) + (batch_size-1))//batch_size}"})
            pbar.update(len(minibatch))
    t1 = time.time()
    timespan = format_timespan(t1 - t0)
    logger.info(f"Generating Answer dataset took {timespan}")

    return all_answers 


def set_logger(logfile_name="logfile.log"):
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(logfile_name)
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def glan_instruction_generation(args):
    """
    GLAN Pipeline
    """
    GENERATE_DISCIPLINES = args.generate_disciplines
    GENERATE_QUESTION_ONLY = args.generate_question_only
    DISCIPLINES_FILEPATH = args.disciplines_filepath
    LANGUAGE = args.language
    MODEL_NAME = args.model_name
    MODEL_NAME_FOR_ANSWER = args.model_name_for_answer
    MAX_NUMBER_OF_FIELDS = args.max_number_of_fields
    MAX_NUMBER_OF_SUBJECTS = args.max_number_of_subjects
    MAX_NUMBER_OF_SUBTOPICS = args.max_number_of_subtopics
    MAX_NUMBER_OF_SESSION_NAME = args.max_number_of_session_name
    NUM_ITERATIONS = args.num_iterations
    NUM_QUESTIONS_PER_ITERATION = args.num_questions_per_iteration
    QUESTION_MAX_TOKENS = args.question_max_tokens
    QUESTION_BACTH_SIZE = args.question_batch_size
    ANSWER_MAX_TOKENS = args.answer_max_tokens
    ANSWER_BACTH_SIZE = args.answer_batch_size
    OUTPUT_DIR = args.output_dir
    UUID = str(uuid.uuid4())[:4]

    set_logger(args.logfile_name)
    
    logger.info(f"GENERATE_DISCIPLINES = {GENERATE_DISCIPLINES}")
    logger.info(f"GENERATE_QUESTION_ONLY = {GENERATE_QUESTION_ONLY}")    
    logger.info(f"DISCIPLINES_FILEPATH = {DISCIPLINES_FILEPATH}")
    logger.info(f"LANGUAGE = {LANGUAGE}")
    logger.info(f"MODEL_NAME = {MODEL_NAME}")
    logger.info(f"MODEL_NAME_FOR_ANSWER = {MODEL_NAME_FOR_ANSWER}")
    logger.info(f"MAX_NUMBER_OF_FIELDS = {MAX_NUMBER_OF_FIELDS}")
    logger.info(f"MAX_NUMBER_OF_SUBJECTS = {MAX_NUMBER_OF_SUBJECTS}")
    logger.info(f"MAX_NUMBER_OF_SUBTOPICS = {MAX_NUMBER_OF_SUBTOPICS}")
    logger.info(f"MAX_NUMBER_OF_SESSION_NAME = {MAX_NUMBER_OF_SESSION_NAME}")
    logger.info(f"NUM_ITERATIONS = {NUM_ITERATIONS}")
    logger.info(f"NUM_QUESTIONS_PER_ITERATION = {NUM_QUESTIONS_PER_ITERATION}")
    logger.info(f"QUESTION_MAX_TOKENS = {QUESTION_MAX_TOKENS}")
    logger.info(f"QUESTION_BACTH_SIZE = {QUESTION_BACTH_SIZE}")
    logger.info(f"ANSWER_MAX_TOKENS = {ANSWER_MAX_TOKENS}")
    logger.info(f"ANSWER_BACTH_SIZE = {ANSWER_BACTH_SIZE}")
    logger.info(f"OUTPUT_DIR = {OUTPUT_DIR}") 
        
    t0 = time.time()
    all_questions = []

    if GENERATE_DISCIPLINES:
        logger.info(f"===== Generate a Taxonomy of human knowledge and capabilities")
        t0 = time.time()
        taxonomy_json, disciplines = generate_taxonomy(max_number_of_fields=MAX_NUMBER_OF_FIELDS, model_name="gpt-4o", temperature=0.5)
        t1 = time.time()
        logger.info(f"Generating taxonomy took {t1 - t0:.4f} seconds.")
    else:
        logger.info(f"===== Load pre-defined disciplines")
        disciplines = read_text_to_list(DISCIPLINES_FILEPATH)

    for idx1, discipline in enumerate(disciplines):
        logger.info("====================================================================================================")        
        logger.info(f"===== [Discipline {idx1}] Generating Subjects for discipline: {discipline}") 
        logger.info("====================================================================================================")        
        subjects_json = generate_subjects(
            discipline, 
            max_number_of_subjects=MAX_NUMBER_OF_SUBJECTS, 
            max_number_of_subtopics=MAX_NUMBER_OF_SUBTOPICS, 
            model_name=MODEL_NAME, 
            temperature=1.0, 
            top_p=0.95
        )
        
        logger.info(f"Number of subjects is {len(subjects_json['subjects'])}") 
        for idx2, s in enumerate(subjects_json["subjects"]):
            subject = s['subject']
            level = s['level']
            subtopics = ", ".join(s['subtopics'])
            
            logger.info("\t====================================================================================================")        
            logger.info(f"\t===== [Subject {idx2}] Generating Syllabus: Discipline: {discipline} - Subject: {subject} - Level: {level}") 
            logger.info("\t====================================================================================================")        
            class_sessions, key_concepts = generate_syllabus(
                subject, 
                level, 
                subtopics,
                max_number_of_session_name=MAX_NUMBER_OF_SESSION_NAME, 
                model_name=MODEL_NAME, 
                temperature=1.0, 
                top_p=0.95
            )
            logger.info(f"\tNumber of class sessions is {len(class_sessions)}")

            questions = generate_questions(
                class_sessions, 
                key_concepts, 
                subject, 
                level, 
                subtopics,
                model_name=MODEL_NAME, 
                num_iterations=NUM_ITERATIONS,
                num_questions_per_iteration=NUM_QUESTIONS_PER_ITERATION, 
                max_tokens=QUESTION_MAX_TOKENS, 
                batch_size=QUESTION_BACTH_SIZE,
                language=LANGUAGE
            )
            # logger.info(f"\t===== Waiting for 30 seconds to avoid rate limit error.") 
            # time.sleep(30)
            all_questions.extend(questions)

    t1 = time.time()
    timespan = format_timespan(t1 - t0)
    logger.info(f"Generating Question dataset took {timespan}")

    num_questions = len(all_questions)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{OUTPUT_DIR}/GLAN_Questions_{LANGUAGE}_{num_questions}_Samples_{UUID}.jsonl"

    with jsonlines.open(filename, mode='w') as writer:
        for question in all_questions:
            writer.write(question)

    if not GENERATE_QUESTION_ONLY:
        all_answers = generate_answers(
            all_questions, 
            model_name=MODEL_NAME_FOR_ANSWER, 
            max_tokens=ANSWER_MAX_TOKENS, 
            batch_size=ANSWER_BACTH_SIZE
        )

        instructions = []
        for q, a in zip(all_questions, all_answers):
            if a not in "DO NOT KNOW":
                q.update({"answer": a})
                instructions.append(q)

        num_instructions = len(instructions)
        new_filename = filename.replace("Questions", "Instructions")

        with jsonlines.open(new_filename, mode='w') as writer:
            for instruction in instructions:
                writer.write(instruction)