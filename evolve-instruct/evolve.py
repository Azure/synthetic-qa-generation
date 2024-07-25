import ast
import json
import uuid
from typing import List

import pandas as pd
import numpy as np
from enum import Enum

import time
import torch
from datasets import Dataset, DatasetDict
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

import markdown
from bs4 import BeautifulSoup
from datasets import load_dataset
import os, openai
from dotenv import load_dotenv
from openai import AzureOpenAI, RateLimitError

load_dotenv()  # take environment variables from .env.
MAX_ITERATIONS = 1
MAX_RETRIES = 2

def md_to_text(md, do_md_to_text=True):
    if not do_md_to_text:
        return md
    assert md is not None, "Markdown is None"
    html = markdown.markdown(md)
    soup = BeautifulSoup(html, features='html.parser')
    return soup.get_text()


class Mutation(Enum):
    FRESH_START = 0
    ADD_CONSTRAINTS = 1
    DEEPEN = 2
    CONCRETIZE = 3
    INCREASE_REASONING = 4
    COMPLICATE = 5
    SWITCH_TOPIC = 6

# Retrieved from https://github.com/nlpxucan/WizardLM/tree/main
base_depth_instruction = "I want you act as a Prompt Rewriter.\r\n \
					Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\r\n \
					But the rewritten prompt must be reasonable and must be understood and responded by humans.\r\n \
					Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#. \r\n \
					You SHOULD complicate the given prompt using the following method: \r\n\
					{} \r\n\
					You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#. \r\n\
					'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\r\n"

base_breadth_instruction = "I want you act as a Prompt Creator.\r\n\
Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.\r\n\
This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.\r\n\
The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.\r\n\
The #Created Prompt# must be reasonable and must be understood and responded by humans.\r\n\
'#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#\r\n"

complicate_prompt = base_depth_instruction.format("#Given Prompt# to make it slightly more complicated.'")
constraints_prompt = base_depth_instruction.format("Please add one more constraints/requirements into #The Given Prompt#'")
deepen_prompt = base_depth_instruction.format("If #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased.")
concretizing_prompt = base_depth_instruction.format("Please replace general concepts with more specific concepts.")
reasoning_prompt = base_depth_instruction.format("If #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.")

class WizardLM:
    def __init__(
            self,
            llm_pipeline: pipeline = None,
            seed_data: List[str] = None,
            column_names: List[str] = ["instruction"],
            num_rows: int = 10,
            min_len_chars: int = 512,
            max_len_chars: int = 1024,
            verbose: bool = False,
            language: str = "Korean",
    ):
        """
        Open-Source Implementation of https://arxiv.org/abs/2304.12244

        :param llm_pipeline: Pipeline that takes a HF dataset containing one string column and returns a list of strings
        :param seed_data: Optional data to create Q:A pairs from, list of strings containing prompts
        :param num_rows: Number of desired Q:A pairs
        :param min_len_bytes: Lower limit for prompt length in bytes
        :param max_len_bytes: Upper limit for prompt length in bytes
        :param verbose: Whether to enable verbose printing.
        """
        self.llm_pipeline = llm_pipeline
        self.column_names = column_names
        self.num_rows = num_rows
        self.verbose = verbose
        self.seed_text_list = []
        self.seed_data = seed_data
        self.prompts = []
        self.final_prompts = []
        self.final_answers = []
        self.min_len_bytes = min_len_chars
        self.max_len_bytes = max_len_chars
        self.prompt_templates = dict()
        self.prompt_templates['base'] = ""
        seed = None
        np.random.seed(seed)
        self.language = language
        self.prompt_templates[Mutation.FRESH_START] = \
            self.prompt_templates['base'] + \
f"""Write one question or request containing one or more of the following words. Write in {self.language}.: <PROMPT>"""

        self.prompt_templates[Mutation.COMPLICATE] = \
            self.prompt_templates['base'] + \
f"""{complicate_prompt}\nWrite in {self.language}.

#Given Prompt#:
<PROMPT>
"""

        self.prompt_templates[Mutation.ADD_CONSTRAINTS] = \
            self.prompt_templates['base'] + \
f"""{constraints_prompt}\nWrite in {self.language}.

#The Given Prompt#:
<PROMPT>
"""

        self.prompt_templates[Mutation.DEEPEN] = \
            self.prompt_templates['base'] + \
f"""{deepen_prompt}\nWrite in {self.language}.

#The Given Prompt#:
<PROMPT>
"""

        self.prompt_templates[Mutation.CONCRETIZE] = \
            self.prompt_templates['base'] + \
f"""{concretizing_prompt}\nWrite in {self.language}.

#The Given Prompt#:
<PROMPT>
"""

        self.prompt_templates[Mutation.INCREASE_REASONING] = \
            self.prompt_templates['base'] + \
f"""{reasoning_prompt}\nWrite in {self.language}.

#The Given Prompt#:
<PROMPT>
"""

        self.prompt_templates[Mutation.SWITCH_TOPIC] = \
            self.prompt_templates['base'] + \
f"""{base_breadth_instruction}\nWrite in {self.language}.

#Given Prompt#:
<PROMPT>
"""
    def run(self):
        self.create_seed_prompts()
        self.create_prompts()
        self.create_answers()

        list_qa = []
        for i in range(len(self.final_prompts)):
            if len(self.final_answers[i]) > 10:
                list_qa.append(
                    {
                        'input': self.final_prompts[i],
                        'output': self.final_answers[i],
                    }
                )
        with open(f"{self.seed_data.replace('.jsonl', '').replace('json', '')}.%s.json" % str(uuid.uuid4())[:4], "wt") as f:
            f.write(json.dumps(list_qa, indent=2, ensure_ascii=False))        

    def run_question_only(self):
        self.create_seed_prompts()
        self.create_prompts()

        list_q = []
        for i in range(len(self.final_prompts)):
            list_q.append(
                {
                    'input': self.final_prompts[i],
                }
            )
        with open(f"{self.seed_data.replace('.jsonl', '').replace('json', '')}.%s.json" % str(uuid.uuid4())[:4], "wt") as f:
            f.write(json.dumps(list_q, indent=2, ensure_ascii=False))

    def create_seed_prompts(self):
        """
        Turn self.seed_data into a list of strings of text self.source_text_list
        Each text string can represent as little as a word, or as much as document.
        Just has to be representative of some concept or body of text.

        :return: None
        """

        import os

        if isinstance(self.seed_data, str) and os.path.exists(self.seed_data):
            data = load_dataset("json", data_files=self.seed_data)
            self.seed_text_list = []
            for d in data['train']:
                s = ""
                if isinstance(self.column_names, str):
                    s = d[self.column_names]
                else:
                    for col in self.column_names:
                        s += d[col] + "\n"
                self.seed_text_list.append(s.strip())
            assert self.seed_text_list, "data import failed, got empty list"

    def create_prompts(self):
        print("Creating %d prompts." % self.num_rows)
        assert self.seed_text_list, "must have seed text list"
        t0 = time.time()
        self.prompts.clear()
        for i in range(self.num_rows):
            new_prompt = np.random.choice(self.seed_text_list)
            self.prompts.append(new_prompt)
        i = 0
        print(f"length of self prompts={len(self.prompts)}")

        while self.mutate(i):
            print("Iteration: %d" % i)
            i += 1
            if i >= MAX_ITERATIONS:
                print("Reached maximum number of iterations.")
                break            
        t1 = time.time()
        print("Done creating %d prompts in %.4f seconds." % (len(self.final_prompts), t1 - t0))

    def create_answers(self):
        print("Creating answers for %d prompts." % len(self.final_prompts))
        t0 = time.time()
        ds = self.convert_list_to_dataset(self.final_prompts)
        self.final_answers = self.llm_pipeline(ds['train'])
        t1 = time.time()
        print("Done creating answers for %d prompts in %.4f seconds." % (ds['train'].num_rows, t1 - t0))

    def convert_list_to_dataset(self, text_list):
        df = pd.DataFrame({'text': text_list})
        ds = DatasetDict()
        ds['train'] = Dataset.from_pandas(df)
        return ds

    def mutate(self, iteration):
        assert len(self.prompts) == self.num_rows
        list_prompts = []
        mutations = []
        for i in range(self.num_rows):
            mutation = np.random.choice(Mutation)
            mutations.append(mutation)
            # if mutation == Mutation.FRESH_START:
            #     mutation = Mutation.COMPLICATE
            before = self.prompts[i]
            prompt = self.prompt_templates[mutation].replace("<PROMPT>", before)
            
            if mutation == Mutation.SWITCH_TOPIC:
                prompt += "#Created Prompt#:\r\n"
            else:
                prompt += "#Rewritten Prompt:\r\n"
            
            print(f"Full prompt={prompt}")
            list_prompts.append(prompt)

        ds = self.convert_list_to_dataset(list_prompts)
        assert ds['train'].num_rows == len(list_prompts) == self.num_rows == len(self.prompts)
        
        # Processing transformed prompts using the LLM pipeline
        t0 = time.time()
        after = self.llm_pipeline(ds['train'])
        assert len(after) == self.num_rows
        t1 = time.time()
        
        llm_pipeline_name = self.llm_pipeline.__class__.__name__
        print(f"{llm_pipeline_name} took {t1 - t0:.4f} seconds")

        for i in range(len(after)):
            after[i] = after[i].split("Prompt#:")[-1].strip()
            for pp in ['New Prompt:\n', 'New Prompt: ']:
                if after[i][:len(pp)] == pp:
                    after[i] = after[i][len(pp):]
            after[i] = after[i].strip()
            
            #use_new_prompt, why = self.change_approved(self.prompts[i], after[i])
            use_new_prompt = True
            if self.verbose:
                print("===========================")
                print("Old Prompt: %s" % self.prompts[i])
                print("Mutation: %s" % mutations[i].name)
                print("New Prompt: %s" % after[i])
                print("===========================")

            if use_new_prompt:
                if self.max_len_bytes >= len(after[i]) >= self.min_len_bytes:
                    self.final_prompts.append(after[i])
                    print("Prompt was accepted, now have %d good prompts." % len(self.final_prompts))
                    self.prompts[i] = np.random.choice(self.seed_text_list)
                    print("Creating new prompt.")
                else:
                    self.prompts[i] = after[i]
                    print("Prompt was successfully modified.")
            else:
                print("Mutation rejected, will try again. Reason: %s" % why)
            print("", flush=True)
        print("final_prompt=")
        print(self.final_prompts)
        return len(self.final_prompts) < self.num_rows

    def change_approved(self, before, after):
        if before == after:
            return False, "same"
        if after.count('\n') > after.count(" ") * 2:
            return False, "too many lines"
        if after.count('\n') == after.count("- ") > 10:
            return False, "too many items"
        if self.prompt_templates['base'] and self.prompt_templates['base'] in after:
            return False, "prompt leaked 1"
        if "#New Prompt#" in after:
            return False, "prompt leaked 2"
        if "new prompt" in after.lower():
            return False, "prompt leaked 3"
        if "how can i assist" in after.lower():
            return False, "AI"
        if "as an ai" in after.lower():
            return False, "AI"
        if "gpt" in after.lower() and "gpt" not in before.lower():
            return False, "AI"        
        if "ai assistant" in after.lower():
            return False, "AI"
        if "i'm sorry" in after.lower() and "sorry" not in before.lower() and len(after) < 400:
            return False, "sorry"
        if False:
            # too slow in general, not needed
            prompt = """Are the two following prompts equal to each other?
To be equal, they must meet two requirements:
1. Both prompts have the same constraints and requirements.
2. Both prompts have the same depth and breath of the inquiry.
First prompt: %s
Second prompt: %s
Answer with 'Equal' or 'Not Equal'. No need to explain the reason.""" % (before, after)
            answer = self.llm_pipeline(prompt)
            if 'not equal' not in answer.lower():
                return False, "equal"
        return True, "ok"


class AzureGPTPipeline:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs 
        self.client = AzureOpenAI(
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key        = os.getenv("AZURE_OPENAI_API_KEY"),
            api_version    = os.getenv("AZURE_OPENAI_API_VERSION")
        )
            
    def __call__(self, dataset, **kwargs):
        ret = []

        gen_count = 0
        for d in dataset:
            response = None
            retries = 0
            while not response and retries < MAX_RETRIES:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": d['text']}],
                        **kwargs     
                    )
                except RateLimitError as e:
                    print("Rate limit exceeded. Retrying in 10 seconds...")
                    retries += 1
                    time.sleep(10)
            if response:
                ret.append(response.choices[0].message.content)
            else:
                ret.append("")
            gen_count += 1
            if gen_count % 10 == 0:
                print(gen_count)
        return ret


class HFPipeline:
    def __init__(self, model, max_new_tokens=None, batch_size=None, **kwargs):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
        print("loading model")
        model_obj = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map="auto")
        pad_token_id = model_obj.config.eos_token_id
        del model_obj
        print("loading pipeline")
        self.pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            **kwargs,
        )
        print("loading pipeline done.")
        self.pipeline.tokenizer.pad_token_id = pad_token_id
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    def __call__(self, dataset):
        """
        Passes dataset to LLM and returns the responses.
        :param dataset:  Hugging Face dataset containing a 'text' column with prompts.
        :return: list of strings with responses.
        """
        ret = []
        for i, out in enumerate(tqdm(
                self.pipeline(
                    KeyDataset(dataset, "text"),
                    max_new_tokens=self.max_new_tokens,
                    batch_size=self.batch_size,
                )
        )):
            # remove input in case pipeline is using completion/plain prompt
            response = out[0]["generated_text"]
            response = response.replace(dataset[i]['text'], '').strip()
            ret.append(response)
        return ret

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument("--seed_file", type=str)
    parser.add_argument("--column_names", nargs='+', default="instruction")
    parser.add_argument("--num_rows", type=int, default=5)
    parser.add_argument("--min_len_chars", type=int, default=32)
    parser.add_argument("--max_len_chars", type=int, default=512)
    parser.add_argument("--temperature", type=int, default=0.7)
    parser.add_argument("--top_p", type=int, default=0.95)
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--language", type=str, default="Korean")
    parser.add_argument("--question_only", type=bool, default=True)
    
    args = parser.parse_args()

    llm_pipeline = AzureGPTPipeline(
        args.model_name, 
        max_tokens=args.max_len_chars,
        temperature=args.temperature,
        top_p=args.top_p
    )

    wizardlm = WizardLM(
        llm_pipeline=llm_pipeline,
        seed_data=args.seed_file,
        column_names=args.column_names,
        num_rows=args.num_rows,
        min_len_chars=args.min_len_chars,
        max_len_chars=args.max_len_chars,
        language=args.language,
        verbose=True,
    )

    if args.question_only:
        wizardlm.run_question_only()
    else:
        wizardlm.run()