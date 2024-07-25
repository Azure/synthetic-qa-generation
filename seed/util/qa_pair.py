from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate

class QAPair(BaseModel):
    question: str = Field(description="Question generated from the text")
    answer: str = Field(description="Answer related to the question")


def get_qna_prompt_template(language="English"):

    prompt = PromptTemplate.from_template(
    """Context information is below. You are only aware of this context and nothing else.
    <context>
    {context}
    </context>

    You are the SME (Subject Matter Expert) in {domain}. 
    Based on the context provided, please generate exactly **{num_questions}** questions. Your questions should cover a variety of topics related to the subject matter. 
    Each question must be accompanied by its corresponding answer, which should be based solely on the information given in the context.

    Restrict the question(s) to the context information provided only. Provide the response in JSON format that includes the question and answer. The ANSWER should be a complete sentence.

    # Format:
    ```json
    {{
        "QUESTION": "Question here..",
        "ANSWER": "Answer here.."    
    }},  
    ```
    """
    ) 

    prompt += f"Write the QUESTION and ANSWER in {language}."

    return prompt
