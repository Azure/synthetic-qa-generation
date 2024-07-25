import os
from azure.ai.generative.synthetic.qa import QADataGenerator, QAType
from typing import Dict, List, Tuple, Any, Union, Optional
from azure.ai.generative._telemetry import ActivityType, monitor_with_activity, ActivityLogger

activity_logger = ActivityLogger(__name__)
logger, module_logger = activity_logger.package_logger, activity_logger.module_logger

class CustomQADataGenerator(QADataGenerator):
    def __init__(self, templates_dir: str, **kwargs):
        self.templates_dir = templates_dir
        super().__init__(**kwargs)
        
    def _get_template(self, filename) -> str:
        logger.debug("Getting prompt template from %s file", filename)
        filepath = os.path.join(self.templates_dir, filename)
        with open(filepath, encoding="utf-8") as f:
            template = f.read()
        return template
    
    def _get_messages_for_qa_type(self, qa_type: QAType, text: str, num_questions: int) -> List:
        logger.debug("Getting prompt messages for %s QA type", qa_type)
        template_filename = {
            QAType.SHORT_ANSWER: "prompt_qa_short_answer.txt",
            QAType.LONG_ANSWER: "prompt_qa_long_answer.txt",
            QAType.BOOLEAN: "prompt_qa_boolean.txt",
            QAType.SUMMARY: "prompt_qa_summary.txt",
            QAType.CONVERSATION: "prompt_qa_conversation.txt",
        }
        filename = template_filename[qa_type]
        messages = self._get_messages_from_file(filename)
        input_variables: Dict[str, Any] = {"text": text}
        if qa_type == QAType.SUMMARY:
            input_variables["num_words"] = 100
        else:
            input_variables["num_questions"] = num_questions
        messages[-1]["content"] = messages[-1]["content"].format(**input_variables)
        return messages
    
    def _get_messages_for_modify_conversation(self, questions: List[str]) -> List:
        messages = self._get_messages_from_file("prompt_qa_conversation_modify.txt")
        questions_str = "\n".join([f"[Q]: {q}" for q in questions])
        messages[-1]["content"] = messages[-1]["content"].format(questions=questions_str)
        return messages
    