import argparse
from glan import glan_instruction_generation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')

    parser.add_argument("--generate_disciplines", type=bool, default=False)
    parser.add_argument("--generate_question_only", type=bool, default=False)

    parser.add_argument("--disciplines_filepath", type=str, default="disciplines_sample.txt")
    parser.add_argument("--language", type=str, default="Korean")
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--model_name_for_answer", type=str, default="gpt-4o")
    
    parser.add_argument("--max_number_of_fields", type=int, default=1)
    parser.add_argument("--max_number_of_subjects", type=int, default=2)
    parser.add_argument("--max_number_of_subtopics", type=int, default=5)
    parser.add_argument("--max_number_of_session_name", type=int, default=3)

    parser.add_argument("--num_iterations", type=int, default=2)
    parser.add_argument("--num_questions_per_iteration", type=int, default=5)

    parser.add_argument("--question_max_tokens", type=int, default=768)
    parser.add_argument("--question_batch_size", type=int, default=5)
    parser.add_argument("--answer_max_tokens", type=int, default=2048)
    parser.add_argument("--answer_batch_size", type=int, default=5)

    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--logfile_name", type=str, default="logfile.log")
    
    args = parser.parse_args()    
    print(args)
    glan_instruction_generation(args)