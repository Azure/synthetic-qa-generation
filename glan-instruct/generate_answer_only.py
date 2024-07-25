import argparse
import jsonlines
from glan import generate_answers, read_jsonl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')

    parser.add_argument("--questions_filepath", type=str, default="samples/GLAN_Instructions_Korean_20_Samples_3b7e.jsonl")
    parser.add_argument("--model_name_for_answer", type=str, default="gpt-4o")
    parser.add_argument("--answer_max_tokens", type=int, default=2048)
    parser.add_argument("--answer_batch_size", type=int, default=5)
    
    args = parser.parse_args()   

    filename = args.questions_filepath
    all_questions = read_jsonl(filename)

    all_answers = generate_answers(
        all_questions, 
        model_name=args.model_name_for_answer, 
        max_tokens=args.answer_max_tokens, 
        batch_size=args.answer_batch_size
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