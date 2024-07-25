import json
import re
import argparse

def transform_jsonl(input_file_path, output_file_path):
    output_data = []
    idx = 1
 
    with open(input_file_path, 'r', encoding='utf-8-sig') as infile:
        for line in infile:
            conversation = json.loads(line)
            skill = None
            # Extract skill string in "system" message
            system_message = next((msg for msg in conversation['messages'] if msg['role'] == 'system'), None)
            if system_message:
                match = re.search(r'SME \(Subject Matter Expert\) in ([\w\s]+)', system_message['content'])
                if match:
                    skill = match.group(1).strip()
            # Extract "user" message
            for message in conversation['messages']:
                if message['role'] == 'user' and skill:
                    output_data.append({
                        "idx": idx,
                        "Skill": skill,
                        "Difficulty": 5,
                        "Instruction": message['content']
                    })
                    idx += 1

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for entry in output_data:
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Transformed {len(output_data)} entries. Please run evolve.py to generate the augmented dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument("--input_file_path", type=str, default="../seed/dataset/advertising-multiple-oai.jsonl", help="Path to the input JSONL file")
    parser.add_argument("--output_file_path", type=str, default="seed.jsonl", help="Path to the output JSONL file")
    args = parser.parse_args()

    transform_jsonl(args.input_file_path, args.output_file_path)
