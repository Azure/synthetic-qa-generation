#!/bin/bash

INPUT_FILE="../seed/samples/advertising-oai.jsonl"
SEED_FILE="seed.jsonl"
COLUMN_NAMES="Instruction"
NUM_ROWS=10
MAX_LEN_CHARS=256

python convert.py --input_file "$INPUT_FILE" --output_file "$SEED_FILE"

python evolve.py --seed_file "$SEED_FILE" --column_names "$COLUMN_NAMES" --num_rows "$NUM_ROWS" --max_len_chars "$MAX_LEN_CHARS"
