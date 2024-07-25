# Generate questions and answers
python generate.py \
    --disciplines_filepath disciplines_sample.txt \
    --language English \
    --max_number_of_subjects 1 \
    --max_number_of_subtopics 3 \
    --max_number_of_session_name 3 \
    --num_iterations 1 \
    --num_questions_per_iteration 4 \
    --question_max_tokens 256 \
    --question_batch_size 4 \
    --answer_max_tokens 512 \
    --answer_batch_size 4

# Generate answers only
# python generate_answer_only.py \
#     --questions_filepath samples/GLAN_Questions_English_16_Samples_5c63.jsonl \
#     --model_name_for_answer gpt-4o \
#     --answer_max_tokens 512 \
#     --answer_batch_size 4