# Initialize counter
counter=1

# Read disciplines.txt line by line
while IFS= read -r line || [[ -n "$line" ]]; do
    # Create the corresponding disciplines file
    discipline_file="disciplines_line${counter}.txt"
    echo Created "$discipline_file"
    echo "$line" > "$discipline_file"

    # # Run the Python script with the current disciplines file
    python generate.py \
        --disciplines_filepath "$discipline_file" \
        --language Korean \
        --max_number_of_subjects 15 \
        --max_number_of_subtopics 30 \
        --max_number_of_session_name 30 \
        --num_iterations 15 \
        --num_questions_per_iteration 18 \
        --question_max_tokens 1024 \
        --question_batch_size 9 \
        --model_name_for_answer gpt-4o \
        --answer_max_tokens 2048 \
        --answer_batch_size 9

    # Increment counter
    ((counter++))

    # Delete the temporary disciplines file
    rm "$discipline_file"

done < disciplines_sample.txt