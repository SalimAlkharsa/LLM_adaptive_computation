import re

def parse_logs(file_path):
    with open(file_path, 'r') as file:
        log_data = file.read()
    
    # Define regex patterns
    sample_pattern = r"Sample \d+:\n(.+?)Expected:(.+?)Predicted:(.+?)(?:\-{10,}|$)"
    matches = re.findall(sample_pattern, log_data, re.DOTALL)

    parsed_data = []
    
    for match in matches:
        question = match[0].strip().split('Question:')[-1].strip()
        expected = match[1].strip()
        predicted = match[2].strip()
        parsed_data.append({
            "Question": question,
            "Expected": expected,
            "Predicted": predicted
        })
    
    return parsed_data

def save_parsed_logs(parsed_data, output_file):
    with open(output_file, 'w') as file:
        for entry in parsed_data:
            file.write("Question:\n")
            file.write(entry["Question"] + "\n")
            file.write("Expected:\n")
            file.write(entry["Expected"] + "\n")
            file.write("Predicted:\n")
            file.write(entry["Predicted"] + "\n")
            file.write("-" * 40 + "\n")

# Example usage
input_file = 'first_batch.log'
output_file = 'parsed_output_TEST.log'

parsed_data = parse_logs(input_file)
save_parsed_logs(parsed_data, output_file)

print(f"Parsed logs saved to {output_file}")
