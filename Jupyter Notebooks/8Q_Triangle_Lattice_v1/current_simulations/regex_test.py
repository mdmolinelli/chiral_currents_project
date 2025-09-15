import re

# Define the regex pattern
pattern = r"0|(?:\(([-+]?(?:\d*\.\d+|\d+)(?:[+-]\d*\.\d+j|[+-]\d+j|j)?|[-+]?\d*\.\d+j|[-+]?\d+j)\)\[([\d,]+)\](?:[+-]\(([-+]?(?:\d*\.\d+|\d+)(?:[+-]\d*\.\d+j|[+-]\d+j|j)?|[-+]?\d*\.\d+j|[-+]?\d+j)\)\[([\d,]+)\])*)"

single_state_pattern = r"\(([-+]?(?:\d*\.\d+|\d+)(?:[+-]\d*\.\d+j|[+-]\d+j|j)?|[-+]?\d*\.\d+j|[-+]?\d+j)\)\[([\d,]+)\]"

# Test strings
test_strings = [
    "0",  # Valid: Represents the zero state
    "(1)[0]",  # Valid: Single float with one index
    "(0.5)[1,2]",  # Invalid: Lists are not the same length (only one list here)
    "(-1.25)[3,4,5]",  # Valid: Single list
    "(2)[0]+(1j)[1]",  # Valid: Lists are the same length
    "(0.5)[1,2]-(1)[3,4]",  # Inalid: Lists are not the same length
    "(1)[0,1,2]+(0.75)[3,4,5]",  # Valid: Lists are the same length
    "(1)[0,1,2]+(0.75+0.75j)[3,4,5]-(0.5)[6,7,8]",  # Valid: Lists are the same length
    "(1)[0,1,2]+(0.75+0.75j)[3,4,5](0.5)[6,7]",  # Invalid: Lists are not the same length
]


def extract_coefficients_and_states(test_string):
    """
    Extract all coefficients and states from the string.
    """
    # Find all matches for (coefficient)[state]
    matches = re.findall(single_state_pattern, test_string)
    result = []
    for coefficient, state in matches:
        # Convert the state into a Python list of integers
        state_list = [int(x) for x in state.split(",")]
        result.append((coefficient, state_list))  # Store as (coefficient, state_list)
    return result

# Test the function on each string
for test_string in test_strings:
    extracted = extract_coefficients_and_states(test_string)
    print(f"'{test_string}' -> Extracted values: {extracted}")