import re

def parse_mmlu_response(model_output: str) -> str | None:
    """
    Parses the model output to find the predicted MMLU answer letter.

    Looks for the pattern "The correct answer is X" where X is A, B, C, or D.

    Args:
        model_output: The raw text output from the language model.

    Returns:
        The predicted letter ("A", "B", "C", or "D") if found, otherwise None.
    """
    # Strict regex matching the exact format requested in the prompt
    match = re.search(r"The correct answer is ([ABCD])", model_output)
    if match:
        # Check if there's potentially extraneous text after the expected pattern
        # This implementation prioritizes finding the pattern anywhere.
        # A stricter check might ensure the pattern is at the end, possibly with just punctuation.
        return match.group(1)
    
    # Optional: Add more lenient parsing attempts here if needed
    # For example, case-insensitive search or allowing minor variations.

    return None 