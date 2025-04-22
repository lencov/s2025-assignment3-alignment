# Assignment 3 Report - Results Summary

## Section 3: Measuring Zero-Shot Performance

### 3.2 Zero-shot MMLU baseline (Qwen2.5-0.5B)

*   **(c) Parsing Failures:** 5 out of 14,042 model generations failed parsing (0.04%).
    *   Examples:
        *   `The correct answer is I only``` (Missing space after 'is') - Occurred 3 times.
        *   `Answer:``` (Model failed to generate the required sentence format).
        *   `The primary impact of "Humanism ceasing to be the exclusive privilege of the few" was that it led to a broader and more inclusive understanding of the classics...` (Model generated a descriptive answer instead of the requested format and likely ran out of tokens).
*   **(d) Throughput:** Generation throughput was approximately **51.93 examples/second**.
*   **(e) Performance:** The zero-shot baseline achieved an accuracy of **33.65%** on the MMLU test set (calculated on 14,037 parsable examples).
*   **(f) Error Analysis:** Analysis of 10 random incorrect predictions revealed diverse error types. Frequent issues included **factual inaccuracies** across multiple domains (e.g., history, biology, psychology, math) and **misinterpretation or reasoning errors** based on provided text passages. There was also evidence of **calculation errors** in quantitative problems and occasional **output formatting issues** where the model repeated itself or failed to adhere strictly to the prompt format even when producing a parsable letter.

### 3.3 Zero-shot GSM8K baseline (Qwen2.5-0.5B)

*   **(c) Parsing Failures:** 0 out of 1,319 model generations failed parsing (0.00%).
*   **(d) Throughput:** Generation throughput was approximately **21.00 examples/second**.
*   **(e) Performance:** The zero-shot baseline achieved an accuracy of **22.02%** on the GSM8K test set (calculated on 1,317 examples where both prediction and ground truth could be parsed).
*   **(f) Error Analysis:** Analysis of incorrect predictions revealed frequent **reasoning and logic errors**. The model struggled with multi-step problems, often making mistakes in intermediate calculations, using incorrect numbers, or misinterpreting the relationships described. **Arithmetic errors** (e.g., using addition instead of subtraction) were also common, particularly in the final step. Several examples showed **incomplete outputs**, where the model was cut off mid-calculation. One instance involved **hallucination**, where the correct answer was followed by unrelated text, leading to incorrect parsing. 