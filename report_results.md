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