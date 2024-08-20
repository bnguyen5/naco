# Reference-based Metrics Disprove Themselves in Question Generation

This repository contains the code and data supporting our paper that (1) investigates the effectiveness of reference-based metrics for Question Generation (QG) and (2) proposes a multi-dimensional reference-free metric for QG that achieves state-of-the-art alignment with human judgement of question quality.

### Data Folder

- **`data/squad_candidates.json`**: Contains 748 examples from the test set of the SQuAD dataset, each annotated with 2 additional human-written references.
- **`data/hotpotqa_candidates.json`**: Contains 64 examples from the HotpotQA dataset, each has 4 annotated questions: 2 additional human-written references, and 2 less desirable candidate questions

### Code Files

- **`naco.py`** implements the proposed metric NACo to evaluate the naturalness, answerability, and complexity of generated questions.
  - The current implementation of NACo uses GPT3.5 as the underlying Large Language Model (LLM). Hence, it relies on OpenAI API calls through the `call_llm_func` function. However, users can easily use a different LLM by replacing `call_llm_func` with their own implementation that interacts with other large language models (LLMs), either locally or via other API services.
