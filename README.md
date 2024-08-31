# Reference-based Metrics Disprove Themselves in Question Generation

This repository contains the code and data supporting our paper that (1) investigates the effectiveness of reference-based metrics for Question Generation (QG) and (2) proposes a multi-dimensional reference-free metric for QG that achieves state-of-the-art alignment with human judgement of question quality.

### Data Folder

- **`data/squad_candidates.json`**: Contains 748 examples from the test set of the SQuAD dataset, each annotated with 2 additional human-written references. Each line is a data instance with the following information:
  - `og_id`: Id of the reference question from the [original SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/).
  - `context`: Context Paragraph
  - `answer`: Answer Text
  - `refq`: Original question from the [original SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/)
  - `annoq`: New reference created by our human annotators
  - `annoq_clue_refq`: New reference created by our human annotators that make use of clue words used by `refq`
- **`data/hotpotqa_candidates.json`**: Contains 96 examples from the HotpotQA dataset, each has 4 annotated questions: 2 additional human-written references, and 2 less desirable candidate questions
  - `og_id`: Id of the reference question from the [original HotpotQA dataset](https://hotpotqa.github.io/).
  - `contex_w_ans`: Concatenated Context Passages and Highlighted Answer Span
  - `refq`: Original question from the [original HotpotQA dataset](https://hotpotqa.github.io/)
  - `annoq`: New reference created by our human annotators
  - `annoq_clue_refq`: New reference created by our human annotators that make use of clue words used by `refq`
  - `1hop_q`: Single-hop question generated by GPT3.5 and verified by our human annotators
  - `non_q`: Non-question sentence using similar wording as `refq`, generated by GPT3.5 and verified by our human annotators

### Code Files

- **`naco.py`** implements the proposed metric NACo to evaluate the naturalness, answerability, and complexity of generated questions.
  - The current implementation of NACo uses GPT3.5 as the underlying Large Language Model (LLM). Hence, it relies on OpenAI API calls through the `call_llm_func` function. However, users can easily use a different LLM by replacing `call_llm_func` with their own implementation that interacts with other large language models (LLMs), either locally or via other API services.
- **`run_naco.py`** provides a demo script for using NACo for evaluating question quality on the demo data (`data/demo_train.json` and `data/demo_test.json`).
  - First, set up your OpenAI API Key
  - Then, simply run `python run_naco.py` for demonstration.
  - Example output
  ```
  [{'naco': 0.9885057471264367, 'naturalness': 1.0, 'answerability': 1.0, 'complexity': 4.666666666666667, 'expected_complexity': 4.833333333333334}, {'naco': 0.9195402298850573, 'naturalness': 1.0, 'answerability': 1.0, 'complexity': 3.6666666666666665, 'expected_complexity': 4.833333333333334}, {'naco': 0.8965517241379309, 'naturalness': 1.0, 'answerability': 1.0, 'complexity': 3.3333333333333335, 'expected_complexity': 4.833333333333334}, {'naco': 0, 'naturalness': 0.3333333333333333, 'answerability': 0.0, 'complexity': 2.6666666666666665, 'expected_complexity': 4.833333333333334}]
  ```
