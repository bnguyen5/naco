"""
This module implements NACo, a reference-free metric for evaluating question generation using large language models (LLMs).
Users are primarily intended to interact with the `get_naco` function, which calculates the NACO score
for a set of questions, answers, and contexts.

Functions:
- get_naco: The main function to calculate NACO scores.
- create_cotqa_prompt: Helper function to generate the CotQA prompt for the LLM.
- call_llm: Function to call the LLM API with a prompt.
- get_individual_scores: Function to evalaute question's quality based on individual criteria.
- combine_scores: Function to combine individual scores into the final NACo score.
"""
import re
import string
import statistics
from collections import Counter
from tqdm import tqdm
import openai
import logging
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type  # for exponential backoff

# Retry decorator for handling API errors with exponential backoff
@retry(
    retry=retry_if_exception_type((
        openai.error.APIError,
        openai.error.APIConnectionError,
        openai.error.RateLimitError,
        openai.error.ServiceUnavailableError,
        openai.error.Timeout
    )),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10)
)
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def create_cotqa_prompt(contexts, question):
    """
    Generates the CoTQA prompt for the LLM to answer the candidate question in the context of provided passages.

    Args:
        contexts (list of str): The context passage(s).
        question (str): The question to evaluate.

    Returns:
        str: The formatted CoTQA prompt for the LLM.
    """
    prompt = (
        "You will be given one or more context passages and a sentence. "
        "If the sentence is a question, your task is to output a text span from the context passage to answer the question. "
        "Your answer should NOT be complete sentences.\n"
        "Instructions: \n"
        "\t1a. Let's read the passage first and then read the sentence. "
        "Is the sentence a question? If yes, what information indicates that it is a question? If not, output 'not a question' and stop generation.\n"
        "\t1b. If it is a question, consider: what is the question asking about? Do not make any assumption that was not specifically mentioned in the question. "
        "If the question is unclear, or has grammar errors, output 'Question unnatural'. Otherwise, output 'Question natural'.\n"
        "\t2. Now find the answer to the question. Speak out loud your detailed reasoning process to answer the question.\n"
        "\t3. Highlight your answer between two <ans> tokens.\n\n"
        "Format your output as follows:\n"
        "\t1. Your response to 1a and 1b.\n"
        "\t2. Step-by-step reasoning:\n"
        "\t\tStep 1: [reasoning step should be a single sentence with one clause] \n"
        "\t\tStep 2: [reasoning step should be a single sentence with one clause] \n"
        "\t\t...\n"
        "\t3. Answer: <ans>[answer text]</ans>\n\n"
    )
    for ct_idx, ct in enumerate(contexts, start=1):
        prompt += f"Context Passage {ct_idx}: {ct}\n"
    
    prompt += f"Sentence: {question}\nResponse: "
    return prompt


def call_llm(prompt, model_name):
    """
    Calls the LLM API with a prompt.

    Args:
        prompt (str): The prompt to send to the LLM.
        model_name (str): The name of the LLM model to use.

    Returns:
        str: The LLM's response to the prompt.
    """
    completion = chat_completion_with_backoff(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a data annotator."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content


def get_individual_scores(cotqa_res, answer):
    """
    Extracts and calculates individual scores (naturalness, answerability, and complexity) of the candidate question based on the LLM's CoTQA response.

    Args:
        cotqa_res (str): The LLM's response to the CoTQA prompt.
        answer (str): The expected answer.

    Returns:
        tuple: A tuple containing naturalness, answerability, and complexity scores.
    """
    def find_ans(input_string):
        patterns = [r'<ans>(.*?)<ans>', r'<ans>(.*?)</ans>']
        for pattern in patterns:
            matches = re.findall(pattern, input_string)
            if matches:
                return matches[0]
        return ""

    def normalize_answer(s):
        """Lower text and remove punctuation, articles, and extra whitespace."""
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def calculate_f1_score(prediction, ground_truth):
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        return (2 * precision * recall) / (precision + recall)

    n_cand = 1
    if "not a question" in cotqa_res.lower():
        n_cand = 0
    elif "unnatural" in cotqa_res.lower():
        n_cand = 0.5

    pred_ans = find_ans(cotqa_res)
    a_cand = calculate_f1_score(pred_ans, answer)

    reasoning_counts = [m.start() for m in re.finditer('step', cotqa_res.lower())]
    if len(reasoning_counts) > 2 and "step by step" in cotqa_res.lower():
        c_cand = len(reasoning_counts) - 2
    elif len(reasoning_counts) == 0:
        c_cand = 1
    else:
        c_cand = len(reasoning_counts)

    return n_cand, a_cand, c_cand


def combine_scores(n_cand, a_cand, c_cand, expected_c, w_n=1/3, w_a=1/3, w_c=1/3):
    """
    Combines individual scores into a final NACo score.

    Args:
        n_cand (float): Naturalness score.
        a_cand (float): Answerability score.
        c_cand (float): Complexity score.
        expected_c (float): Expected complexity.
        w_n (float): Weight for naturalness.
        w_a (float): Weight for answerability.
        w_c (float): Weight for complexity.

    Returns:
        float: The final NACO score.
    """
    c_cand = expected_c / c_cand  if c_cand > expected_c else c_cand / expected_c

    if n_cand == 0 or a_cand ==0:
        naco = 0
    else:
        naco = w_n * n_cand + w_a * a_cand + w_c * c_cand

    return naco


def get_naco(contexts, answers, questions, model_name, call_llm_func=call_llm, n=1, w_n=1/3, w_a=1/3, w_c=1/3, expected_c=None, examples=[]):
    """
    Calculates NACO scores for a set of candidate questions

    Args:
        contexts (list of list of str): Context passages.
        answers (list of str): Expected answers.
        questions (list of str): Questions to evaluate.
        model_name (str): LLM model to use.
        call_llm_func (function): Function to call the LLM.
        n (int): Number of runs for each example.
        w_n (float): Weight for naturalness.
        w_a (float): Weight for answerability.
        w_c (float): Weight for complexity.
        expected_c (float): Expected complexity.
        examples (list of tuples): Example contexts, answers, and reference questions to calculate expected complexity if not provided.

    Returns:
        list of dict: A list of dictionaries containing the NACO score and its components for each question.
    """
    if expected_c is None and not examples:
        raise ValueError("If expected complexity is not provided, a set of references must be provided as `examples` parameter to calculate expected complexity.")
    
    if expected_c is None:
        logging.info("No expected reasoning provided, calculating expected complexity.")
        all_ref_cs = []
        for cts, ans, question in tqdm(examples, desc="Calculating expected complexity"):
            total_c_score = 0
            for _ in range(n):
                cotqa_prompt = create_cotqa_prompt(cts, question)
                cotqa_res = call_llm_func(prompt=cotqa_prompt, model_name=model_name)
                _, _, ref_c = get_individual_scores(cotqa_res, ans)
                total_c_score += ref_c
            avg_ref_c = total_c_score / n
            all_ref_cs.append(avg_ref_c)
        expected_c = statistics.median(all_ref_cs)

    scores = []
    for cts, ans, question in tqdm(zip(contexts, answers, questions), total=len(contexts), desc="Evaluating questions"):
        total_n_score = 0
        total_a_score = 0
        total_c_score = 0

        for _ in range(n):
            cotqa_prompt = create_cotqa_prompt(cts, question)
            cotqa_res = call_llm_func(prompt=cotqa_prompt, model_name=model_name)
            n_cand, a_cand, c_cand = get_individual_scores(cotqa_res, ans)
            total_n_score += n_cand
            total_a_score += a_cand
            total_c_score += c_cand

        n_avg = total_n_score / n
        a_avg = total_a_score / n
        c_avg = total_c_score / n

        naco = combine_scores(n_avg, a_avg, c_avg, expected_c, w_n=w_n, w_a=w_a, w_c=w_c)
        scores.append({
            'naco': naco,
            'naturalness': n_avg,
            'answerability': a_avg,
            'complexity': c_avg,
            'expected_complexity': expected_c
        })

    return scores
