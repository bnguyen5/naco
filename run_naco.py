import openai
import os
import json
from reformatted_naco import get_naco


def load_json_file(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

if __name__ == "__main__":
    # Set OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Load demo data
    demo_train = load_json_file("data/demo_train.json")
    demo_test = load_json_file("data/demo_test.json")

    # Prepare test data
    test_cs = [demo_test['contexts']] * 4
    test_as = [demo_test['answer']] * 4
    test_qs = [
        demo_test['annotated_question_noclue'],
        demo_test['gpt35_1hop'],
        demo_test['bart_base_predicted'],
        demo_test['random_q']
    ]

    # Run NACo
    naco_scores = get_naco(test_cs, test_as, test_qs, model_name="gpt-4o-mini", n=3, examples=demo_train)
    print(naco_scores)
