import json
import numpy as np
from datasets import load_dataset, Dataset
from transformers import pipeline, AutoTokenizer, LlamaForCausalLM


MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

GENERATION_MAX_TOKENS = 1024

TEST_GEN_TEMPLATE_DEP = """
Extract three self-contained, semantically independent sentences from the provided text (containing no or minimal pronouns), where the subject is an entity and the sentence includes an adverbial phrase (such as purpose or location). Then, transform each sentence by removing certain parts to create three question-answer pairs. Ensure that each correct answer contains no more than three words. Provide the output in JSON format.

# Steps

1. **Sentence Selection**:
   - Choose exactly three semantically independent sentences.
   - Ensure each sentence has an entity as the subject.
   - Include sentences containing adverbial phrases, such as purpose, location, time, or means.

2. **Create Questions**:
   - For each selected sentence, remove key parts (such as the subject, action, or adverbial phrase) to form a question.
   - Ensure the question maintains grammatical coherence.

3. **Generate Answers**:
   - Provide correct answers corresponding to the removed parts from the sentence.
   - Ensure answers are concise and do not exceed three words.

4. **Maintain Accuracy**:
   - Questions and answers must directly reflect the content of the original sentences.
   - Ensure the extracted parts in the questions relate specifically to an entity, place, time, or intent.

# Output Format

[
  {{
    "question": "",
    "distractor1": "", 
    "distractor2": "", 
    "distractor3": "",
    "correct_answer": ""
  }}
]

# Notes

- If fewer than three suitable sentences exist in the text, indicate this with an explanation in the JSON output.
- Focus on extracting content directly from the input text without introducing external details.
- Prioritize clarity and grammatical correctness when forming questions and answers.
- Please ignore authors' information in the given text.


Use the following content to generate the questions: {text}
"""

TEST_GEN_TEMPLATE_PAPER = """"
Assume you are a teacher who wants to check if the given text is memorized by your students, please design some questions.
Note that you should not design question like "... in this paper", because students can not see the given text when fininshing these questions.
Output format should be like json and does not contain other sentences:
[
  {{
    "question": "",
    "distractor1": "", 
    "distractor2": "", 
    "distractor3": "",
    "correct_answer": ""
  }}
]
Given text: {text}
"""

TEST_GEN_TEMPLATE_CODE = """

"""

TEST_GEN_TEMPLATE_MATH = """

"""

TEST_CHECK_TEMPLATE = """
Answer the following question:
Question: {question}
A: {distractor1}
B: {distractor2}
C: {distractor3}
D: {correct_answer}

You should only answer with A, B, C, or D, without any additional information.
Your answer:
"""

TEXT_FILTER_TEMPLATE = """
The correct text should meet following standards:
1. Contains information about something novel instead of facts;
2. Understandable without any additional information;
3. Can be used to refine knowledge.

Please tell me if the given text meets all standards, you should only answer with T or F without any additional information.

Given text: {text}
"""

BATCH_GEN_SIZE = 128


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
model = LlamaForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="float16"
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = 128001
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)


def load_mimir_dataset(name: str, split: str, member_or_non: bool) -> Dataset:
    dataset = load_dataset(
        "iamgroot42/mimir", name, split=split, trust_remote_code=True
    )

    assert "member" in dataset.column_names
    assert "nonmember" in dataset.column_names

    all_texts = []
    if member_or_non:
        all_texts = [dataset["member"][k] for k in range(len(dataset))]
    else:
        all_texts = [dataset["nonmember"][k] for k in range(len(dataset))]

    new_dataset = Dataset.from_dict({"text": all_texts})
    return new_dataset


def chat(user_prompts: list[str]) -> list[str]:
    """Chat with llm with a batch of prompts.

    Args:
        user_prompts (list[str]): prompts batch.

    Returns:
        list[str]: responses.
    """
    chat_prompt_batch = []
    for user_prompt in user_prompts:
        chat_prompt_batch.append(
            [
                {
                    "role": "system",
                    "content": "You are an AI assistant that helps people find information.",
                },
                {"role": "user", "content": user_prompt},
            ]
        )
    llm_resp_batch = llm(
        chat_prompt_batch,
        batch_size=len(chat_prompt_batch),
        max_new_tokens=GENERATION_MAX_TOKENS,
    )
    results = []
    for llm_resp in llm_resp_batch:
        results.append(llm_resp[0]["generated_text"][-1]["content"])
    return results


def check_test(single_choice_tests: list[dict]) -> list[bool]:
    """Check if the llm can solve this test.

    Args:
        single_choice_tests (list[dict]): A set of single choice tests.

    Returns:
        list[bool]: Results of each test.
    """
    test_prompts = list(map(
        lambda test: TEST_CHECK_TEMPLATE.format(
            question=test["question"],
            distractor1=test["distractor1"],
            distractor2=test["distractor2"],
            distractor3=test["distractor3"],
            correct_answer=test["correct_answer"],
        ),
        single_choice_tests,
    ))
    test_resps = chat(test_prompts)
    test_res = list(map(lambda resp: resp == "D", test_resps))
    return test_res


def text_filter(texts: list[str]) -> list[bool]:
    """Filter useless texts.

    Args:
        texts (list[str]): texts.

    Returns:
        list[bool]: Results of each text filtering.
    """
    text_filter_prompts = list(map(lambda text: TEXT_FILTER_TEMPLATE.format(text=text), texts))
    text_filter_resps = chat(text_filter_prompts)
    text_filter_res = list(map(lambda resp: resp == "T", text_filter_resps))
    print("Filter a batch!")
    return text_filter_res


def gen_test(texts: list[str]) -> list[str]:
    """Generate tests according to given texts.

    Args:
        texts (list[str]): given texts.

    Returns:
        list[str]: A batch of responses.
    """
    gen_test_prompts = list(map(lambda text: TEST_GEN_TEMPLATE_PAPER.format(text=text), texts))
    gen_test_resps = chat(gen_test_prompts)
    print("Gen a batch!")
    return gen_test_resps


def parse_or_throw_json(raw_string: str):
    """Parse json

    Args:
        raw_string (str): raw string

    Returns:
        list or dict: Return none if failed
    """
    try:
        res = json.loads(raw_string)
        return res
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    member_dataset = load_mimir_dataset(
        name="dm_mathematics", split="ngram_13_0.8", member_or_non=True
    )["text"]

    text_filter_batches = np.array_split(member_dataset, len(member_dataset) / BATCH_GEN_SIZE)

    text_filter_results = list(map(lambda text_batch: text_filter(text_batch), text_filter_batches))
    text_filter_results_merged = np.concatenate(text_filter_results)

    print("Filter Success!")

    text_filtered = list(filter(lambda text_and_filter_res: text_and_filter_res[1], zip(member_dataset, text_filter_results_merged)))
    text_filtered_batches = np.array_split(text_filtered, len(member_dataset) / BATCH_GEN_SIZE)

    gen_tests_results = list(map(lambda text_batch: gen_test(text_batch), text_filtered_batches))
    gen_tests_results_merged = np.concatenate(gen_tests_results)

    print("Gen Success!")

    gen_tests_struct = list(filter(lambda test: test is not None, map(lambda test: parse_or_throw_json(test), gen_tests_results_merged)))
    gen_tests_struct_merged = np.concatenate(gen_tests_struct).tolist()

    with open("member_qa.json", "w", encoding="utf-8") as f:
        json.dump(gen_tests_struct_merged, f, ensure_ascii=False, indent=4)
