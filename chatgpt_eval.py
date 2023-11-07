from bdc.utils import call_chatgpt
from openai.error import InvalidRequestError
from lm_eval.tasks import (
    rama_task1_select_proper_candidate,
    rama_task2_result_evidence,
    rama_task4_predict_category,
    rama_task5_career_knowledge,
    rama_task6_predict_industries,
    rama_task7_find_similar_company,
)
from tqdm import tqdm
from collections import defaultdict
import time

# %%
spc = rama_task1_select_proper_candidate.SelectProperCandidates()
# %%
example_str = ["A", "B", "C", "D"]
# %%
input_list = []
output_list = []
error = 0

verbose = True

for di, data in tqdm(enumerate(spc.dataset["test"]), total=len(spc.dataset["test"])):
    doc = spc._process_doc(data)

    cc_char = [example_str[ci] for ci in range(len(doc["choices"]))]
    choices = [f"{cc}: {example}" for cc, example in zip(cc_char, doc["choices"])]

    message = doc["query"] + "\n" + "\n".join(choices)
    message = message + "\n 정답: "

    try:
        response = call_chatgpt(
            engine="gpt-4-0613",  # [gpt-35-turbo-0613, gpt-4-0613]
            user_message=message,
            system_message="""""",
            temperature=0,
            max_tokens=5,
        )
    except InvalidRequestError:
        print(f"length error: {di}")
        error += 1
        continue

    answer = example_str[doc["gold"]]

    input_list.append(di)
    output_list.append(response)
    if verbose:
        print(answer, response)
    if di % 5 == 0:
        time.sleep(1)
# %%
acc = 0
by_type_acc = defaultdict(list)

for di, response in zip(input_list, output_list):
    doc = spc.dataset["test"][di]

    answer = example_str[doc["answer"]]

    if answer in response:
        acc += 1

    by_type_acc[doc["type"]].append(answer in response)

print(acc / len(input_list), acc, len(input_list))
# %%
for type, results in by_type_acc.items():
    print(type, sum(results) / len(results), len(results))

# %%
