
import json
import os
import re
import random

import cv2
from tqdm import tqdm

from constants import DESCRIPTIVE_RESP_INST


instruction = """<image>\nGiven a chart, a question, and several answers which may contain the correct one, your task is: 
If there is a correct response, output it. If not, generate the correct answer.
"""


def calculate_blank_ratio(image_path, threshold=245):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Image not found or invalid!")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    white_pixels = cv2.countNonZero(mask)

    total_pixels = image.shape[0] * image.shape[1]
    blank_ratio = white_pixels / total_pixels

    return blank_ratio

def descriptive_query_helper(qid, subplot_loc):
    if qid in [18, 19]:
        # skip subplot location when asking about the layout of the subplots
        return DESCRIPTIVE_RESP_INST[qid] 
    if isinstance(subplot_loc, list):
        if subplot_loc[0] == 0:
            # when there is only one subplot
            prefix = "For the current plot, "
        else:
            # when there are multiple subplots
            prefix = f"For the subplot at row {subplot_loc[0]} and column {subplot_loc[1]}, "
    # when subplots do not form a grid
    elif isinstance(subplot_loc, str):
        prefix = f"For {subplot_loc}, "
    else:
        raise ValueError(f"Invalid subplot_loc: {subplot_loc}")
    # return the question with the subplot location
    return DESCRIPTIVE_RESP_INST[qid].format(prefix)

def extract_to_ft():
    data_info = json.load(open("../CharXiv/data/descriptive_test.json"))

    data_ids = [int(d) for d in data_info]

    ft_data = []
    conflict_cnt = 0
    for data_idx in tqdm(data_ids):
        if os.path.exists(f"charts/{data_idx}_aj.jpg"):
            img_path = f"charts/{data_idx}_aj.jpg"
        elif os.path.exists(f"charts/{data_idx}_ck.jpg"):
            img_path = f"charts/{data_idx}_ck.jpg"
        elif sum((os.path.exists(f"charts/{data_idx}_fx{debug_num}.jpg") for debug_num in range(3))) > 0:
            for debug_num in range(2, -1, -1):  # from 2 to 0
                if os.path.exists(f"charts/{data_idx}_fx{debug_num}.jpg"):
                    img_path = f"charts/{data_idx}_fx{debug_num}.jpg"
                    break
        elif os.path.exists(f"charts/{data_idx}.jpg"):
            img_path = f"charts/{data_idx}.jpg"
        else:
            # raise ValueError(f"Image not found for {d}")
            continue

        if calculate_blank_ratio(img_path) > 0.95:
            continue
        
        qst_list = []
        direct_answer_lst = []
        if os.path.exists(f"mul_qas/{data_idx}.jsonl"):
            with open(f"mul_qas/{data_idx}.jsonl", "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    question = data['question']
                    answers = []

                    # Iterate over each answer
                    for i in range(5):
                        answer_key = f'answer_{i}'
                        answer = data.get(answer_key, "")

                        # Check the length of the answer and decide whether to include it
                        if len(answer.split()) > 128:
                            # With 50% probability, remove the answer
                            if random.random() <= 0.5:
                                continue
                        
                        answers.append(f"{len(answers) + 1}. {answer}")

                    qst_str = "Question:\n" + question + "\nResponses:\n" + "\n".join(answers)
                    ans_str = data['gt_answer']

                    if ans_str.strip() in ["", "\"\"", "."]:
                        continue

                    qst_list.append(qst_str)
                    direct_answer_lst.append(ans_str)
        else:
            continue
        
        if direct_answer_lst == ["", "", "", ""]:
            continue
        
        final_question_lst = []
        final_answer_lst = []
        for an_idx, (qst, da) in enumerate(zip(qst_list, direct_answer_lst)):
            if "not applicable" in da.lower() and random.random() < 0.8:  # drop 80% of the "not applicable" answers
                final_question_lst.append("")
                final_answer_lst.append("")
                continue
            final_question_lst.append(qst)
            final_answer_lst.append(da)

        # create ft data
        assert len(final_question_lst) == len(final_answer_lst)
        for q, a in zip(final_question_lst, final_answer_lst):
            if a.strip() in ["", "\"\""]:
                continue
            skip = False
            for noise in ["```python", "```", "axes[", "ax[", "ax.", "subplots(", "00000", "code", "np"]:
                if noise in a:
                    skip = True
                    break
            
            if "\\u2212" in a:
                a = a.replace("\\u2212", "-")

            ft_data.append({'conversations': [{'from': 'human', 'value': instruction + q}, {'from': 'gpt', 'value': a}], 'image': img_path})

    # # save as jsonl
    with open('critic_data.jsonl', 'w') as f:
        for d in ft_data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    extract_to_ft()