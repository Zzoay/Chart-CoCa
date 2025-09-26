

import os

import json
import random
import re
import shutil

import cv2
import numpy as np
import torch
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from lmdeploy.vl import load_image
from tqdm import tqdm

from constants import *


batch_size = 16 * torch.cuda.device_count()
from_scratch = True
extract_info = True
code_rerun_num = 4
dataset_path = '../CharXiv/'
info_path = os.path.join(dataset_path, 'data')
image_path = os.path.join(dataset_path, 'images')
model = '/home/share/ckpt/InternVL2-8B'


pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=4096, tp=torch.cuda.device_count(), cache_max_entry_count=0.9))

prompt = """Given a question, a response to the question, and a ground truth answer, your task is to judge whether the response correctly answer the given question, by comparing with the ground truth. First, extract the answer from the response, and then compare the extracted answer with the ground truth. Finally, if the response is correct, output 1, otherwise, output 0.
Output Format:
```
## Extracted Answer
...
## Comparison
...
## Score
...
```
"""

qst_list = []
response_list = []
direct_answer_lst = []
direct_answer_short_lst = []

data_info = json.load(open("../CharXiv/data/descriptive_test.json"))

data_ids = [int(d) for d in data_info]


ft_data = []
conflict_cnt = 0
for data_idx in tqdm(data_ids):
    if data_idx < 774:
        continue
    if os.path.exists(f"mul_qas/{data_idx}.jsonl"):
        fw = open(f"mul_qas/{data_idx}_score.jsonl", "w")
        with open(f"mul_qas/{data_idx}.jsonl", "r") as f:
            for line in f:
                data = json.loads(line.strip())
                question = data['question']
                answers = []
                responses = []

                batch_input = []
                # Iterate over each answer
                for i in range(5):
                    answer_key = f'answer_{i}'
                    answer = data.get(answer_key, "")
                    
                    answers.append(f"\nResponse {len(answers) + 1}:\n{answer}")
                    responses.append(answer)

                    batch_input.append(f"{prompt}\n\n##Question\n{question}\n\n##Response\n{answer}\n\n##Ground Truth\n{data['gt_answer']}")
                    
                # qst_str = "Question:\n" + question + f"\n\nResponses:\nThe Number of Responses:{len(answers)}\n" + "\n".join(answers)
                ans_str = data['gt_answer']
                # ans_str_short = data['gt_answer_short']

                if ans_str.strip() in ["", "\"\"", "."]:
                    continue

                # qst_list.append(qst_str)
                # response_list.append(responses)
                # direct_answer_lst.append(ans_str)
                # direct_answer_short_lst.append(ans_str_short)

                while True:
                    try:
                        batch_responses = pipe(batch_input, gen_config=GenerationConfig(max_new_tokens=512, do_sample=True, top_p=0.6))
                        batch_responses = [response.text for response in batch_responses]
                    
                        scores = []
                        for response in batch_responses:
                            score = int(response[response.index('## Score')+len('## Score'):].strip())
                            scores.append(score)

                        break
                    except ValueError:
                        continue
                for idx, score in enumerate(scores):
                    data[f'score_{idx}'] = score
                fw.write(json.dumps(data, ensure_ascii=False) + "\n")
        fw.close()
    else:
        continue


