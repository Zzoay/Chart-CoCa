
import os

import json
import random
import re
import shutil
import traceback

import numpy as np
import torch
import matplotlib.pyplot as plt
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from lmdeploy.vl import load_image
from tqdm import tqdm

from utils import process_axes
from constants import *


batch_size = 32
from_scratch = False
code_rerun_num = 4
dataset_path = '../CharXiv/'
info_path = os.path.join(dataset_path, 'data')
image_path = os.path.join(dataset_path, 'images')

def descriptive_query_helper(qid, subplot_loc):
    if qid in [18, 19]:
        # skip subplot location when asking about the layout of the subplots
        return DESCRIPTIVE_RESP_INST[qid] 
    if isinstance(subplot_loc, list):
        if subplot_loc[0] == 0:
            # when there is only one subplot
            # prefix = "For the current plot, "
            prefix = f"For the subplot at row 1 and column 1, "
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

def extract_python_code_from_markdown(markdown_text):
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, markdown_text, re.DOTALL)
    if not matches:
        pattern = r"```\s*(.*?)\s*```"
        matches = re.findall(pattern, markdown_text, re.DOTALL)
        if not matches:
            return markdown_text
    return [match.strip() for match in matches][-1]

def refine_error_records(error_string, code_block):
    e_split = error_string.strip().split('\n')
    for i, e_line in enumerate(e_split):
        if '<string>' in e_line:
            line_num = re.findall(r'\d+', e_line)[-1]
            for idx, line in enumerate(code_block.split('\n')):
                if idx + 1 == int(line_num):
                    e_line += f'\n{line.strip()}'
                    break
            # e_split[i] = e_line
            e_split = [e_line, e_split[-2], e_split[-1]]
            break
        
        if i > 200:
            break

    e_record = '\n'.join(e_split)
    return e_record

def weighted_random_choice(start, end):
    numbers = list(range(start, end + 1))
    weights = [1 / (i + 1) for i in range(len(numbers))]
    total_weight = sum(weights)
    probabilities = [weight / total_weight for weight in weights]
    return random.choices(numbers, weights=probabilities, k=1)[0]

def run_python_files_and_save_images(code_folder_path, code_filename):
    # error_record = open('error_record.txt', 'w+')
    for filename in os.listdir(code_folder_path):
        if filename.endswith('.py') and filename.split('.')[0] == code_filename:
            file_path = os.path.join(code_folder_path, filename)
            try:
                with open(file_path, 'r') as f:
                    code = f.read()
                exec(code)
                plt.close('all')
            except Exception as e:
                # print(f"Error in id {filename.split('.')[0].split('_')[0]}...")
                error_string = traceback.format_exc()
                error_string = refine_error_records(error_string, code)
                with open(f"charts/{filename.split('.')[0]}.txt", 'w') as f:
                    f.write(error_string)
            break

def remove_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


if __name__ == '__main__':
    data_info = json.load(open(os.path.join(info_path, 'descriptive_test.json'), 'r')) 
    data_ids = [int(d) for d in data_info if int(d)]
    data_info = {int(d): data_info[d] for d in data_info if int(d)}
    
    # create questions
    questions = []
    for info in data_info.values():
        q_lst = []
        for i in range(4):
            q_str = descriptive_query_helper(info['qids'][i], info['subplot_loc'])
            q_lst.append(f'{i+1}. ' + q_str)
        questions.append("\n".join(q_lst))

    images = [load_image(os.path.join(image_path, f'{d}.jpg')) for d in data_ids]

    for i in tqdm(range(0, len(data_ids), batch_size)):
        batch_ids = data_ids[i:i+batch_size]
        batch_images = images[i:i+batch_size]
        batch_questions = questions[i:i+batch_size]
        print(f"IDs: {batch_ids[:5]} ... {batch_ids[-5:]}")

        # if batch_ids[0] < 352:
        #     continue

        print(f"Checking code...")
        # check the coherence of the code and the description
        batch_codes_to_check, batch_data_ids_to_check = [], []
        for data_id in batch_ids:
            if os.path.exists(f'charts/{data_id}_ck.jpg'):
                batch_data_ids_to_check.append(data_id)
                with open(f'codes/{data_id}_ck.py', 'r') as f:
                    code_block = f.read()
                batch_codes_to_check.append(code_block)
            elif sum((os.path.exists(f'charts/{data_id}_fx{debug_num}.jpg') for debug_num in range(code_rerun_num))) > 0:
                batch_data_ids_to_check.append(data_id)
                for debug_num in range(code_rerun_num-1, -1, -1):  # from 3 to 0
                    if os.path.exists(f'charts/{data_id}_fx{debug_num}.jpg'):
                        with open(f'codes/{data_id}_fx{debug_num}.py', 'r') as f:
                            code_block = f.read()
                        batch_codes_to_check.append(code_block)
                        break
            elif os.path.exists(f'charts/{data_id}.jpg'):
                batch_data_ids_to_check.append(data_id)
                with open(f'codes/{data_id}.py', 'r') as f:
                    code_block = f.read()
                batch_codes_to_check.append(code_block)
        assert len(batch_data_ids_to_check) == len(batch_codes_to_check)
        
        print(f"Extracting information from the code...")  
        for data_id, code in zip(batch_data_ids_to_check, batch_codes_to_check):
            # if data_id != 25:
            #     continue
            code_block = code
            row, col, find_subplot_in_code = 1, 1, False
            colorbar = "Not Applicable."  # specific element
            for line_idx, line in enumerate(code_block.split('\n')):
                if "plt.subplots(" in line:
                    numbers = re.findall(r'\d+', line)
                    if len(numbers) >= 2:  
                        row, col = int(numbers[0]), int(numbers[1])
                        find_subplot_in_code = True
                if "colorbar" in line:
                    colorbar = "Colorbar exists, look at the code to find the relevant information."
                if line_idx > 1000:  # prevent infinite loop
                    break
            info_str = f"General information: number of subplots: {row} row by {col} column, {row*col} in total. \nEach Subplot:\n"
            row_inject, col_inject = row, col
            full_code = code_block + '\n' + 'output = process_axes(fig, axes, row_inject, col_inject) if "axes" in locals() or "axes" in globals() else process_axes(fig, ax, row_inject, col_inject) if "ax" in locals() or "ax" in globals() else "{}"'
            full_code += f'\nplt.savefig("charts/{data_id}_aj.jpg", facecolor="white")'  # save the new image
            local_namespace = {}
            # try:
            print(f"Executing the code {data_id}...")
            # try:
            exec(full_code, {'process_axes': process_axes, 'np': np, 're': re, 'json': json, 'row_inject': row_inject, 'col_inject': col_inject}, local_namespace)
            info_str += local_namespace.get('output', 'None.')
            # except Exception as e:
            # info_str = "None."

            with open(os.path.join(f'infos/{data_id}.txt'), 'w') as f:
                f.write(info_str)