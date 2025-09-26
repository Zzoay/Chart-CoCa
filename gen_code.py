
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


batch_size = 16 * torch.cuda.device_count()
from_scratch = False
extract_info = True
code_rerun_num = 4
dataset_path = '../CharXiv'
info_path = os.path.join(dataset_path, 'data')
image_path = os.path.join(dataset_path, 'images')
model = '/home/share/ckpt/InternVL2-8B'


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
            start_idx = markdown_text.find("import matplotlib")
            if start_idx != -1:
                return markdown_text[start_idx:]
            return markdown_text
    return [match.strip() for match in matches][-1]

def clear_directories(directory_paths):
    for directory_path in directory_paths:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

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

def extract_python_code_from_markdown(markdown_text):
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, markdown_text, re.DOTALL)
    if not matches:
        pattern = r"```\s*(.*?)\s*```"
        matches = re.findall(pattern, markdown_text, re.DOTALL)
        if not matches:
            return markdown_text.replace("```python", "").replace("```", "")
    return [match.strip() for match in matches][-1]

def save_extracted_code_to_file(code_block, output_file_path):
    with open(output_file_path, 'w') as output_file:
        output_file.write(code_block)

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
    """Remove the file if it exists."""
    if os.path.exists(file_path):
        os.remove(file_path)


if __name__ == '__main__':
    if from_scratch:
        print("Clearing directories...")
        clear_directories(['codes', 'charts', 'descriptions', 'qas', 'infos', 'direct_answers'])

    data_info = json.load(open(os.path.join(info_path, 'descriptive_test.json'), 'r')) 
    data_ids = [int(d) for d in data_info]

    # load remaining data
    if not from_scratch:
        print(f"Loading remaining data...")
        remain_data_ids = []
        for data_id in data_ids:
            if not os.path.exists(f'infos/{data_id}.txt'):
                remain_data_ids.append(data_id)

                # Define file paths for different file types and configurations
                base_paths = [
                    f'charts/{data_id}',
                    f'codes/{data_id}',
                    f'descriptions/{data_id}.txt',
                    # f'infos/{data_id}.txt'
                ]

                # Remove base files
                for base_path in base_paths[:-2]:
                    remove_if_exists(f'{base_path}.jpg')
                    remove_if_exists(f'{base_path}.txt')
                    remove_if_exists(f'{base_path}.py')

                # Remove files with _fx{i} suffix
                for i in range(code_rerun_num):
                    for suffix in [f'_fx{i}.jpg', f'_fx{i}.txt', f'_fx{i}.py']:
                        remove_if_exists(f'charts/{data_id}{suffix}')
                        remove_if_exists(f'codes/{data_id}{suffix}')

                # Remove _ck files
                for suffix in ['_ck.txt', '_ck.jpg', '_ck.py']:
                    remove_if_exists(f'charts/{data_id}{suffix}')
                    remove_if_exists(f'codes/{data_id}{suffix}')
                
                # Remove description and info files
                for path in base_paths[-2:]:
                    remove_if_exists(path)
        data_info = {str(data_id): data_info[str(data_id)] for data_id in remain_data_ids}
        data_ids = remain_data_ids
    
    # create questions
    questions = []
    for info in data_info.values():
        q_lst = []
        for i in range(4):
            q_str = descriptive_query_helper(info['qids'][i], info['subplot_loc'])
            q_lst.append(f'{i+1}. ' + q_str)
        questions.append("\n".join(q_lst))

    images = [load_image(os.path.join(image_path, f'{d}.jpg')) for d in data_ids]

    # create pipeline
    pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=20480, tp=torch.cuda.device_count(), cache_max_entry_count=0.9))

    # output = {str(data_idx): {"desc": "", "code": "", "qa": ""} for data_idx in data_ids}
    for i in tqdm(range(0, len(data_ids), batch_size)):
        batch_ids = data_ids[i:i+batch_size]
        batch_images = images[i:i+batch_size]
        batch_questions = questions[i:i+batch_size]
        print(f"IDs: {batch_ids[:5]} ... {batch_ids[-5:]}")

        # generate descriptions
        print(f"Generating descriptions...")
        batch_prompts = [(DESCRIPTION_PROMPT, img) for img in batch_images]
        batch_responses = pipe(batch_prompts, gen_config=GenerationConfig(max_new_tokens=8192, do_sample=True, top_p=0.6))
        batch_descriptions =[response.text for response in batch_responses]

        for data_id, description in zip(batch_ids, batch_descriptions):
            with open(f'descriptions/{data_id}.txt', 'w') as f:
                f.write(description)
            # output[str(data_id)]['desc'] = description
        
        batch_prefixes = []
        for data_id, description in zip(batch_ids, batch_descriptions):
            pattern = r'\b(\d+) by (\d+)\b'
            matches = re.findall(pattern, description)
            if not matches:
                batch_prefixes.append('')
                continue
            for match in matches:
                num_rows = int(match[0])
                num_cols = int(match[1])
                break
            p = f"""\n\n**IMPORTANT**\nThe subplots layout in the code should be:
            ```python
            # Subplot Layout ({num_rows} rows, {num_cols} columns)
            fig, ax = plt.subplots({num_rows}, {num_cols}, ...) 
            ```
            where ... stands for the remaining arguments of the `subplots` function.
            """
            batch_prefixes.append(p)
        assert len(batch_prefixes) == len(batch_ids)

        print(f"Generating code...")
        # generate code
        # batch_prompts = [(CODE_PROMPT.format(description), img) for description, img in zip(batch_descriptions, batch_images)]
        batch_prompts = [(CODE_PROMPT.format(description)+prefix) for description, img, prefix in zip(batch_descriptions, batch_images, batch_prefixes)]
        batch_responses = pipe(batch_prompts, gen_config=GenerationConfig(max_new_tokens=8192, do_sample=True, top_p=0.6))
        batch_codes = [response.text for response in batch_responses]
        
        # save code
        for data_id, code in zip(batch_ids, batch_codes):
            code_block = extract_python_code_from_markdown(code)
            # if "matplotlib.use('Agg')" not in code_block:
            #     code_block = code_block.replace("import matplotlib", "import matplotlib\nmatplotlib.use('Agg')")  # use Agg backend to avoid GUI
            if "plt.savefig" in code_block:
                lines = code_block.split('\n')
                filtered_lines = [line for line in lines if "plt.savefig" not in line]
                code_block = "\n".join(code_block.split("\n")[:-1])
            # code_block = code_block.replace("plt.tight_layout()\n", "")
            fig_name = f"charts/{data_id}.jpg"
            if code_block is not None:
                save_extracted_code_to_file(code_block + f"\nplt.savefig('{fig_name}', facecolor='white')", f'codes/{data_id}.py')
        
        print(f"Running code and saving images...")
        # run code and save images
        for data_id in batch_ids:
            run_python_files_and_save_images('codes', str(data_id))
        
        print(f"Regenerating and rerunning code...")
        # 1. code run unsuccessfully
        for debug_num in range(code_rerun_num):
            print(f"Retrying {debug_num+1}")
            batch_errors, batch_codes_with_errors, batch_data_ids_with_errors = [], [], []
            for data_id in batch_ids:
                if os.path.exists(f'charts/{data_id}_fx{debug_num-1}.txt'):
                    with open(f'charts/{data_id}_fx{debug_num-1}.txt', 'r') as f:
                        error_string = f.read()
                    batch_errors.append(error_string)
                    with open(f'codes/{data_id}_fx{debug_num-1}.py', 'r') as f:
                        code_block = f.read()
                    batch_codes_with_errors.append(code_block)
                    batch_data_ids_with_errors.append(data_id)
                elif debug_num == 0 and os.path.exists(f'charts/{data_id}.txt'):
                    with open(f'charts/{data_id}.txt', 'r') as f:
                        error_string = f.read()
                    batch_errors.append(error_string)
                    with open(f'codes/{data_id}.py', 'r') as f:
                        code_block = f.read()
                    batch_codes_with_errors.append(code_block)
                    batch_data_ids_with_errors.append(data_id)
            
            assert len(batch_errors) == len(batch_codes_with_errors) == len(batch_data_ids_with_errors)
            # re-run code
            # batch_prompts = [(CODE_PROMPT.format(batch_descriptions[batch_ids.index(data_idx)]), batch_images[batch_ids.index(data_idx)]) for data_idx, _ in zip(batch_data_ids_with_errors, batch_errors)]
            batch_prompts = [(CODE_PROMPT.format(batch_descriptions[batch_ids.index(data_idx)])+batch_prefixes[batch_ids.index(data_idx)]) for data_idx, _ in zip(batch_data_ids_with_errors, batch_errors)]
            if len(batch_prompts) > 0:
                print(f"Re-generated code...")
                batch_responses = pipe(batch_prompts, gen_config=GenerationConfig(max_new_tokens=8192, do_sample=True, top_p=0.6))
                batch_codes_bug_fixed = [response.text for response in batch_responses]

                print(f"Running re-generated code...")
                # save code
                for data_id, code in zip(batch_data_ids_with_errors, batch_codes_bug_fixed):
                    code_block = extract_python_code_from_markdown(code)
                    # if "matplotlib.use('Agg')" not in code_block:
                    #     code_block = code_block.replace("import matplotlib", "import matplotlib\nmatplotlib.use('Agg')")  # use Agg backend to avoid GUI
                    if "plt.savefig" in code_block:
                        lines = code_block.split('\n')
                        filtered_lines = [line for line in lines if "plt.savefig" not in line]
                        code_block = "\n".join(code_block.split("\n")[:-1])
                    # code_block = code_block.replace("plt.tight_layout()\n", "")
                    fig_name = f"charts/{data_id}_fx{debug_num}.jpg"
                    if code_block is not None:
                        save_extracted_code_to_file(code_block + f"\nplt.savefig('{fig_name}', facecolor='white')", f'codes/{data_id}_fx{debug_num}.py')
                # run code and save images
                for data_id in batch_data_ids_with_errors:
                    run_python_files_and_save_images('codes', f"{data_id}_fx{debug_num}")

        print(f"Collecting running-successfully code and images...")
        # # 2. code run successfully
        batch_codes_to_check, batch_data_ids_to_check, batch_images_to_check = [], [], []
        for data_id in batch_ids:
            if sum((os.path.exists(f'charts/{data_id}_fx{debug_num}.jpg') for debug_num in range(code_rerun_num))) > 0:
                batch_data_ids_to_check.append(data_id)
                for debug_num in range(code_rerun_num-1, -1, -1):  # from 3 to 0
                    if os.path.exists(f'charts/{data_id}_fx{debug_num}.jpg'):
                        with open(f'codes/{data_id}_fx{debug_num}.py', 'r') as f:
                            code_block = f.read()
                        batch_codes_to_check.append(code_block)
                        img = load_image(f'charts/{data_id}_fx{debug_num}.jpg')
                        batch_images_to_check.append(img)
                        break
            elif os.path.exists(f'charts/{data_id}.jpg'):
                batch_data_ids_to_check.append(data_id)
                with open(f'codes/{data_id}.py', 'r') as f:
                    code_block = f.read()
                batch_codes_to_check.append(code_block)
                img = load_image(f'charts/{data_id}.jpg')
                batch_images_to_check.append(img)
        
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
        
        if extract_info:
            print(f"Extracting information from the code...")
            # extract information from the code
            extract_info = []
            for data_id, code in zip(batch_data_ids_to_check, batch_codes_to_check):
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
                print(f"Executing the code {data_id}...")
                try:
                    exec(full_code, {'process_axes': process_axes, 'np': np, 're': re, 'json': json, 'row_inject': row_inject, 'col_inject': col_inject}, local_namespace)
                    info_str += local_namespace.get('output', 'None. Look at the code.')
                    extract_info.append(info_str)
                except Exception as e:
                    extract_info.append(f"None. Look at the code.")

                with open(os.path.join(f'infos/{data_id}.txt'), 'w') as f:
                    f.write(info_str)