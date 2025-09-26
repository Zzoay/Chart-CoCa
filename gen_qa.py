
import os

import math
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
code_rerun_num = 4
dataset_path = '../CharXiv/'
info_path = os.path.join(dataset_path, 'data')
image_path = os.path.join(dataset_path, 'images')
model = '/home/share/ckpt/InternVL2-8B'


def convert_to_number(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s

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
    weights = [1 / math.log(i + 2) for i in range(len(numbers))]
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

    for i in tqdm(range(0, len(data_ids), batch_size)):
        batch_ids = data_ids[i:i+batch_size]
        batch_questions = questions[i:i+batch_size]
        print(f"IDs: {batch_ids[:5]} ... {batch_ids[-5:]}")

        batch_codes_to_check, batch_data_ids_to_check = [], []
        for data_id in batch_ids:
            if os.path.exists(f'charts/{data_id}_aj.jpg'):
                batch_data_ids_to_check.append(data_id)
                if os.path.exists(f'charts/{data_id}_ck.jpg'):
                    with open(f'codes/{data_id}_ck.py', 'r') as f:
                        code_block = f.read()
                    batch_codes_to_check.append(code_block)
                elif sum((os.path.exists(f'charts/{data_id}_fx{debug_num}.jpg') for debug_num in range(code_rerun_num))) > 0:
                    for debug_num in range(code_rerun_num-1, -1, -1):  # from 3 to 0
                        if os.path.exists(f'charts/{data_id}_fx{debug_num}.jpg'):
                            with open(f'codes/{data_id}_fx{debug_num}.py', 'r') as f:
                                code_block = f.read()
                            batch_codes_to_check.append(code_block)
                            break
                elif os.path.exists(f'charts/{data_id}.jpg'):
                    with open(f'codes/{data_id}.py', 'r') as f:
                        code_block = f.read()
                    batch_codes_to_check.append(code_block)
            # elif os.path.exists(f'charts/{data_id}_ck.jpg'):
            #     batch_data_ids_to_check.append(data_id)
            #     with open(f'codes/{data_id}_ck.py', 'r') as f:
            #         code_block = f.read()
            #     batch_codes_to_check.append(code_block)
            # elif sum((os.path.exists(f'charts/{data_id}_fx{debug_num}.jpg') for debug_num in range(code_rerun_num))) > 0:
            #     batch_data_ids_to_check.append(data_id)
            #     for debug_num in range(code_rerun_num-1, -1, -1):  # from 3 to 0
            #         if os.path.exists(f'charts/{data_id}_fx{debug_num}.jpg'):
            #             with open(f'codes/{data_id}_fx{debug_num}.py', 'r') as f:
            #                 code_block = f.read()
            #             batch_codes_to_check.append(code_block)
            #             break
            # elif os.path.exists(f'charts/{data_id}.jpg'):
            #     batch_data_ids_to_check.append(data_id)
            #     with open(f'codes/{data_id}.py', 'r') as f:
            #         code_block = f.read()
            #     batch_codes_to_check.append(code_block)
        assert len(batch_data_ids_to_check) == len(batch_codes_to_check)
        
        extract_info = []
        for data_id in batch_data_ids_to_check:
            if os.path.exists(f'infos/{data_id}.txt'):
                with open(os.path.join(f'infos/{data_id}.txt'), 'r') as f:
                    info = f.read()
                extract_info.append(info)
            else:
                extract_info.append("None.")

        # explicit answers
        direct_answers = []
        for data_id, info, code in zip(batch_data_ids_to_check, extract_info, batch_codes_to_check):
            question = batch_questions[batch_ids.index(data_id)]
            
            info_split = info.split('\n')
            info_dict = {}
            num_rows, num_cols, num_total = 1, 1, 1
            if len(info_split) > 1:
                numbers = re.findall(r'\d+', info_split[0])
                num_rows, num_cols, num_total = int(numbers[0]), int(numbers[1]), int(numbers[2])

            if len(info_split) > 3:
                # info_split = info_split[2:]
                info_str = "\n".join(info_split[2:])
                info_dict = json.loads(info_str)

            if info_dict == {}:
                continue
            
            num_plots_actual = len(info_dict)
            loc_in_info = list(info_dict.keys())[-1]
            if loc_in_info == "the current plot":
                num_rows_actual = 1
                num_cols_actual = 1
            else:
                num_rows_actual = int(loc_in_info.split(' ')[loc_in_info.split(' ').index('row')+1])
                num_cols_actual = int(loc_in_info.split(' ')[loc_in_info.split(' ').index('column')+1])
            
            # subquestions
            q_list = []
            q_list.append(question[question.find('1.')+2:question.find('2.')].strip())
            q_list.append(question[question.find('2.')+2:question.find('3.')].strip())
            q_list.append(question[question.find('3.')+2:question.find('4.')].strip())
            q_list.append(question[question.find('4.')+2:].strip())

            for q_idx, q in enumerate(q_list):
                if "the subplot at row 1 and column 1" in q and num_cols == 1 and num_rows == 1:
                    q_list[q_idx] = q.replace("the subplot at row 1 and column 1", "the current plot")
                elif "the current plot" not in q and "layout" not in q and "number of subplots" not in q:
                    q_split = q.split(",")
                    q_split[0] ="For the subplot at row 1 and column 1"  # first replace with “1”
                    q_list[q_idx] = ",".join(q_split)
            
            tmp_ans = ["", "", "", ""]
            for q_idx, q in enumerate(q_list):
                # randomly choose a subplot location
                row_in_question, col_in_question = weighted_random_choice(1, num_rows_actual), weighted_random_choice(1, num_cols_actual)
                loc = f"the subplot at row {row_in_question} and column {col_in_question}"
                if "the current plot" not in q and "layout" not in q and "number of subplots" not in q:
                    q_split = q.split(" ")
                    q_split[q_split.index("row") + 1] = str(row_in_question)
                    q_split[q_split.index("column") + 1] = str(col_in_question) + ','
                    q = " ".join(q_split)
                    q_list[q_idx] = " ".join(q_split)

                specific_info = info_dict.get(loc, {})
                if specific_info == {} and col_in_question == 1 and row_in_question == 1:
                    specific_info = info_dict.get("the current plot", {})
                
                if specific_info == {}:
                    continue
                
                x_tick_labels = specific_info.get('x_tick_labels', [])
                if len(x_tick_labels) > 0:
                    if isinstance(x_tick_labels[0], str) and bool(re.match(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$', x_tick_labels[0])):
                        x_tick_labels = [convert_to_number(x) for x in x_tick_labels]

                y_tick_labels = specific_info.get('y_tick_labels', [])
                if len(y_tick_labels) > 0:
                    if isinstance(y_tick_labels[0], str) and bool(re.match(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$', y_tick_labels[0])):
                        y_tick_labels = [convert_to_number(y) for y in y_tick_labels]
                
                z_tick_labels = specific_info.get('z_tick_labels', [])
                if len(z_tick_labels) > 0:
                    if isinstance(z_tick_labels[0], str) and bool(re.match(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$', z_tick_labels[0])):
                        z_tick_labels = [convert_to_number(z) for z in z_tick_labels]
                
                if len(x_tick_labels) > 8:
                    x_tick_labels = x_tick_labels[:3] + ['...'] + x_tick_labels[-3:]
                    x_tick_labels = '[' + ', '.join(str(x) for x in x_tick_labels).replace("'...'", "...") + ']'
                if len(y_tick_labels) > 8:
                    y_tick_labels = y_tick_labels[:3] + ['...'] + y_tick_labels[-3:]
                    y_tick_labels = '[' + ', '.join(str(y) for y in y_tick_labels).replace("'...'", "...") + ']'
                if len(z_tick_labels) > 8:
                    z_tick_labels = z_tick_labels[:3] + ['...'] + z_tick_labels[-3:]
                    z_tick_labels = '[' + ', '.join(str(z) for z in z_tick_labels).replace("'...'", "...") + ']'

                labels_in_legend = specific_info.get('labels_in_legend', [])
                if labels_in_legend != "Not Applicable" and isinstance(labels_in_legend, list) and len(labels_in_legend) > 0:
                    if not isinstance(labels_in_legend[0], str):
                        labels_in_legend = [str(x) if 'e' not in str(x) else f"{float(x):.2e}" for x in labels_in_legend]
                    # else:
                    #     labels_in_legend = ["'" + x + "'" for x in labels_in_legend]
                
                # if num_rows == 1 and num_cols == 1 and loc == "the subplot at row 1 and column 1":
                #     loc = "the current plot"

                # rows
                if num_rows == 1 and num_cols != 1:
                    if random.random() < 0.3:
                        if int(col_in_question) == 1:
                            loc = "the leftmost subplot"
                            q_split = q.split(",")
                            q_split[0] = "For the leftmost subplot"
                            q = ",".join(q_split)
                            q_list[q_idx] = ",".join(q_split)
                        elif int(col_in_question) == num_cols:
                            loc = "the rightmost subplot"
                            q_split = q.split(",")
                            q_split[0] = "For the rightmost subplot"
                            q = ",".join(q_split)
                            q_list[q_idx] = ",".join(q_split)
                elif num_rows != 1 and num_cols == 1:
                    if random.random() < 0.3:
                        if int(row_in_question) == 1:
                            loc = "the top subplot"
                            q_split = q.split(",")
                            q_split[0] = "For the top subplot"
                            q = ",".join(q_split)
                            q_list[q_idx] = ",".join(q_split)
                        elif int(row_in_question) == num_rows:
                            loc = "the bottom subplot"
                            q_split = q.split(",")
                            q_split[0] = "For the bottom subplot"
                            q = ",".join(q_split)
                            q_list[q_idx] = ",".join(q_split)
                elif num_rows != 1 and num_cols != 1:
                    if random.random() < 0.3:
                        if int(row_in_question) == 1 and int(col_in_question) == 1:
                            loc = "the top left subplot"
                            q_split = q.split(",")
                            q_split[0] = "For the top left subplot"
                            q = ",".join(q_split)
                            q_list[q_idx] = ",".join(q_split)
                        elif int(row_in_question) == num_rows and int(col_in_question) == 1:
                            loc = "the bottom left subplot"
                            q_split = q.split(",")
                            q_split[0] = "For the bottom left subplot"
                            q = ",".join(q_split)
                            q_list[q_idx] = ",".join(q_split)
                        elif int(row_in_question) == 1 and int(col_in_question) == num_cols:
                            loc = "the top right subplot"
                            q_split = q.split(",")
                            q_split[0] = "For the top right subplot"
                            q = ",".join(q_split)
                            q_list[q_idx] = ",".join(q_split)
                        elif int(row_in_question) == num_rows and int(col_in_question) == num_cols:
                            loc = "the bottom right subplot"
                            q_split = q.split(",")
                            q_split[0] = "For the bottom right subplot"
                            q = ",".join(q_split)  
                            q_list[q_idx] = ",".join(q_split)
                    
                if "title" in q:
                    tmp_ans[q_list.index(q)] += f"The title is: {specific_info['title']}" if len(specific_info['title']) > 1 and str(specific_info['title']) not in ['Not Applicable', 'None'] else "Not Applicable"
                elif "label of the x-axis" in q:
                    label_of_xaxis = specific_info['label_of_xaxis']
                    tmp_ans[q_list.index(q)] += f"The label of the x-axis is: {label_of_xaxis}" if len(label_of_xaxis) > 1 and str(label_of_xaxis) not in ['Not Applicable', 'None'] else "Not Applicable"
                elif "label of the y-axis" in q:
                    label_of_yaxis = specific_info['label_of_yaxis']
                    tmp_ans[q_list.index(q)] += f"The label of the y-axis is: {label_of_yaxis}" if len(label_of_yaxis) > 1 and str(label_of_yaxis) not in ['Not Applicable', 'None'] else "Not Applicable"
                elif "leftmost labeled tick on the x-axis" in q:
                    x_tick_leftmost = str(specific_info['x_tick_leftmost'])
                    tmp_ans[q_list.index(q)] += f"The leftmost labeled tick on the x-axis is: {x_tick_leftmost}." if x_tick_leftmost != "Not Applicable" else "Not Applicable"
                elif "rightmost labeled tick on the x-axis" in q:
                    x_tick_rightmost = str(specific_info['x_tick_rightmost'])
                    tmp_ans[q_list.index(q)] += f"The rightmost labeled tick on the x-axis is: {x_tick_rightmost}." if x_tick_rightmost != "Not Applicable" else "Not Applicable"
                elif "highest labeled tick on the y-axis" in q:
                    y_tick_highest = str(specific_info['y_tick_highest'])
                    tmp_ans[q_list.index(q)] += f"The highest labeled tick on the y-axis is: {y_tick_highest}." if y_tick_highest != "Not Applicable" else "Not Applicable"
                elif "lowest labeled tick on the y-axis" in q:
                    y_tick_lowest = str(specific_info['y_tick_lowest'])
                    tmp_ans[q_list.index(q)] += f"The lowest labeled tick on the y-axis is: {y_tick_lowest}." if y_tick_lowest != "Not Applicable" else "Not Applicable"
                elif "difference between consecutive numerical tick values on the x-axis" in q:
                    x_tick_difference = str(specific_info['x_tick_difference'])
                    tmp_ans[q_list.index(q)] += f"X tick labels are: {x_tick_labels}. Thus, the difference between numerical tick values on the x-axis is: {x_tick_difference}." if x_tick_difference != "Not Applicable" else "There is no numerical tick labels on the x-axis. Thus, the answer is 'Not Applicable'."
                elif "difference between consecutive numerical tick values on the y-axis" in q:
                    y_tick_difference = str(specific_info['y_tick_difference'])
                    tmp_ans[q_list.index(q)] += f"Y tick labels are: {y_tick_labels}. Thus, the difference between numerical tick values on the y-axis is: {y_tick_difference}." if y_tick_difference != "Not Applicable" else "There is no numerical tick labels on the y-axis. Thus, the answer is 'Not Applicable'."
                elif "discrete labels are there in the legend" in q:
                    num_of_label_in_legend = len(specific_info['labels_in_legend'])
                    tmp_ans[q_list.index(q)] += f"The labels in the legend are: {', '.join(labels_in_legend)}, {num_of_label_in_legend} in total. Thus, there are {num_of_label_in_legend} labels in the legend." if isinstance(labels_in_legend, list) and num_of_label_in_legend > 0 else "Not Applicable"
                elif "the names of the labels in the legend" in q:
                    tmp_ans[q_list.index(q)] += f"The labels in the legend are: {labels_in_legend}." if isinstance(labels_in_legend, list) and len(labels_in_legend) > 0 else "Not Applicable"
                elif "total number of explicitly labeled ticks across all axes" in q:
                    ticks_num = str(specific_info['num_ticks_all_axes'])
                    axes_num = 2 if specific_info['num_ticks_zaxis'] == 0 else 3
                    tmp_ans[q_list.index(q)] += f"Fist, determine axes num: There are {axes_num} axes. "
                    if axes_num == 2:
                        tmp_ans[q_list.index(q)] += f"Then, determine number of ticks: For all axes, there are {x_tick_labels} ({specific_info['num_ticks_xaxis']} in total) and {y_tick_labels} ({specific_info['num_ticks_yaxis']} in total) explicitly labeled ticks. In total, there are {ticks_num} explicitly labeled ticks across all axes." if specific_info['num_ticks_xaxis'] > 0  or specific_info['num_ticks_yaxis'] > 0 else "Not Applicable"
                    else:
                        tmp_ans[q_list.index(q)] += f"Then, determine number of ticks: For all axes, there are {x_tick_labels} ({specific_info['num_ticks_xaxis']} in total), {y_tick_labels} ({specific_info['num_ticks_yaxis']} in total), and {specific_info['z_tick_labels']} ({specific_info['num_ticks_zaxis']} in total) explicitly labeled ticks. In total, there are {ticks_num} explicitly labeled ticks across all axes." if specific_info['num_ticks_xaxis'] > 0  or specific_info['num_ticks_yaxis'] > 0 or specific_info['num_ticks_zaxis'] > 0 else "Not Applicable"
                elif "layout of the subplots" in q:
                    # if num_plots_actual < num_total:
                    #     tmp_ans[q_list.index(q)] += ""  # ignore this question
                    if num_rows == 1 and num_cols == 1:
                        tmp_ans[q_list.index(q)] += f"The plot does not contain subplots. Thus, the layout is 1 by 1."
                    else:
                        tmp_ans[q_list.index(q)] += f"The layout of subplots is {num_rows} by {num_cols}."
                elif "the number of subplots" in q:
                    # if num_plots_actual < num_total:
                    #     tmp_ans[q_list.index(q)] += ""  # ignore this question
                    if num_rows == 1 and num_cols == 1:
                        tmp_ans[q_list.index(q)] += f"The plot does not contain subplots. Thus, the number of subplots is 1."
                    else:
                        tmp_ans[q_list.index(q)] += f"There are {num_total} subplots in the plot."
                elif "colorbar" in q:
                    # default
                    if "difference between the maximum and minimum values" in q and specific_info['colorbar_diff'] != "Not Applicable":
                        colorbar_diff = str(specific_info['colorbar_diff'])
                        colorbar_min = specific_info['colorbar_max_value'] - specific_info['colorbar_diff']
                        tmp_ans[q_list.index(q)] += f"The colorbar range is from {colorbar_min} to {specific_info['colorbar_max_value']}. Thus, the difference between the maximum and minimum values is: {colorbar_diff}."
                    elif "maximum value of the tick labels" in q and specific_info['colorbar_max_value'] != "Not Applicable":
                        colorbar_max = str(specific_info['colorbar_max_value'])
                        tmp_ans[q_list.index(q)] += f"The maximum value of the tick labels in the colorbar is: {colorbar_max}." if colorbar_max != "Not Applicable" else "Not Applicable"
                    else:
                        tmp_ans[q_list.index(q)] += "There is no colorbar in this plot. Thus, the answer is 'Not Applicable.'"
                elif "how many lines" in q:
                    if specific_info['num_lines'] == "Not Applicable":
                        tmp_ans[q_list.index(q)] += "Not Applicable"
                    else:
                        tmp_ans[q_list.index(q)] += f"There are {specific_info['num_lines']} lines."
                elif "any lines intersect" in q:
                    if specific_info['intersection_of_lines'] == "Not Applicable":
                        tmp_ans[q_list.index(q)] += "Not Applicable"
                    else:
                        tmp_ans[q_list.index(q)] += f"Yes, there are lines that intersect." if specific_info['intersection_of_lines'] == 'Yes' else "No, there are no lines that intersect."
                elif "general trend of data from left to right" in q:
                    if specific_info['trend_of_lines'] == "Not Applicable":
                        tmp_ans[q_list.index(q)] += "Not Applicable"
                    else:
                        tmp_ans[q_list.index(q)] += f"The general trend of data from left to right is: {specific_info['trend_of_lines']}."

            direct_answers.append("\n".join(tmp_ans))

            with open(os.path.join(f'direct_answers/{data_id}.jsonl'), 'w') as f:
                for q, ans in zip(q_list, tmp_ans):
                    json.dump({'question': q, 'answer': ans}, f, ensure_ascii=False)
                    f.write('\n')
        
        # assert len(batch_data_ids_to_check) == len(direct_answers)
        # for direct_ans, data_id in zip(direct_answers, batch_data_ids_to_check):
        #     with open(os.path.join(f'direct_answers/{data_id}.txt'), 'w') as f:
        #         f.write(direct_ans)