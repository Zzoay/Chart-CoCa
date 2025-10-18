import gc
import os
import json
import torch
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from lmdeploy.vl import load_image
from tqdm import tqdm
from constants import *

# Configuration parameters
batch_size = 64  # Reduced batch size since we're using two GPUs
repeat_sample_num = 5  # Number of answers to generate per question
dataset_path = '../CharXiv/'
info_path = os.path.join(dataset_path, 'data')
image_path = os.path.join(dataset_path, 'images')
mul_ans_save_dir = os.path.join(dataset_path, "results/mul-internvl")
final_ans_save_dir = os.path.join(dataset_path, "results/final")

 # or download from huggingface first
generator_model = 'OpenGVLab/InternVL2-8B' 
selector_model = 'yaozz/Chart-Answer-Selector' 
# generator_model = 'ckpt/InternVL2-8B' 
# selector_model = 'ckpt/InternVL2-8B-Selector'  


model_suffix = generator_model.split('/')[-1]


# Create output directories if they don't exist
os.makedirs(mul_ans_save_dir, exist_ok=True)
os.makedirs(final_ans_save_dir, exist_ok=True)

# Base instructions
generator_instruction = """<IMAGE_TOKEN>\nGiven a chart and a question, your task is to provide the correct answer based on the information in the chart.
"""

selector_instruction = """<IMAGE_TOKEN>\nGiven a chart, a question, and several answers which may contain the correct one, your task is: 
If there is a correct response, output it. If not, generate the correct answer.
"""

def descriptive_query_helper(qid, subplot_loc):
    """Helper function to generate descriptive task queries"""
    if qid in [18, 19]:
        return DESCRIPTIVE_RESP_INST[qid] 
    if isinstance(subplot_loc, list):
        if subplot_loc[0] == 0:
            prefix = "For the current plot, "
        else:
            prefix = f"For the subplot at row {subplot_loc[0]} and column {subplot_loc[1]}, "
    elif isinstance(subplot_loc, str):
        prefix = f"For {subplot_loc}, "
    else:
        raise ValueError(f"Invalid subplot_loc: {subplot_loc}")
    return DESCRIPTIVE_RESP_INST[qid].format(prefix)

def get_number_instruction(answer):
    """Generate instruction for number-specific questions"""
    base = answer.split('.')
    whole, decimal = base[0], None if len(base) == 1 else base[1]
    if whole is not None and decimal is None:
        return "* Your final answer must be an exact integer."
    elif whole is not None and decimal is not None:
        num_decimal = len(decimal)
        return f"* Your final answer must be a number with {num_decimal} decimal places."
    else:
        raise ValueError(f"Invalid answer: {answer}")

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    for device_id in range(torch.cuda.device_count()):
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()
    print("GPU memory cleared successfully")

def check_candidate_answers(task_type):
    """Check if there are enough candidate answer"""
    file_pattern = f"gen-InternVL2-8B-{task_type}_val-"
    candidate_files = [
        f for f in os.listdir(mul_ans_save_dir) 
        if f.startswith(file_pattern) and f.endswith(".json")
    ]
    
    if len(candidate_files) >= repeat_sample_num:
        valid_files = 0
        for f in candidate_files[:repeat_sample_num]:
            file_path = os.path.join(mul_ans_save_dir, f)
            if os.path.getsize(file_path) > 0:
                valid_files += 1
        return valid_files >= repeat_sample_num
    return False

def load_candidate_answers(task_type):
    """Load existing candidate answers"""
    file_pattern = f"gen-InternVL2-8B-{task_type}_val-"
    candidate_files = [
        f for f in os.listdir(mul_ans_save_dir) 
        if f.startswith(file_pattern) and f.endswith(".json")
    ]
    candidate_files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
    ans_data = []
    for f in candidate_files[:repeat_sample_num]:
        file_path = os.path.join(mul_ans_save_dir, f)
        ans_data.append(json.load(open(file_path, 'r')))
    return ans_data

def process_descriptive_tasks():
    """Process descriptive tasks: generate multiple answers then final answer"""
    print("Processing descriptive tasks...")
    
    # Load data
    data_info = json.load(open(os.path.join(info_path, f'descriptive_val.json')))
    data_ids = [int(d) for d in data_info]
    images = [load_image(os.path.join(image_path, f'{d}.jpg')) for d in data_ids]
    
    if not check_candidate_answers('descriptive'):
        # Initialize generator pipeline on GPU 0
        generator_pipe = pipeline(generator_model, backend_config=TurbomindEngineConfig(
            session_len=20480, 
            tp=2,  
            device_id=[0,1],
            cache_max_entry_count=0.9
        ))
        
        # Generate multiple answers, the current lmdeploy does not seem to allow n >= 1, so we repeat the generation n times.
        ans_data = []
        for repeat_i in tqdm(range(repeat_sample_num), desc="Generating multiple answers"):
            ret = {}
            queries = []
            
            # Prepare queries
            for d_id in data_info:
                item = data_info[d_id]
                subplot_loc = item['subplot_loc']
                for idx, qid in enumerate(item['qids']):
                    question = descriptive_query_helper(qid, subplot_loc)
                    ret[f"{d_id}_{idx}"] = {
                        "figure_id": item['figure_id'], 
                        "subq_idx": idx, 
                        "qid": qid
                    }
                    qst_str = "\nQuestion:\n" + question
                    queries.append(generator_instruction + qst_str)
            
            # Process in batches with generator model
            for i in tqdm(range(0, len(data_ids), batch_size), desc=f"Generator batch {repeat_i+1}"):
                batch_ids = data_ids[i:i+batch_size]
                batch_images = images[i:i+batch_size]
                batch_queries = []
                
                for idx in range(i, min(i + batch_size, len(data_ids))):
                    start_idx = idx * 4
                    batch_queries.extend(queries[start_idx:start_idx + 4])
                
                expanded_images = [img for img in batch_images for _ in range(4)][:len(batch_queries)]
                batch_prompts = [(query, img) for query, img in zip(batch_queries, expanded_images)]
                
                # Generate answers with sampling
                batch_responses = generator_pipe(
                    batch_prompts, 
                    gen_config=GenerationConfig(
                        max_new_tokens=512, 
                        do_sample=True,
                        top_p=0.6,
                        temperature=0.7
                    )
                )
                
                # Save results
                batch_responses_text = [response.text for response in batch_responses]
                for j, d_id in enumerate(batch_ids):
                    for idx in range(4):
                        key = f"{d_id}_{idx}"
                        if key in ret:
                            ret[key]["response"] = batch_responses_text[j*4 + idx]
            
            # Save and store generated answers
            output_path = os.path.join(mul_ans_save_dir, f"gen-InternVL2-8B-descriptive_val-{repeat_i}.json")
            with open(output_path, 'w') as f:
                json.dump(ret, f, indent=4, ensure_ascii=False)
            ans_data.append(ret)

        del generator_pipe
        clear_gpu_memory()
    else:
        print("Using existing descriptive candidate answers...")
        ans_data = load_candidate_answers("descriptive")
    
    selector_pipe = pipeline(selector_model, backend_config=TurbomindEngineConfig(
        session_len=20480, 
        tp=2,  
        device_id=[0,1], 
        cache_max_entry_count=0.9
    ))
    
    # Generate final answers
    print("Generating final answers for descriptive tasks...")
    ret = {}
    queries = []
    for d_id in data_info:
        item = data_info[d_id]
        subplot_loc = item['subplot_loc']
        for idx, qid in enumerate(item['qids']):
            question = descriptive_query_helper(qid, subplot_loc)
            ret[f"{d_id}_{idx}"] = {"figure_id": item['figure_id'], "subq_idx": idx, "qid": qid}
            
            # Collect multiple responses
            mul_responses = [ans_data[repeat_i][f"{d_id}_{idx}"]["response"] 
                            for repeat_i in range(repeat_sample_num)]
            mul_responses_str = "\n".join(f"Response {i+1}:\n{x}\n" for i,x in enumerate(mul_responses))
            qst_str = "\nQuestion:\n" + question + "\nResponses:\n\n" + mul_responses_str
            queries.append(selector_instruction + qst_str)
    
    # Process in batches with selector model
    for i in tqdm(range(0, len(data_ids), batch_size), desc="Selector processing"):
        batch_ids = data_ids[i:i+batch_size]
        batch_images = images[i:i+batch_size]
        batch_queries = []
        
        for idx in range(i, min(i + batch_size, len(data_ids))):
            start_idx = idx * 4
            batch_queries.extend(queries[start_idx:start_idx + 4])
        
        expanded_images = [img for img in batch_images for _ in range(4)][:len(batch_queries)]
        batch_prompts = [(query, img) for query, img in zip(batch_queries, expanded_images)]
        
        # Generate final answers
        batch_responses = selector_pipe(
            batch_prompts, 
            gen_config=GenerationConfig(
                max_new_tokens=512, 
                do_sample=True,
                top_p=0.6
            )
        )
        
        # Save results
        batch_critics = [response.text for response in batch_responses]
        for j, d_id in enumerate(batch_ids):
            for idx in range(4):
                key = f"{d_id}_{idx}"
                if key in ret:
                    ret[key]["response"] = batch_critics[j*4 + idx]
    
    # Save final results
    with open(os.path.join(final_ans_save_dir, f"results/final/gen-InternVL2-8B-descriptive_val.json"), 'w') as f:
        json.dump(ret, f, indent=4, ensure_ascii=False)

def process_reasoning_tasks():
    """Process reasoning tasks: generate multiple answers then final answer"""
    print("Processing reasoning tasks...")
    
    # Load data
    data_info = json.load(open(os.path.join(info_path, f'reasoning_val.json')))
    data_ids = [int(d) for d in data_info]
    images = [load_image(os.path.join(image_path, f'{d}.jpg')) for d in data_ids]
    
    if not check_candidate_answers('reasoning'):
        # Initialize generator pipeline
        generator_pipe = pipeline(generator_model, backend_config=TurbomindEngineConfig(
            session_len=20480, 
            tp=2,  
            device_id=[0, 1],  
            cache_max_entry_count=0.9
        ))
        
        # Generate multiple answers
        ans_data = []
        for repeat_i in tqdm(range(repeat_sample_num), desc="Generating multiple answers"):
            ret = {}
            queries = []
            
            # Prepare queries
            for d_id in data_info:
                item = data_info[d_id]
                inst_category = item['inst_category']
                
                if inst_category in [1, 2, 3]:
                    question = REASONING_RESP_INST[inst_category].format(item['query'])
                elif inst_category == 4:
                    question = REASONING_RESP_INST[inst_category].format(
                        item['query'], 
                        get_number_instruction(item['answer'])
                    )
                
                ret[f"{d_id}"] = {
                    "figure_id": item['figure_id'], 
                    "inst_category": item['inst_category'], 
                    "raw_question": item['query']
                }
                
                qst_str = "\nQuestion:\n" + question
                queries.append(generator_instruction + qst_str)
            
            # Process in batches with generator model
            for i in tqdm(range(0, len(data_ids), batch_size), desc=f"Generator batch {repeat_i+1}"):
                batch_ids = data_ids[i:i+batch_size]
                batch_images = images[i:i+batch_size]
                batch_queries = queries[i:i+batch_size]
                
                batch_prompts = [(query, img) for query, img in zip(batch_queries, batch_images)]
                
                # Generate answers with sampling
                batch_responses = generator_pipe(
                    batch_prompts, 
                    gen_config=GenerationConfig(
                        max_new_tokens=512, 
                        do_sample=True,
                        top_p=0.6,
                        temperature=0.7
                    )
                )
                
                # Save results
                batch_responses_text = [response.text for response in batch_responses]
                for j, d_id in enumerate(batch_ids):
                    ret[f"{d_id}"]["response"] = batch_responses_text[j]
            
            # Save and store generated answers
            output_path = os.path.join(mul_ans_save_dir, f"gen-{model_suffix}-reasoning_val-{repeat_i}.json")
            with open(output_path, 'w') as f:
                json.dump(ret, f, indent=4, ensure_ascii=False)
            ans_data.append(ret)

        del generator_pipe
        clear_gpu_memory()
    else:
        print("Using existing reasoning candidate answers...")
        ans_data = load_candidate_answers("reasoning")
    
    # Initialize selector pipeline
    selector_pipe = pipeline(selector_model, backend_config=TurbomindEngineConfig(
        session_len=20480, 
        tp=2,  
        device_id=[0, 1],  
        cache_max_entry_count=0.9
    ))
    
    # Generate final answers using selector model
    print("Generating final answers for reasoning tasks...")
    ret = {}
    queries = []
    for d_id in data_info:
        item = data_info[d_id]
        inst_category = item['inst_category']
        
        if inst_category in [1, 2, 3]:
            question = REASONING_RESP_INST[inst_category].format(item['query'])
        elif inst_category == 4:
            question = REASONING_RESP_INST[inst_category].format(
                item['query'], 
                get_number_instruction(item['answer'])
            )
        
        ret[f"{d_id}"] = {"figure_id": item['figure_id'], "inst_category": item['inst_category'], "raw_question": item['query']}
        
        # Collect multiple responses
        mul_responses = [ans_data[repeat_i][f"{d_id}"]["response"] 
                        for repeat_i in range(repeat_sample_num)]
        mul_responses_str = "\n".join(f"Response {i+1}:\n{x}\n" for i,x in enumerate(mul_responses))
        qst_str = "\nQuestion:\n" + question + "\nResponses:\n\n" + mul_responses_str
        queries.append(selector_instruction + qst_str)
    
    # Process in batches with selector model
    for i in tqdm(range(0, len(data_ids), batch_size), desc="Selector processing"):
        batch_ids = data_ids[i:i+batch_size]
        batch_images = images[i:i+batch_size]
        batch_queries = queries[i:i+batch_size]
        
        batch_prompts = [(query, img) for query, img in zip(batch_queries, batch_images)]
        
        # Generate final answers
        batch_responses = selector_pipe(
            batch_prompts, 
            gen_config=GenerationConfig(
                max_new_tokens=512, 
                do_sample=True,
                top_p=0.6
            )
        )
        
        # Save results
        batch_critics = [response.text for response in batch_responses]
        for j, d_id in enumerate(batch_ids):
            ret[f"{d_id}"]["response"] = batch_critics[j]
    
    # Save final results
    with open(os.path.join(final_ans_save_dir, f"gen-{model_suffix}-reasoning_val.json"), 'w') as f:
        json.dump(ret, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    print(f"Using {torch.cuda.device_count()} GPUs for processing")
    process_descriptive_tasks()
    process_reasoning_tasks()
    print("All tasks completed successfully")
