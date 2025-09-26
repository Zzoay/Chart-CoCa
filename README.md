# Chart-CoCa
This repository contains the official implementation of our CIKM 2025 paper:
"Chart-CoCa: Self-Improving Chart Understanding of Vision LMs via Code-Driven Synthesis and Candidate-Conditioned Answering"

Chart-CoCa centers around chart understanding and employs code as an intermediary for chart synthesis and candidate-conditioned answering, enabling Vision Language Models (VLMs) to enhance their own capabilities in comprehending charts.

## Data Preparation

Download the CharXiv dataset. This dataset is of great significance as it provides data sources for synthetic data generation and model evaluation.

Clone the `internvl` official repository. This repository contains essential code and resources that will be used in the later steps, especially when it comes to fine-tuning the model.

## Requirments
- lmdeploy
- vllm
- tqdm
- matplotlib
- numpy
- cv2

## Usage Steps

To generate chart descriptions, codes, and synthesize charts, run the following command:



```
python gen_code.py
```

This step is crucial as it lays the foundation for subsequent processes by creating the necessary components related to the charts.

Next, extract chart information using:



```
python gen_info.py
```

The information obtained here will be used in further analysis and operations.

To get candidates, execute:



```
python gen_mul_qa.py
```

This helps in gathering potential options or answers that will be refined in the subsequent steps.

Generate fine-tuning data by running:



```
python build_critic_data.py
```

The fine-tuning data is essential for optimizing the model's performance.

You can fine-tune the answer model either through `internvl` official code or `llamafactory`. Follow their respective documentation and procedures to complete this step.

Next, obtain the results by running:



```
python predict.py
```

Please note that each step is sequential and builds upon the previous one. Make sure there are no errors in each stage to ensure accurate results.


Finally, use the official code in CharXiv to evaluate the model predictions.



## Citation
Once published, please cite our work using the official CIKM 2025 proceedings reference.
```bib
@article{jiang2025chartcoca,
      title={Chart-CoCa: Self-Improving Chart Understanding of Vision LMs via Code-Driven Synthesis and Candidate-Conditioned Answering}, 
      author={Gongyao Jiang and Qiong Luo},
      year={2025},
      eprint={2508.11975},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2508.11975}, 
}
```