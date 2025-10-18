# Chart-CoCa
This repository contains the implementation of our CIKM 2025 paper:
[Chart-CoCa: Self-Improving Chart Understanding of Vision LMs via Code-Driven Synthesis and Candidate-Conditioned Answering](https://arxiv.org/abs/2508.11975)

Chart-CoCa centers around chart understanding and employs code as an intermediary for chart synthesis and candidate-conditioned answering, enabling Vision Language Models (VLMs) to enhance their own capabilities in comprehending charts.


## Requirments
- lmdeploy
- vllm
- tqdm
- matplotlib
- numpy
- cv2

## Data Preparation

Download the CharXiv dataset. This dataset is of great significance as it provides data sources for synthetic data generation and model evaluation.

Clone the `internvl` official repository. This repository contains essential code and resources that will be used in the later steps, especially when it comes to fine-tuning the model.

## Inference
Obtain the results by directly running:

```
python predict.py
```
The code first generates answer candidates using the base model, then generates the final answer based on these candidates.
This will result in prediction data compatible with the CharXiv evaluation code.

The candidate generation model is OpenGVLab/InternVL2-8B, and the final answer generation model is yaozz/Chart-Answer-Selector, which was trained on our synthetic data. Our final answer generation model was uploaded to Huggingface.


Finally, use the official code in CharXiv to evaluate the model predictions.


## Data Synthesis and Training

To generate chart descriptions, codes, and synthesize charts, run the following command:



```
python gen_code.py
```

This step is crucial as it lays the foundation for subsequent processes by creating the necessary components related to the charts. Meanwhile, the chart information is extracted.

Next, extract chart information using:



To get answer candidates for synthetic charts, execute:
```
python gen_mul_qa.py
```

This generates multiple answers for fine-tuning.

Next, collect and process fine-tuning data by running:



```
python build_ft_data.py
```



Then, you can fine-tune the answer model either through `internvl` official code or `llamafactory`. Follow their respective documentation and procedures to complete this step.





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
