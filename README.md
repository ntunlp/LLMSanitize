# LLMSanitize
An open-source library for contamination detection in NLP datasets and Large Language Models (LLMs).  

## Installation
The library has been designed and tested with **Python 3.9** and **CUDA 11.8**.  

First make sure you have CUDA 11.8 installed, and create a conda environment with Python 3.9: 
```bash
conda create -name llmsanitize python=3.9
```

Next activate the environment:
```bash
conda activate llmsanitize
```

Then install all the dependencies for LLMSanitize:
```bash
pip install -r requirements.txt
```

Alternatively, you can combine the three steps above by just running:  
```bash
sh scripts/install.sh
```

Notably, we use the following important libraries:
- datasets 2.17.1
- einops 0.7.0
- huggingface-hub 0.20.3
- openai 0.27.8
- torch 2.1.2
- transformers 4.38.0
- vllm 0.3.3

## Supported Methods
So far we support the following contamination detection methods:

| **Method** | **Use Case** | **Short description** | **White-box access?** | **Reference** |  
|---|---|---|---|-------------------|
| gpt-2 | data contamination | String matching | _ | [paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) Section 4 |
| gpt-3 | data contamination | String matching | _ |
| exact | data contamination | String matching | _ |
| palm | data contamination | String matching | _ |
| gpt-4 | data contamination | String matching | _ |
| platypus | data contamination | Embeddings similarity | _ |
| guided-prompting | model contamination | Likelihood | yes |
| sharded-likelihood | model contamination | Likelihood | yes |
| min-prob | model contamination | LLM-based method | no |
| cdd | model contamination | Likelihood | no |

## vLLM
The following methods require to launch a vLLM instance which will handle model inference:

| **Method** | 
|---|
| guided-prompting |
| min-prob |
| cdd |

To launch the instance, first run the following command in a terminal: 
```bash
sh scripts/vllm_hosting.sh
```
You are required to specify a **port number** and **model name** in this shell script. 

## Run Contamination Detection
To run contamination detection, follow the multiple test scripts in scripts/tests/ folder.  

For instance, to run sharded-likelihood on Hellaswag with Llama-2-7B:
```bash
sh scripts/tests/model/sharded-likelihood/test_hellaswag.sh -m <path_to_your_llama-2-7b_folder> 
```

To run a method using vLLM like guided-prompting for instance, the only difference is to pass the port number as argument:
```bash
sh scripts/tests/model/guided-prompting/test_hellaswag.sh -m <path_to_your_llama-2-7b_folder> -p <port_number_from_your_vllm_instance>
```
