# LLMSanitize
An open-source library for contamination detection in NLP datasets and Large Language Models (LLMs).  

## Installation
- To query OpenAI models, put OpenAI key to `openai_creds/openai_api_key.txt`
- Install python environment (using conda)
    1. install cuda 12.1
    2. run `sh scripts/install.sh`

## Supported Methods
So far we support the following contamination detection methods:

| **Method** | **Use Case** | **Short description** |  
|---|---|---|
| gpt-2 | data contamination | 8-gram matching | 
| gpt-3 | data contamination | 13-gram matching |
| exact | data contamination | exact substring matching |
| palm | data contamination | 70% overlap in 8-gram matching | 
| gpt-4 | data contamination | 50-chars substring matching |
| platypus | data contamination | SentenceTransformers cosine similarity |
| guided-prompting | model contamination | https://arxiv.org/abs/2308.08493 | 
| sharded-likelihood | model contamination | https://arxiv.org/abs/2310.17623 |
| min-prob | model contamination | Min-K% Prob: https://arxiv.org/abs/2310.16789 | 

## vLLM
The following methods require to launch a vLLM instance which will handle model inference:

| **Method** | 
|---|
| guided-prompting |
| min-prob |

To launch the instance, first run the following command in a terminal: 
```bash
sh scripts/vllm_hosting.sh
```
You are required to specify a **port number** and **model name** in this shell script. 

## Run Contamination Check

1. `sh scripts/run_contamination.sh`
