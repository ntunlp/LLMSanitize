# LLMSanitize
An open-source library for contamination detection in NLP datasets and Large Language Models (LLMs).  
Relies on Python 3.9.

## Supported Methods
So far we support the following contamination detection methods:

| **Method** | **Use Case** |  
|:-:|:-:|
| gpt-2 | data contamination |
| gpt-3 | data contamination |
| exact | data contamination |
| palmn | data contamination |
| gpt-4 | data contamination |
| platypus | data contamination |
| guided-prompting | model contamination |
| sharded-lieklihood | model contamination |
| min-prob | model contamination |


## Installation
- To query OpenAI models, put OpenAI key to `openai_creds/openai_api_key.txt`
- Install python environment (using conda)
    1. install cuda 12.1
    2. run `sh scripts/install.sh`

## Run Contamination Check
(pre-requisite) If the method requires local model inference, run vllm serving in a separate bash/zsh window:
```bash
sh scripts/vllm_hosting.sh
```
Then: 
1. `sh scripts/run_contamination.sh`


### to-do list
- Standardize the outputs from each method
- Make data contamination support large pre-training files not fitting in memory 
- Implement According To prompting model contamination, from there: https://arxiv.org/abs/2305.13252
- Implement LLMDecontaminator model contamination, from there: https://arxiv.org/abs/2311.04850
