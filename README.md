# LLMSanitize
An open-source library for contamination detection in NLP datasets and Large Language Models (LLMs).  
Relies on Python 3.9.

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
| sharded-lieklihood | model contamination | https://arxiv.org/abs/2310.17623 |
| min-prob | model contamination | Min-K% Prob: https://arxiv.org/abs/2310.16789 | 


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
