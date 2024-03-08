"""
This file implements the model contamination detection through the min-K-prob approach.
"""

# TODO: Add the inherited copyright here.
# Most codes are copied from https://github.com/swj0419/detect-pretrain-code/blob/main/src/run.py
import copy
import logging
logging.basicConfig(level='ERROR')
import zlib
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from llmsanitize.model_methods.llm import LLM


def load_model(name1, name2):
    if "davinci" in name1:
        model1 = None
        tokenizer1 = None
    else:
        model1 = AutoModelForCausalLM.from_pretrained(name1, return_dict=True, device_map='auto')
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained(name1)

    if "davinci" in name2:
        model2 = None
        tokenizer2 = None
    else:
        model2 = AutoModelForCausalLM.from_pretrained(name2, return_dict=True, device_map='auto')
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained(name2)
    return model1, model2, tokenizer1, tokenizer2


# def calculatePerplexity_gpt3(prompt, modelname):
#     prompt = prompt.replace('\x00', '')
#     responses = None
#     # Put your API key here
#     openai.api_key = "YOUR_API_KEY"  # YOUR_API_KEY
#     while responses is None:
#         try:
#             responses = openai.Completion.create(
#                 engine=modelname,
#                 prompt=prompt,
#                 max_tokens=0,
#                 temperature=1.0,
#                 logprobs=5,
#                 echo=True)
#         except openai.error.InvalidRequestError:
#             print("too long for openai API")
#     data = responses["choices"][0]["logprobs"]
#     all_prob = [d for d in data["token_logprobs"] if d is not None]
#     p1 = np.exp(-np.mean(all_prob))
#     return p1, all_prob, np.mean(all_prob)
#
#
# def calculatePerplexity(sentence, model, tokenizer, gpu):
#     """
#     exp(loss)
#     """
#     input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
#     input_ids = input_ids.to(gpu)
#     with torch.no_grad():
#         outputs = model(input_ids, labels=input_ids)
#     loss, logits = outputs[:2]
#
#     '''
#     extract logits:
#     '''
#     # Apply softmax to the logits to get probabilities
#     probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
#     # probabilities = torch.nn.functional.softmax(logits, dim=-1)
#     all_prob = []
#     input_ids_processed = input_ids[0][1:]
#     for i, token_id in enumerate(input_ids_processed):
#         probability = probabilities[0, i, token_id].item()
#         all_prob.append(probability)
#     return torch.exp(loss).item(), all_prob, loss.item()


def calculate_perplexity(prompt, llm: LLM):
    # TODO: This can be moved to LLM class as a internal function.
    prompt = prompt.replace('\x00', '')
    _, responses, _ = llm.query(prompt, return_full_response=True)
    # print("Response", responses)
    data = responses["choices"][0]["logprobs"]
    all_prob = [d for d in data["token_logprobs"] if d is not None]
    p1 = np.exp(-np.mean(all_prob))
    return p1, all_prob, np.mean(all_prob)


def inference(llm1: LLM, llm2: LLM, text):
    pred = {}

    # if "davinci" in modelname1:
    #     p1, all_prob, p1_likelihood = calculatePerplexity_gpt3(text, modelname1)
    #     p_lower, _, p_lower_likelihood = calculatePerplexity_gpt3(text.lower(), modelname1)
    # else:
    p1, all_prob, p1_likelihood = calculate_perplexity(text, llm1)
    p_lower, _, p_lower_likelihood = calculate_perplexity(text.lower(), llm1)

    # if "davinci" in modelname2:
    #     p_ref, all_prob_ref, p_ref_likelihood = calculatePerplexity_gpt3(text, modelname2)
    # else:
    #     p_ref, all_prob_ref, p_ref_likelihood = calculatePerplexity(text, model2, tokenizer2, gpu=model2.device)
    p_ref, all_prob_ref, p_ref_likelihood = calculate_perplexity(text, llm2)

    # ppl
    pred["ppl"] = p1
    # Ratio of log ppl of large and small models
    pred["ppl/Ref_ppl (calibrate PPL to the reference model)"] = p1_likelihood - p_ref_likelihood

    # Ratio of log ppl of lower-case and normal-case
    pred["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()
    # Ratio of log ppl of large and zlib
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
    pred["ppl/zlib"] = np.log(p1) / zlib_entropy
    # min-k prob
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = int(len(all_prob) * ratio)
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min_{ratio * 100}% Prob"] = -np.mean(topk_prob).item()

    return pred

# Following the logic from this paper: https://arxiv.org/pdf/2310.16789.pdf
def main_min_prob(
    args,
    test_data
):
    llm1 = LLM.from_args(args=args)

    tmp_args = copy.deepcopy(args)
    tmp_args.model_name = args.model_name_2
    tmp_args.openai_creds_key_file = args.openai_creds_key_file_2
    tmp_args.local_port = args.local_port_2
    llm2 = LLM.from_args(args=tmp_args)

    print(f"all data size: {len(test_data)}")
    all_output = []
    test_data = test_data
    debug = 0  # TODO: This is for debug, remove `debug` variable when finished.
    for text in tqdm(test_data):  # TODO: Use multiprocessing to accelerate here.
        print(text)
        new_ex = inference(llm1, llm2, text["text"])  # Here, `test_data` is Dataset, and `text` is a dictionary.
        all_output.append(new_ex)
        debug += 1
        if debug > 5:
            break
    return all_output

#
# if __name__ == '__main__':
#     args = Options()
#     args = args.parser.parse_args()
#     args.output_dir = f"{args.output_dir}/{args.target_model}_{args.ref_model}/{args.key_name}"
#     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
#
#     # load model and data
#     model1, model2, tokenizer1, tokenizer2 = load_model(args.target_model, args.ref_model)
#     if "jsonl" in args.data:
#         data = load_jsonl(f"{args.data}")
#     else:  # load data from huggingface
#         dataset = load_dataset(args.data, split=f"WikiMIA_length{args.length}")
#         data = convert_huggingface_data_to_list_dic(dataset)
#
#     all_output = evaluate_data(data, model1, model2, tokenizer1, tokenizer2, args.key_name, args.target_model, args.ref_model)
#     # TODO: Implement this line:
#     fig_fpr_tpr(all_output, args.output_dir)
