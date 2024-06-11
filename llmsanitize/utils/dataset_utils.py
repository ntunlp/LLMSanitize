

def get_answers_list(data_point, dataset_name):
    choices = []
    if dataset_name == "allenai/ai2_arc":
        choices = data_point["choices"]["text"]
    if dataset_name == "Rowan/hellaswag":
        choices = data_point["endings"]
    if dataset_name == "cais/mmlu":
        choices = data_point["choices"]
    if dataset_name == "truthful_qa":
        choices = data_point["correct_answers"]
    if dataset_name == "winogrande":
        choices = [data_point["option1"], data_point["option2"]]

    return choices

def get_answer_index(data_point, dataset_name):
    answer_index = -1
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if dataset_name == "allenai/ai2_arc":
        key = data_point["answerKey"].lower()
        answer_index = alphabet.index(key)
    if dataset_name == "Rowan/hellaswag":
        answer_index = int(data_point["label"])
    if dataset_name == "cais/mmlu":
        answer_index = data_point["answer"]
    if dataset_name == "truthful_qa":
        best_answer = data_point["best_answer"]
        correct_answers = data_point["correct_answers"]
        answer_index = correct_answers.index(best_answer)
    if dataset_name == "winogrande":
        answer_index = int(data_point["answer"]) - 1

    return answer_index
