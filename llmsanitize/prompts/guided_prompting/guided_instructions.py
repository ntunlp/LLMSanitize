"""
    @Author: Shahriar Golchin
    @Links: https://github.com/shahriargolchin/time-travel-in-llms/blob/main/src/prompts/guided_instructions.py
"""
# guided instruction for fiil-in-the-blank task
GUI_FIM = """INSTRUCTION:
You are provided with the FIRST PIECE of an instance from the {split_name} split of the {dataset_name} dataset.
Finish the SECOND PIECE of the instance as EXACTLY appeared in the dataset.
ONLY rely on the original form of the instance in the dataset to finish the SECOND PIECE.

FIRST PIECE:
{first_piece}

LABEL: {label}

SECOND PIECE:"""

# guided instruction for Open QA task
GUI_QA = """INSTRUCTION:
You are provided with the FIRST PIECE of an instance from the {split_name} split of the {dataset_name} dataset.
Finish the SECOND PIECE of the instance as EXACTLY appeared in the dataset.
ONLY rely on the original form of the instance in the dataset to finish the SECOND PIECE.

ANSWER: {label}

FIRST PIECE:
{first_piece}

SECOND PIECE:"""

# guided instruction for classification task
GUI_CLS = """INSTRUCTION:
You are provided with the FIRST PIECE of an instance from the {split_name} split of the {dataset_name} dataset.
Finish the SECOND PIECE of the instance as EXACTLY appeared in the dataset.
ONLY rely on the original form of the instance in the dataset to finish the SECOND PIECE.

LABEL: {label}

FIRST PIECE:
{first_piece}

SECOND PIECE:"""

# guided instruction for natural language inference task
GUI_NLI = """INSTRUCTION:
You are provided with SENTENCE 1 from the {split_name} split of the {dataset_name} dataset.
Finish SENTENCE 2 as appeared in the dataset.
SENTENCE 2 MUST EXACTLY match the instance in the dataset.

SENTENCE 1:
{first_piece}

LABEL: {label}

SENTENCE 2:"""

# guided instruction for summarization task
GUI_SUM = """INSTRUCTION:
You are provided with the FIRST PIECE of a summary from the {split_name} split of the {dataset_name} dataset.
Finish the SECOND PIECE of the summary as EXACTLY appeared in the dataset.
ONLY rely on the original form of the summary in the dataset to finish the SECOND PIECE.

FIRST PIECE:
{first_piece}

SECOND PIECE:"""

# guided instruction for extreme summarization task (one-sentence summary)
GUI_XSUM = """INSTRUCTION:
You are provided with the FIRST PIECE of a one-sentence summary from the {split_name} split of the {dataset_name} dataset.
Finish the SECOND PIECE of the summary as EXACTLY appeared in the dataset.
ONLY rely on the original form of the summary in the dataset to finish the SECOND PIECE.

FIRST PIECE:
{first_piece}

SECOND PIECE:"""