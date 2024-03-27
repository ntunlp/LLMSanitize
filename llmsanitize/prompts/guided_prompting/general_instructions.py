"""
    @Author: Shahriar Golchin
    @Links: https://github.com/shahriargolchin/time-travel-in-llms/blob/main/src/prompts/general_instructions.py
"""
# general instruction for fiil-in-the-blank task
GI_FIM = """INSTRUCTION:
Finish the SECOND PIECE based on the FIRST PIECE, such that these two pieces become a single instance with the LABEL token to connect them.

FIRST PIECE:
{first_piece}

LABEL: {label}

SECOND PIECE:"""

# general instruction for Open QA task
GI_QA = """INSTRUCTION:
Finish the SECOND PIECE based on the FIRST PIECE, such that these two pieces become a single question with the following ANSWER.

ANSWER: {label}

FIRST PIECE:
{first_piece}

SECOND PIECE:"""

# general instruction for classification task
GI_CLS = """INSTRUCTION:
Finish the SECOND PIECE based on the FIRST PIECE, such that these two pieces become a single instance with the following LABEL.

LABEL: {label}

FIRST PIECE:
{first_piece}

SECOND PIECE:"""

# general instruction for natural language inference task
GI_NLI = """INSTRUCTION:
Finish SENTENCE 2 based on SENTENCE 1, such that the following LABEL shows the logical relationship between SENTENCE 1 and SENTENCE 2.

SENTENCE 1:
{first_piece}

LABEL: {label}

SENTENCE 2:"""

# general instruction for summarization task
GI_SUM = """INSTRUCTION:
Finish the SECOND PIECE based on the FIRST PIECE, such that these two pieces become a single summary.

FIRST PIECE:
{first_piece}

SECOND PIECE:"""

# general instruction for extreme summarization task (one-sentence summary)
GI_XSUM = """INSTRUCTION:
Finish the SECOND PIECE based on the FIRST PIECE, such that these two pieces become a single one-sentence summary.

FIRST PIECE:
{first_piece}

SECOND PIECE:"""