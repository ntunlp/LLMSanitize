import re
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from lmsanitize.configs.config import supported_methods, config
from lmsanitize.utils.method_utils import guided_prompt_process_fn
from lmsanitize.base_contamination_checker import BaseContaminationChecker

class ModelContaminationChecker(BaseContaminationChecker):
    def __init__(self, args):
        super(ModelContaminationChecker, self).__init__(args)

    def run_contamination(self, method):
        if not(method in self.supported_methods.keys()):
            methods = list(self.supported_methods.keys())
            raise KeyError(f'Please pass in a data contamination method which is supported, among: {methods}')

        if method == "guided-prompting":
            self.contamination_guided_prompting()
            
        if method == "sharded-likelihood":
            self.sharded_likelihood_comparison_test()

    def contamination_guided_prompting(self):
        import lmsanitize.prompts.guided_prompting.general_instructions as gi_prompts
        import lmsanitize.prompts.guided_prompting.guided_instructions as gui_prompts
        # method-specific dataset processing:
        ## only examine eval data here
        process_fn = guided_prompt_process_fn
        
        # based on task type, choose prompt template
        type_str = self.guided_prompting_task_type
        guided_template = getattr(gui_prompts, f"GUI_{type_str}")
        general_template = getattr(gi_prompts, f"GI_{type_str}")

        # TODO: randomly sample a subset of data instances to check (with seed)
        # process each example
        for idx, example in enumerate(self.eval_data):
            process_fn(example, idx, config=config, use_local_model=self.use_local_model, 
                        split_name=self.eval_set_key, dataset_name=self.eval_data_name, label_key=self.label_key, 
                        text_key=self.text_key, general_template=general_template, guided_template=guided_template)

        print("Early Stopping for debugging")
        
    def sharded_likelihood_comparison_test(self):
        None
