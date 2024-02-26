from lmsanitize.utils.openai_api_utils import *
from lmsanitize.configs.config import *

class LLM:
    def __init__(self, use_local_model=False):
        self.config = config
        if not use_local_model:
            initialize_openai(config)
            self.query_fn = query_llm_api
        else:
            initialize_openai_local(config)
            self.query_fn = query_llm_api
            self.config.openai.model_name = self.config.local.model_path
    
    def query(self, prompt):
        outputs, cost = self.query_fn(self.config, prompt)
        assert len(outputs) == 1
        return outputs[0], cost