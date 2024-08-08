"""
Contamination detection class for open_data contamination use cases: func(data1, data2)
"""

from llmsanitize.base_contamination_checker import BaseContaminationChecker
from llmsanitize.open_data_methods.gpt2 import main_gpt2
from llmsanitize.open_data_methods.gpt3 import main_gpt3
from llmsanitize.open_data_methods.exact import main_exact
from llmsanitize.open_data_methods.palm import main_palm
from llmsanitize.open_data_methods.gpt4 import main_gpt4
from llmsanitize.open_data_methods.platypus import main_platypus


class OpenDataContaminationChecker(BaseContaminationChecker):
    def __init__(self, args):
        super(OpenDataContaminationChecker, self).__init__(args)

    def run_contamination(self, method):
        if not(method in self.supported_methods.keys()):
            methods = list(self.supported_methods.keys())
            raise KeyError(f'Please pass in a open_data contamination method which is supported, among: {methods}')

        if method == "gpt-2":
            self.contamination_gpt2()
        elif method == "gpt-3":
            self.contamination_gpt3()
        elif method == "exact":
            self.contamination_exact()
        elif method == "palm":
            self.contamination_palm()
        elif method == "gpt-4":
            self.contamination_gpt4()
        elif method == "platypus":
            self.contamination_platypus()

    def contamination_gpt2(self):
        main_gpt2(
            train_data=self.train_data,
            eval_data=self.eval_data,
            train_data_name=self.train_data_name,
            eval_data_name=self.eval_data_name,
            eval_set_key=self.eval_set_key,
            stream_train_data=self.stream_train_data,
            text_key=self.text_key,
            text_keys=self.text_keys
        )

    def contamination_gpt3(self):
        main_gpt3(
            train_data=self.train_data,
            eval_data=self.eval_data,
            train_data_name=self.train_data_name,
            eval_data_name=self.eval_data_name,
            eval_set_key=self.eval_set_key,
            stream_train_data=self.stream_train_data,
            text_key=self.text_key,
            text_keys=self.text_keys
        )

    def contamination_exact(self):
        main_exact(
            train_data=self.train_data,
            eval_data=self.eval_data,
            train_data_name=self.train_data_name,
            eval_data_name=self.eval_data_name,
            eval_set_key=self.eval_set_key,
            stream_train_data=self.stream_train_data,
            text_key=self.text_key,
            text_keys=self.text_keys
        )

    def contamination_palm(self):
        main_palm(
            train_data=self.train_data,
            eval_data=self.eval_data,
            train_data_name=self.train_data_name,
            eval_data_name=self.eval_data_name,
            eval_set_key=self.eval_set_key,
            stream_train_data=self.stream_train_data,
            text_key=self.text_key,
            text_keys=self.text_keys
        )

    def contamination_gpt4(self):
        main_gpt4(
            train_data=self.train_data,
            eval_data=self.eval_data,
            train_data_name=self.train_data_name,
            eval_data_name=self.eval_data_name,
            eval_set_key=self.eval_set_key,
            stream_train_data=self.stream_train_data,
            text_key=self.text_key,
            text_keys=self.text_keys
        )

    def contamination_platypus(self):
        main_platypus(
            train_data=self.train_data,
            eval_data=self.eval_data,
            train_data_name=self.train_data_name,
            eval_data_name=self.eval_data_name,
            eval_set_key=self.eval_set_key,
            stream_train_data=self.stream_train_data,
            text_key=self.text_key,
            text_keys=self.text_keys
        )
