from setuptools import setup, find_packages

install_requires = [
    "datasets==2.17.1",
    "evaluate==0.4.1",
    "huggingface-hub==0.20.3",
    "matplotlib==3.8.3",
    "nltk==3.8.1",
    "numpy==1.26.4",
    "openai==1.35.7",
    "pandas==2.2.1",
    "rouge-score==0.1.2",
    "scipy==1.12.0",
    "sentence-transformers==3.0.1",
    "sentencepiece==0.2.0",
    "torch==2.1.2",
    "transformers==4.38.0",
    "triton==2.1.0",
    "vllm==0.3.3"
]

setup(
    name='llmsanitize',
    version='0.0.1',
    author='Mathieu Ravaut (NTU-NLP lab)',
    description='LLMSanitize: a package to detect contamination in LLMs',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license="Apache 2.0",
    url='https://github.com/ntunlp/LLMSanitize',
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.9',
)
