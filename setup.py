from setuptools import find_packages, setup

setup(
    name='lmsanitize',
    packages=find_packages(include=['lmsanitize']),
    version='0.0.1',
    description='An open-source library for contamination detection in NLP',
    author='NTU-NLP',
    install_requires=[
        'transformers==4.38.1',
        'datasets==2.17.1'
    ],
)