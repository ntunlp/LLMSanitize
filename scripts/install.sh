# install python-3.9 + cuda 12.1
conda create -n lmsanitize python=3.9 -y
source activate lmsanitize

# # install vllm package
# pip install vllm

# install required packages
pip install -r requirements.txt

# required post-processing for some packages
python -c "import nltk; nltk.download('punkt')"