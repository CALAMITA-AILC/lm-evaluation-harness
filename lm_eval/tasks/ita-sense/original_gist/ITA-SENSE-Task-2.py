# Huggingface Hub ID of the dataset. If data is not on HF, set the variable to 'other'
DATA_HUB_ID = "other"
######## N.B. We perfomed a test using the 'lm-eval-harness' library, we provide the fields used in the .yml file to load the dataset ########
######## dataset_path: json
######## dataset_name: null
######## dataset_kwargs:
########   data_files: 'path-to-the-data'

# Task type. Values accepted: multiple_choice, open-ended 
OUTPUT_TYPE = "open-ended"
######## "generate_until" strategy in lm-eval-harness ########

### Prompting details

# Template string used to compile the prompt. Use {{}} variables to fill using the dataset columns.
PROMPT_TEMPLATE = "{{instruction}} Input: \"{{input}}\""

# Column in the dataset that contains the gold reference 
TARGET_COLUMN = "output"

# Which split (also known as "configuration" in HF datasets) to sample the few-shot examples from.
FEWSHOT_SPLIT = "train"

# Number of few-shot examples.
N_SHOTS_TO_SAMPLE = 0

# Sample strategy. Accepted values: random, sequential_from-top, sequential_from-bottom
SHOTS_SAMPLE_STRATEGY = "random"

### Evaluation details

######## N.B. this could also be implemented as a regex 'filter' in lm-eval-harness ########
######## N.B. if 're' is already imported just use: extract_answer = lambda x: re.findall('[0-9]+', x)[0] ########
def extract_answer(x):
    import re
    return re.findall('[0-9]+', x)[0]

# (optional). The post-processing function to be applied to every model output.
POSTPROCESSING_FUNC = extract_answer

METRIC_LIST = ["exact_match"]