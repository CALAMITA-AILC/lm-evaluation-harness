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

METRIC_LIST = ["harmonicRougeBertScore", "rougeL", "bertScore"]

######## The following code was tested in a 'utils.py' file (following common practises done in lm-eval-harness) ########
######## If the global variables are a problem, remove them and do everything inside the function ########

from evaluate import load
from rouge_score import rouge_scorer

ROUGE_SCORER = None
BERT_SCORER = None

def process_results_gen(doc, results):

    expected_output = doc["output"].strip()
    result = results[0].strip()

    global ROUGE_SCORER
    if ROUGE_SCORER is None:
        ROUGE_SCORER = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    
    global BERT_SCORER
    if BERT_SCORER is None:
        BERT_SCORER = load("bertscore", keep_in_memory=True)

    rouge_result = ROUGE_SCORER.score(expected_output, result)['rougeL'].fmeasure
    bert_result = BERT_SCORER.compute(predictions=[result], references=[expected_output], lang="it")["f1"][0]

    rouge_bert_result = (5 * rouge_result * bert_result) / (4 * rouge_result + bert_result)

    return {"rougeBertScore": rouge_bert_result, "rougeL": rouge_result, "bertScore": bert_result}