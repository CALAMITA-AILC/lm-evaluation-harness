#it's essential to refer to the full repository for a more comprehensive understanding of the dataset and evaluation methods.
#in particular take a look at readme.md and \evaluation\instructions.txt

#dataset is available at this repository. Extract it from Termite.zip file using password "youshallnotpass"

OUTPUT_TYPE = "open-ended"

#sql database dumps are available in the repository under Termite/databases/ folder and are named [no_insert_*db.sql] where *db is the name of a database
PROMPT_1_TEMPLATE = """Considera il seguente database : {DUMP SQL}"""

#nl query are retrievable in repository in Termite/queries.json where all natural language queries are stored
PROMPT_2_TEMPLATE = """Traduci in SQL la seguente query in linguaggio naturale : {NL QUERY}"""

#output of PROMPT_2_TEMPLATE must be processed in order to be feeded to the evaluation script \evaluation\evaluation.py

#evaluation can be executed by running evaluation.py under \evaluation
EVALUATION_METRIC = ["execution accurracy"]
