DATA_HUB_ID = "nlp-unibo/AMELIA"

OUTPUT_TYPE = "multiple_choice" # this is a multi-label task

PROMPT_TEMPLATE = "Classifica la seguente premessa come di fatto 'F', legale 'L' o entrambe. Le premesse di fatto (F) descrivono situazioni ed eventi fattuali relativi al caso di specie. Le premesse legali (L) specificano il contenuto giuridico (norme giuridiche, precedenti, interpretazione delle leggi e dei principi applicabili). L'output atteso è una lista con tutte le label applicabili. Ad esempio: ['F', 'L']. Testo: {{Text}} Lista:"

TARGET_COLUMN = "Type" # apply dropna, Type is a valid attribute only for premises

MC_OPTIONS = ["F", "L"]

N_SHOTS_TO_SAMPLE = 0


from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
mlb.fit([MC_OPTIONS])
y_true = mlb.transform(df_test[TARGET_COLUMN].dropna())

def binarize(output):
    return mlb.transform(output)
    
POSTPROCESSING_FUNC = binarize


from sklearn.metrics import f1_score

def macro_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def f1_classes(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None) # returns array with the scores for each class

METRIC_LIST = [macro_f1_score, f1_classes] # macro f1 is the official score. Additionally, we would like to evaluate the f1 score of each class to provide further insights.
