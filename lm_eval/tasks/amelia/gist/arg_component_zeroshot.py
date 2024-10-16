DATA_HUB_ID = "nlp-unibo/AMELIA"

OUTPUT_TYPE = "multiple_choice"

PROMPT_TEMPLATE = "Classifica il seguente testo argomentativo come premessa 'prem' o conclusione 'conc'. Per premessa (prem) si intende una proposizione che fornisce una ragione o un supporto per l'argomentazione. Per conclusione (conc) si intende l'affermazione che segue logicamente dalle premesse e rappresenta il punto finale che viene argomentato. Testo: {{Text}}"

TARGET_COLUMN = "Component"

MC_OPTIONS = ["conc", "prem"]

N_SHOTS_TO_SAMPLE = 0


from sklearn.metrics import f1_score

def macro_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def f1_classes(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None) # returns array with the scores for each class

METRIC_LIST = [macro_f1_score, f1_classes] # macro f1 is the official score. Additionally, we would like to evaluate the f1 score of each class to provide further insights.