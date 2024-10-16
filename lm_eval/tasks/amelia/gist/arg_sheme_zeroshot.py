DATA_HUB_ID = "nlp-unibo/AMELIA"

OUTPUT_TYPE = "multiple_choice" # this is a multi-label task

PROMPT_TEMPLATE = '''Classifica la seguente premessa legale in uno o più dei seguenti schemi argomentativi: Rule, Prec, Class, Itpr, Princ. 
Rule: se esiste un riferimento esplicito o implicito a un articolo di legge o la citazione del testo di una norma.
Prec: se esiste un riferimento ad una precedente pronuncia della Corte di Cassazione o della Corte di Giustizia dell'Unione Europea.
Class: se c'è la definizone di un concetto giuridico o degli elementi costitutivi dello stesso.
Itpr: se c'è il riferimento a uno dei criteri interpretativi contenuti all'art. 12 delle preleggi (letterale, teleologica, psicologica, sistematica) al codice civile.
Princ: se c'è un riferimento espresso a un prinicpio generale del diritto (es. principio di proporzionalità).
L'output atteso è una lista con tutte le label applicabili. Ad esempio: ['Prec', 'Princ', 'Rule']. 
Testo: {{Text}} Lista:'''

TARGET_COLUMN = "Scheme" # apply dropna, Scheme is a valid attribute only for legal premises

MC_OPTIONS = ['Class', 'Itpr', 'Prec', 'Princ', 'Rule']

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
