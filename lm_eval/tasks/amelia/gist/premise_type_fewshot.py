DATA_HUB_ID = "nlp-unibo/AMELIA"

OUTPUT_TYPE = "multiple_choice" # this is a multi-label task

PROMPT_TEMPLATE = '''Classifica la seguente premessa come di fatto 'F', legale 'L' o entrambe. Le premesse di fatto (F) descrivono situazioni ed eventi fattuali relativi al caso di specie. Le premesse legali (L) specificano il contenuto giuridico (norme giuridiche, precedenti, interpretazione delle leggi e dei principi applicabili). L'output atteso è una lista con tutte le label applicabili. Ad esempio: ['F', 'L']. 

Esempi:

Testo: Per i primi giudici nel caso di specie questa esenzione non poteva essere applicata perché la complessiva attività di 'A' srl era un'attività commerciale svolta in concorrenza con altre imprese operanti nel settore
Risposta: ['F']

Testo: In assenza di siffatti elementi, che in via presuntiva avrebbero potuto fare giungere questo giudice a conclusioni diverse in via logica, si deve confermare l’esito cui è giunta la commissione provinciale
Risposta: ['F']

Testo: Su questo si osserva che si deve condividere la circostanza dedotta dal giudice di prime cure per cui deve essere il contribuente, ove sia contestata la inerenza e verità della rappresentazione ricavabile dal documento contabile, a dare la dimostrazione della fondatezza e della correttezza del comportamento tenuto
Risposta: ['L']

Testo: L'Ufficio non potrà impedire ad un imprenditore, per esempio, di cedere immobili con prezzi bassi onulli per ricavare liquidità a fronte di nuovi impegni, ma dovrà rilevare la condotta antieconomica dello stesso sulla base dell’utile di esercizio
Risposta: ['L']

Testo: Invero l'avviso di accertamento è fondato sul mancato rispetto, da parte del contribuente, nel calcolo del ROL, delle disposizioni dell'articolo 96, secondo comma, del TUIR, che ne definisce le modalità
Risposta: ['F', 'L']

Testo: La società 'A', per quanto previsto dall'art. 4, comma 18 del Regolamento CEE n. 2913/1992, riveste il ruolo di 'dichiarante in Dogana', soggetto passivo della obbligazione
Risposta: ['F', 'L']

Testo: {{Text}}'''

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