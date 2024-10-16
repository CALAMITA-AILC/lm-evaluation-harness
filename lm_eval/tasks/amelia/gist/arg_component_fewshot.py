DATA_HUB_ID = "nlp-unibo/AMELIA"

OUTPUT_TYPE = "multiple_choice"

PROMPT_TEMPLATE = '''Classifica il seguente testo argomentativo come premessa 'prem' o conclusione 'conc'. Per premessa (prem) si intende una proposizione che fornisce una ragione o un supporto per l'argomentazione. Per conclusione (conc) si intende l'affermazione che segue logicamente dalle premesse e rappresenta il punto finale che viene argomentato. 

Esempi:

Testo: Si osserva poi che ritenere che la mancata possibilità di detrazione a favore di soggetti come il ricorrente comporti un aiuto di Stato in favore degli ospedali pubblici, in quanto le perdite degli stessi vengono ripianate dalle USL e dalla Regioni trascura di considerare l'accessibilità, indiscriminata, ai servizi dei nosocomi pubblici da parte dei soggetti iscritti al SSN, rispetto a quella ad un libero professionista sanitario che, in quanto tale, ben potrebbe rifiutarsi di prestare i propri servigi al pare di un normale contraente
Risposta: prem

Testo: L'appello è infondato e va respinto
Risposta: conc

Testo: Va osservato che la motivazione dell'atto di accertamento non può esaurirsi nel rilievo dello scostamento, ma deve essere integrata con la dimostrazione dell'applicabilità in concreto dello 'standard' prescelto e con le ragioni per le quali sono state disattese le contestazioni sollevate dal contribuente. (cfr. Cass. S.U. 26635/2009, Cass. 12558/2010, Cass. 12428/2012, Cass. 23070/2012)
Risposta: prem

Testo: Dunque, l'ufficio ha riconosciuto la non imponibilità IVA delle cessioni all'esportazione, così cessando sul punto la materia del contendere
Risposta: conc

Testo: Risulta d'altronde dalle osservazioni scritte del governo spagnolo che quest'ultimo non riesce a discernere tale differenza ad un esame delle pertinenti norme dell'ordinamento spagnolo.
Risposta: prem

Testo: Il Collegio, esaminata l'eccezione preliminare svolta nel suo appello dall'Ufficio e relativa alla richiesta nullità della sentenza per mancata instaurazione del contraddittorio, la respinge
Risposta: conc

Testo: {{Text}}'''

TARGET_COLUMN = "Component"

MC_OPTIONS = ["conc", "prem"]

N_SHOTS_TO_SAMPLE = 0


from sklearn.metrics import f1_score

def macro_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def f1_classes(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None) # returns array with the scores for each class

METRIC_LIST = [macro_f1_score, f1_classes] # macro f1 is the official score. Additionally, we would like to evaluate the f1 score of each class to provide further insights.