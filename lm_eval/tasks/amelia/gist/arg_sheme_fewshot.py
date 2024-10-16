DATA_HUB_ID = "nlp-unibo/AMELIA"

OUTPUT_TYPE = "multiple_choice" # this is a multi-label task

PROMPT_TEMPLATE = '''Classifica la seguente premessa legale in uno o più dei seguenti schemi argomentativi: Rule, Prec, Class, Itpr, Princ. 
Rule: se esiste un riferimento esplicito o implicito a un articolo di legge o la citazione del testo di una norma.
Prec: se esiste un riferimento ad una precedente pronuncia della Corte di Cassazione o della Corte di Giustizia dell'Unione Europea.
Class: se c'è la definizone di un concetto giuridico o degli elementi costitutivi dello stesso.
Itpr: se c'è il riferimento a uno dei criteri interpretativi contenuti all'art. 12 delle preleggi (letterale, teleologica, psicologica, sistematica) al codice civile.
Princ: se c'è un riferimento espresso a un prinicpio generale del diritto (es. principio di proporzionalità).
L'output atteso è una lista con tutte le label applicabili. Ad esempio: ['Prec', 'Princ', 'Rule'].

Esempi:

Testo: Infatti, è ben vero che, ai sensi del combinato disposto dagli articoli 54 e 23 D.Lgs. n. 546/1992, il convenuto in appello deve costituirsi entro 60 giorni dal giorno in cui ricorso è stato notificato.
Risposta: ['Rule']

Testo: L’Amministrazione “ha l'onere di provare ed allegare gli elementi probatori su cui si fondi la contestazione, tra i quali possono rilevare, in via indiziaria, quali elementi sintomatici della mancata esecuzione della prestazione dal fatturante, l’assenza della minima dotazione personale e strumentale, l’immediatezza dei rapporti (cedente/prestatore fatturante interposto e cessionario/committente), una conclamata inidoneità allo svolgimento dell'attività economica e la non corrispondenza tra i cedenti e la società coinvolta nell’operazione”
Risposta: ['Prec']

Testo: In conclusione, per quanto fin qui esposto, i “compro oro” possono essere definiti come “esercizi commerciali che acquistano, commerciano o rivendono oggetti d’oro, di metalli preziosi o recanti pietre preziose usati e li cedono nella forma di materiale, di rottami d’oro o di metalli preziosi alle fonderie o ad altre aziende specializzate nel recupero di materiali preziosi”. Trattano esclusivamente prodotti finiti e non possono, congiuntamente, acquistare oro da gioielleria usato, fonderlo (per proprio conto o con incarico a terzi) e cedere il prodotto finito ottenuto
Risposta: ['Class']

Testo: Si vuole dire, in sostanza, che la finalità del contraddittorio anticipato è quella di mettere il contribuente nella condizione di potere fare valere le proprie osservazioni prima che la decisione sia adottata e, quindi, di far sì che l’Amministrazione possa tener conto di tutti gli elementi del caso nell’adottare (0 non adottare) il provvedimento ovvero nel dare a questo un contenuto piuttosto che un altro.
Risposta: ['Itpr']

Testo: Nell'ordinamento unionale, pertanto, il principio del contraddittorio in ambito tributario prescinde dalla natura del tributo e deve trovare applicazione ogni qualvolta l'amministrazione sulla base della documentazione esibita ritenga dovere dare alla stessa documentazione interpretazione diversa da quella data dal contribuente invitandolo, come detto fornire nel corso del contraddittorio le ragioni della propria scelta
Risposta: ['Princ']

Testo: In sintesi per esterovestizione si intende la fittizia localizzazione della residenza fiscale di un soggetto all'estero, in particolare in un Paese con un trattamento fiscale più vantaggioso di quello nazionale,che la giurisprudenza configura in termini di abuso del diritto riconosciuto, in via tendenziale, come principio generale anche nel diritto dei singoli Stati membri (v.Cass., Sez. Un., n. 30055 del 2008, secondo la quale il divieto di abuso del diritto si traduce in un principio generale antielusivo che trova fondamento, in tema di tributi non armonizzati, nei principi costituzionali di capacità contributiva e di progressività dell'imposizione).
Risposta: ['Prec', 'Class', 'Princ']

Testo: La denuncia, infatti, non codificata nel codice di procedura penale (a differenza della notizia di reato di cui all’articolo 347 c.p.p.), può definirsi come qualunque atto con il quale chiunque abbia notizia di un reato perseguibile d'ufficio ne informa il pubblico ministero o un ufficiale di polizia giudiziaria.
Risposta: ['Rule', 'Itpr', 'Class']
 
Testo: {{Text}}'''

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