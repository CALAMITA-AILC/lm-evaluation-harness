import string
import unicodedata
import re
import random

def peggiorativa(doc):
    if int(doc['pejorative'])==0:
      return 'no'  
    else:
      return 'si'
    
def misogena(doc):
    if int(doc['misogyny'])==0:
      return 'no'  
    else:
      return 'si' 
  
def genera_misoginia(doc):
    if int(doc['pejorative'])==1:
        tipo='peggiorativa'
    elif int(doc['pejorative'])==0:
        tipo='non peggiorativa'
    else:
        tipo=''
        with open('error_log.log','a') as er:
           er.write(doc)
    return f"Di seguito è riportata un'istruzione che descrive un task. Scrivete una risposta che completi adeguatamente la richiesta. Istruzione Considerando che la parola '{doc['word']}' è {tipo} in questa frase: '{doc['text']}', la frase è misogina?  Rispondi solo SI o NO. Risposta: "

def unisci_e_normalizza(stringa):
    #Unisco i caratteri e rimuovo gli spazi
    stringa=(''.join(stringa).strip(''))
    #Tolgo caratteri non ascii e accenti
    stringa=unicodedata.normalize('NFKD',stringa)
    stringa=''.join(filter(lambda x: x in string.ascii_letters,stringa))
    stringa.lower()
    return stringa
def exact_match_custom(dati):
     obiettivo=dati[0]
     risultato=dati[1]
     obiettivo=unisci_e_normalizza(obiettivo)
     risultato=unisci_e_normalizza(risultato)
     if obiettivo==risultato:
          return 1
     else:
          return 0

