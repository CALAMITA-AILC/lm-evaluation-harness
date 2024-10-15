CREATE DATABASE IF NOT EXISTS bowling;

USE bowling;

CREATE TABLE pista (
id_pista INT PRIMARY KEY not null auto_increment,
is_disponibile BOOLEAN,
id_impianto INT
);

CREATE TABLE giocatore (
id_giocatore INT PRIMARY KEY not null auto_increment,
nome_giocatore VARCHAR(255) not null,
cognome_giocatore VARCHAR(255) not null,
data_di_nascita DATE not null,
email VARCHAR(255) not null,
telefono VARCHAR(255) not null
);

CREATE TABLE partita (
ora_inizio TIME not null,
tenuta_il DATE not null,
id_giocatore INT,
tot_score INT not null,
ora_fine TIME not null,
id_pista INT,
primary key (id_pista, ora_inizio, tenuta_il, id_giocatore),
FOREIGN KEY (id_pista) REFERENCES pista(id_pista)
on delete cascade on update cascade,
FOREIGN KEY (id_giocatore) REFERENCES giocatore(id_giocatore)
on delete cascade on update cascade
);

CREATE TABLE scarpe (
id_scarpe INT PRIMARY KEY not null auto_increment,
prezzo FLOAT not null,
taglia INT not null,
marca VARCHAR(255) not null,
id_giocatore INT,
FOREIGN KEY (id_giocatore) REFERENCES giocatore(id_giocatore)
on delete cascade on update cascade
);

CREATE TABLE abbonamento (
id_abbonamento INT PRIMARY KEY not null auto_increment,
valido_dal DATE not null,
scade_il DATE not null,
tipologia_abbonamento VARCHAR(255) not null,
id_giocatore INT,
FOREIGN KEY (id_giocatore) REFERENCES giocatore(id_giocatore)
on delete cascade on update cascade
);

CREATE TABLE noleggio (
id_giocatore INT,
id_scarpe INT,
ora_inizio TIME,
data DATE,
is_consegnato BOOLEAN,

PRIMARY KEY (id_giocatore, id_scarpe, ora_inizio, data),
FOREIGN KEY (id_giocatore) REFERENCES giocatore(id_giocatore)
ON DELETE CASCADE,
FOREIGN KEY (id_scarpe) REFERENCES scarpe(id_scarpe)
ON DELETE CASCADE
);

ALTER TABLE giocatore AUTO_INCREMENT=810;

ALTER TABLE scarpe AUTO_INCREMENT=8000;