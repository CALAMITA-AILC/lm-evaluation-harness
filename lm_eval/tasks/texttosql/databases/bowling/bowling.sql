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
giocatore INT,
tot_score INT not null,
ora_fine TIME not null,
id_pista INT,
primary key (id_pista, ora_inizio, tenuta_il, giocatore),
FOREIGN KEY (id_pista) REFERENCES pista(id_pista)
on delete cascade on update cascade,
FOREIGN KEY (giocatore) REFERENCES giocatore(id_giocatore)
on delete cascade on update cascade
);

CREATE TABLE scarpe (
id_scarpe INT PRIMARY KEY not null auto_increment,
prezzo FLOAT not null,
taglia INT not null,
marca VARCHAR(255) not null,
giocatore INT,
FOREIGN KEY (giocatore) REFERENCES giocatore(id_giocatore)
on delete cascade on update cascade
);

CREATE TABLE abbonamento (
id_abbonamento INT PRIMARY KEY not null auto_increment,
valido_dal DATE not null,
scade_il DATE not null,
tipologia_abbonamento VARCHAR(255) not null,
giocatore INT,
FOREIGN KEY (giocatore) REFERENCES giocatore(id_giocatore)
on delete cascade on update cascade
);

CREATE TABLE noleggio (
giocatore INT,
id_scarpe INT,
ora_inizio TIME,
data DATE,
is_consegnato BOOLEAN,

PRIMARY KEY (giocatore, id_scarpe, ora_inizio, data),
FOREIGN KEY (giocatore) REFERENCES giocatore(id_giocatore)
ON DELETE CASCADE,
FOREIGN KEY (id_scarpe) REFERENCES scarpe(id_scarpe)
ON DELETE CASCADE
);

ALTER TABLE giocatore AUTO_INCREMENT=810;

ALTER TABLE scarpe AUTO_INCREMENT=8000;

INSERT INTO pista (is_disponibile, id_impianto) VALUES (1, 101);
INSERT INTO pista (is_disponibile, id_impianto) VALUES (1, 101);
INSERT INTO pista (is_disponibile, id_impianto) VALUES (1, 101);
INSERT INTO pista (is_disponibile, id_impianto) VALUES (1, 101);
INSERT INTO pista (is_disponibile, id_impianto) VALUES (1, 101);
INSERT INTO pista (is_disponibile, id_impianto) VALUES (1, 102);
INSERT INTO pista (is_disponibile, id_impianto) VALUES (1, 102);
INSERT INTO pista (is_disponibile, id_impianto) VALUES (1, 102);
INSERT INTO pista (is_disponibile, id_impianto) VALUES (1, 102);
INSERT INTO pista (is_disponibile, id_impianto) VALUES (1, 102);
INSERT INTO pista (is_disponibile, id_impianto) VALUES (1, 103);
INSERT INTO pista (is_disponibile, id_impianto) VALUES (1, 103);
INSERT INTO pista (is_disponibile, id_impianto) VALUES (1, 103);
INSERT INTO pista (is_disponibile, id_impianto) VALUES (1, 103);
INSERT INTO pista (is_disponibile, id_impianto) VALUES (1, 103);

INSERT INTO giocatore (nome_giocatore, cognome_giocatore, data_di_nascita, email, telefono) VALUES ('Mario', 'Rossi', '1990-01-01', 'mario.rossi@email.com', '333-333-3333');
INSERT INTO giocatore (nome_giocatore, cognome_giocatore, data_di_nascita, email, telefono) VALUES ('Luca', 'Bianchi', '1995-05-05', 'luca.bianchi@email.com', '444-444-4444');
INSERT INTO giocatore (nome_giocatore, cognome_giocatore, data_di_nascita, email, telefono) VALUES ('Chiara', 'Verde', '1985-12-25', 'chiara.verde@email.com', '555-555-5555');
INSERT INTO giocatore (nome_giocatore, cognome_giocatore, data_di_nascita, email, telefono) VALUES ('Francesco', 'Neri', '1980-07-07', 'francesco.neri@email.com', '666-666-6666');
INSERT INTO giocatore (nome_giocatore, cognome_giocatore, data_di_nascita, email, telefono) VALUES ('Elena', 'Gialla', '1995-03-03', 'elena.gialla@email.com', '777-777-7777');
INSERT INTO giocatore (nome_giocatore, cognome_giocatore, data_di_nascita, email, telefono) VALUES ('Giovanni', 'Blu', '2000-09-09', 'giovanni.blu@email.com', '888-888-8888');
INSERT INTO giocatore (nome_giocatore, cognome_giocatore, data_di_nascita, email, telefono) VALUES ('Sara', 'Rosa', '1992-02-29', 'sara.rosa@email.com', '999-999-9999');
INSERT INTO giocatore (nome_giocatore, cognome_giocatore, data_di_nascita, email, telefono) VALUES ('Alessandro', 'Arancione', '1998-06-06', 'alessandro.arancione@email.com', '111-111-1111');
INSERT INTO giocatore (nome_giocatore, cognome_giocatore, data_di_nascita, email, telefono) VALUES ('Paola', 'Viola', '1985-10-10', 'paola.viola@email.com', '222-222-2222');
INSERT INTO giocatore (nome_giocatore, cognome_giocatore, data_di_nascita, email, telefono) VALUES ('Stefano', 'Marrone', '1995-08-08', 'stefano.marrone@email.com', '333-333-3333');
INSERT INTO giocatore (nome_giocatore, cognome_giocatore, data_di_nascita, email, telefono) VALUES ('Pierluigi', 'Marazzi', '1972-04-09', 'pierluigimarazzi@example.net', '+39 0984046497');
INSERT INTO giocatore (nome_giocatore, cognome_giocatore, data_di_nascita, email, telefono) VALUES ('Romina', 'Inzaghi', '1990-05-05', 'rominainzaghi@example.net', '+39 37779198129');
INSERT INTO giocatore (nome_giocatore, cognome_giocatore, data_di_nascita, email, telefono) VALUES ('Valentina', 'Battisti', '2012-10-23', 'vvbattisti@example.com', '093245441');
INSERT INTO giocatore (nome_giocatore, cognome_giocatore, data_di_nascita, email, telefono) VALUES ('Massimo', 'Branciforte', '1989-09-12', 'ecorbo@example.org', '33652364117');
INSERT INTO giocatore (nome_giocatore, cognome_giocatore, data_di_nascita, email, telefono) VALUES ('Flavia', 'Bova', '2021-06-16', 'flaviabov@example.net', '0243601134');
INSERT INTO giocatore (nome_giocatore, cognome_giocatore, data_di_nascita, email, telefono) VALUES ('Virgilio', 'Sagese', '1990-04-05', 'virgilio.sagese@example.com', '378369367');
INSERT INTO giocatore (nome_giocatore, cognome_giocatore, data_di_nascita, email, telefono) VALUES ('Lamberto', 'Bignardi', '1973-06-26', 'grifeodolores@example.org', '+39 09538704836');
INSERT INTO giocatore (nome_giocatore, cognome_giocatore, data_di_nascita, email, telefono) VALUES ('Luigina', 'Perini', '1975-10-31', 'carnerafilippo@example.net', '+39 04452189386');
INSERT INTO giocatore (nome_giocatore, cognome_giocatore, data_di_nascita, email, telefono) VALUES ('Stefano', 'Redi', '1987-04-11', 'qbottaro@example.com', '+39 05654428335');
INSERT INTO giocatore (nome_giocatore, cognome_giocatore, data_di_nascita, email, telefono) VALUES ('Bruno', 'Tolentino', '2011-01-26', 'ninocagnin@example.com', '018417555');

INSERT INTO partita (ora_inizio, tenuta_il, giocatore, tot_score, ora_fine, id_pista) VALUES ('10:00:00', '2022-10-01', 810, 136, '11:00:00', 1);
INSERT INTO partita (ora_inizio, tenuta_il, giocatore, tot_score, ora_fine, id_pista) VALUES ('14:00:00', '2022-10-02', 811, 153, '15:00:00', 2);
INSERT INTO partita (ora_inizio, tenuta_il, giocatore, tot_score, ora_fine, id_pista) VALUES ('09:00:00', '2022-10-03', 812, 250, '10:00:00', 3);
INSERT INTO partita (ora_inizio, tenuta_il, giocatore, tot_score, ora_fine, id_pista) VALUES ('13:00:00', '2022-10-04', 813, 300, '14:00:00', 4);
INSERT INTO partita (ora_inizio, tenuta_il, giocatore, tot_score, ora_fine, id_pista) VALUES ('15:00:00', '2022-10-05', 814, 81, '16:00:00', 1);
INSERT INTO partita (ora_inizio, tenuta_il, giocatore, tot_score, ora_fine, id_pista) VALUES ('11:00:00', '2022-10-06', 815, 72, '12:00:00', 2);
INSERT INTO partita (ora_inizio, tenuta_il, giocatore, tot_score, ora_fine, id_pista) VALUES ('16:00:00', '2022-12-07', 816, 270, '17:00:00', 3);
INSERT INTO partita (ora_inizio, tenuta_il, giocatore, tot_score, ora_fine, id_pista) VALUES ('09:00:00', '2022-05-08', 817, 32, '10:00:00', 4);
INSERT INTO partita (ora_inizio, tenuta_il, giocatore, tot_score, ora_fine, id_pista) VALUES ('14:00:00', '2022-05-09', 818, 35, '15:00:00', 7);
INSERT INTO partita (ora_inizio, tenuta_il, giocatore, tot_score, ora_fine, id_pista) VALUES ('16:00:00', '2022-07-10', 819, 60, '17:00:00', 7);
INSERT INTO partita (ora_inizio, tenuta_il, giocatore, tot_score, ora_fine, id_pista) VALUES ('10:00:00', '2022-03-11', 820, 95, '11:00:00', 7);
INSERT INTO partita (ora_inizio, tenuta_il, giocatore, tot_score, ora_fine, id_pista) VALUES ('13:00:00', '2022-08-12', 821, 57, '14:00:00', 8);
INSERT INTO partita (ora_inizio, tenuta_il, giocatore, tot_score, ora_fine, id_pista) VALUES ('09:00:00', '2022-10-05', 822, 85, '10:00:00', 12);
INSERT INTO partita (ora_inizio, tenuta_il, giocatore, tot_score, ora_fine, id_pista) VALUES ('15:00:00', '2022-01-14', 823, 67, '16:00:00', 12);
INSERT INTO partita (ora_inizio, tenuta_il, giocatore, tot_score, ora_fine, id_pista) VALUES ('11:00:00', '2022-01-15', 824, 165, '12:00:00', 8);
INSERT INTO partita (ora_inizio, tenuta_il, giocatore, tot_score, ora_fine, id_pista)VALUES ('14:00:00', '2022-05-09', 825, 35, '15:00:00', 8);
INSERT INTO partita (ora_inizio, tenuta_il, giocatore, tot_score, ora_fine, id_pista) VALUES ('16:00:00', '2022-07-10', 826, 60, '17:00:00', 7);
INSERT INTO partita (ora_inizio, tenuta_il, giocatore, tot_score, ora_fine, id_pista) VALUES ('10:00:00', '2022-03-11', 827, 95, '11:00:00', 7);
INSERT INTO partita (ora_inizio, tenuta_il, giocatore, tot_score, ora_fine, id_pista) VALUES ('13:00:00', '2022-08-12', 828, 57, '14:00:00', 15);
INSERT INTO partita (ora_inizio, tenuta_il, giocatore, tot_score, ora_fine, id_pista) VALUES ('09:00:00', '2022-10-05', 829, 85, '10:00:00', 5);
INSERT INTO partita (ora_inizio, tenuta_il, giocatore, tot_score, ora_fine, id_pista) VALUES ('15:00:00', '2022-01-14', 810, 67, '16:00:00', 2);
INSERT INTO partita (ora_inizio, tenuta_il, giocatore, tot_score, ora_fine, id_pista) VALUES ('11:00:00', '2022-01-15', 811, 165, '12:00:00', 8);

INSERT INTO scarpe (prezzo, taglia, marca, giocatore) VALUES (5.00, 42, "Nike", 810);
INSERT INTO scarpe (prezzo, taglia, marca, giocatore) VALUES (5.50, 44, "Adidas", 811);
INSERT INTO scarpe (prezzo, taglia, marca, giocatore) VALUES (6.00, 43, "Puma", 812);
INSERT INTO scarpe (prezzo, taglia, marca, giocatore) VALUES (5.00, 42, "Nike", 813);
INSERT INTO scarpe (prezzo, taglia, marca, giocatore) VALUES (5.50, 45, "Adidas", 814);
INSERT INTO scarpe (prezzo, taglia, marca, giocatore) VALUES (6.00, 46, "Puma", 815);
INSERT INTO scarpe (prezzo, taglia, marca, giocatore) VALUES (5.00, 41, "Nike", 816);
INSERT INTO scarpe (prezzo, taglia, marca, giocatore) VALUES (5.50, 40, "Adidas", 817);
INSERT INTO scarpe (prezzo, taglia, marca, giocatore) VALUES (6.00, 42, "Puma", 818);
INSERT INTO scarpe (prezzo, taglia, marca, giocatore) VALUES (5.00, 44, "Nike", 819);
INSERT INTO scarpe (prezzo, taglia, marca, giocatore) VALUES (5.50, 43, "Adidas", 820);
INSERT INTO scarpe (prezzo, taglia, marca, giocatore) VALUES (6.00, 42, "Puma", 821);
INSERT INTO scarpe (prezzo, taglia, marca, giocatore) VALUES (5.00, 45, "Nike", 822);
INSERT INTO scarpe (prezzo, taglia, marca, giocatore) VALUES (5.50, 41, "Adidas", 823);
INSERT INTO scarpe (prezzo, taglia, marca, giocatore) VALUES (6.00, 42, "Puma", 824);

INSERT INTO noleggio (giocatore, id_scarpe, ora_inizio, data, is_consegnato) VALUES (810, 8000, '10:00','2022-10-01', 1);
INSERT INTO noleggio (giocatore, id_scarpe, ora_inizio, data, is_consegnato) VALUES (811, 8001, '11:00','2022-10-02', 1);
INSERT INTO noleggio (giocatore, id_scarpe, ora_inizio, data, is_consegnato) VALUES (812, 8002, '09:00','2022-10-03', 0);
INSERT INTO noleggio (giocatore, id_scarpe, ora_inizio, data, is_consegnato) VALUES (813, 8003, '13:00','2022-10-04', 1);
INSERT INTO noleggio (giocatore, id_scarpe, ora_inizio, data, is_consegnato) VALUES (814, 8004, '15:00','2022-10-05', 0);
INSERT INTO noleggio (giocatore, id_scarpe, ora_inizio, data, is_consegnato) VALUES (815, 8005, '11:00','2022-10-06',1);
INSERT INTO noleggio (giocatore, id_scarpe, ora_inizio, data, is_consegnato) VALUES (816, 8006, '16:00','2022-12-07', 1);
INSERT INTO noleggio (giocatore, id_scarpe, ora_inizio, data, is_consegnato) VALUES (817, 8007, '09:00','2022-05-08', 0);
INSERT INTO noleggio (giocatore, id_scarpe, ora_inizio, data, is_consegnato) VALUES (818, 8008, '14:00','2022-05-09', 1);
INSERT INTO noleggio (giocatore, id_scarpe, ora_inizio, data, is_consegnato) VALUES (819, 8009, '16:00','2022-07-10', 0);
INSERT INTO noleggio (giocatore, id_scarpe, ora_inizio, data, is_consegnato) VALUES (820, 8010, '10:00','2022-03-11', 1);
INSERT INTO noleggio (giocatore, id_scarpe, ora_inizio, data, is_consegnato) VALUES (821, 8011, '13:00','2022-08-12', 1);
INSERT INTO noleggio (giocatore, id_scarpe, ora_inizio, data, is_consegnato) VALUES (822, 8012, '09:00','2022-10-05', 0);
INSERT INTO noleggio (giocatore, id_scarpe, ora_inizio, data, is_consegnato) VALUES (823, 8013, '15:00','2022-01-14', 1);
INSERT INTO noleggio (giocatore, id_scarpe, ora_inizio, data, is_consegnato) VALUES (824, 8014, '20:00','2022-01-15', 0);

INSERT INTO abbonamento (valido_dal, scade_il, tipologia_abbonamento, giocatore) VALUES ('2023-01-01', '2023-01-31', 'mensile', 810);
INSERT INTO abbonamento (valido_dal, scade_il, tipologia_abbonamento, giocatore) VALUES ('2023-01-01', '2023-01-31', 'mensile', 811);
INSERT INTO abbonamento (valido_dal, scade_il, tipologia_abbonamento, giocatore) VALUES ('2022-03-01', '2022-04-01', 'mensile', 812);
INSERT INTO abbonamento (valido_dal, scade_il, tipologia_abbonamento, giocatore) VALUES ('2023-01-01', '2023-04-01', 'trimestrale', 813);
INSERT INTO abbonamento (valido_dal, scade_il, tipologia_abbonamento, giocatore) VALUES ('2023-01-01', '2023-04-01', 'trimestrale', 814);
INSERT INTO abbonamento (valido_dal, scade_il, tipologia_abbonamento, giocatore) VALUES ('2022-01-01', '2022-12-31', 'annuale', 815);
INSERT INTO abbonamento (valido_dal, scade_il, tipologia_abbonamento, giocatore) VALUES ('2023-01-01', '2023-12-31', 'annuale', 816);
INSERT INTO abbonamento (valido_dal, scade_il, tipologia_abbonamento, giocatore) VALUES ('2022-01-01', '2022-12-31', 'annuale', 817);
INSERT INTO abbonamento (valido_dal, scade_il, tipologia_abbonamento, giocatore) VALUES ('2022-01-01', '2022-12-31', 'annuale', 818);
INSERT INTO abbonamento (valido_dal, scade_il, tipologia_abbonamento, giocatore) VALUES ('2023-01-01', '2023-12-31', 'annuale', 819);








