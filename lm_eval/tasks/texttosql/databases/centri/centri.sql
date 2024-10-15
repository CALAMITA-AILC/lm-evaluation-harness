CREATE DATABASE IF NOT EXISTS centri;

USE centri;

CREATE TABLE IF NOT EXISTS centrodonazione(
id  bigint,
nome_cd  varchar(30),
indirizzo  varchar(100),
citta  varchar(20),
telefono  varchar(15),
email  varchar(50),
sitoWeb  varchar(40),
PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS centrosanitario(
id  bigint,
nome_csn  varchar(30),
indirizzo  varchar(100),
citta  varchar(20),
is_convenzionato boolean,
id_convenzione bigint,
portale  varchar(40),
telefono  varchar(15),
email  varchar(50),

PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS centrosportivo(
id  int,
nome_csp  varchar(30),
indirizzo  varchar(100),
citta  varchar(20),
is_agonistico boolean,
telefono  varchar(15),
email  varchar(50),
sitoWeb  varchar(40),
PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS convenzione_cd(
idDonazione  bigint,
idSportivo  int,
data_inizio_convenzione_d  date,
is_annuale boolean,
PRIMARY KEY (idDonazione,idSportivo),
FOREIGN KEY ( idDonazione)   REFERENCES centrodonazione(id),
FOREIGN KEY ( idSportivo)   REFERENCES centrosportivo(id)
);

CREATE TABLE IF NOT EXISTS convenzione_cs(
idSanitario  bigint,
idSportivo  int,
data_inizio_convenzione_s  date,
PRIMARY KEY (idSanitario,idSportivo),
FOREIGN KEY ( idSanitario)   REFERENCES centrosanitario(id),
FOREIGN KEY ( idSportivo)   REFERENCES centrosportivo(id)
);

INSERT INTO `centrodonazione` (`id`, `nome_cd`, `indirizzo`, `citta`, `telefono`, `email`, `sitoWeb`) VALUES (1, 'AVIS', 'via Golia 85', 'Roma', '06862324', 'Avis.viagolia@hotmail.com', 'www.avisGolia.it'), (2, 'Centro Braschi', 'via Dainelli 135', 'Frosinone', '068617586', 'CentroBraschi@tin.com', 'www.BraschiDonazioni.it'), (3, 'Associazione Donatori', 'via Episcopo450', 'Velletri', '08943583', 'AssociazioneDonatori@gmail.com', 'ADEpiscopo.com'), (4, 'AVIS', 'via dei Angiolieri 260', 'Roma', '08854731', 'Avis.vianagiolieri@libero.it', 'AvisAngiolieri.com'), (5, 'Centro EMA', 'via Sisto 55', 'Napoli', '08934416', 'CentroEma@wind.com', 'www.EMAdonazioni.com'), (6, 'AVIS', 'via Greco 231', 'Salerno', '08942247', 'AvisGreco@outlook.com', NULL), (7, 'Centro Nazionale Donazioni', 'via Rita 22', 'Roma', '088635571', 'CentroNazionaleDonazioni@hotmail.com', 'www.CNdonazioni.org'), (8, 'AVIS', 'via Appia Moderna 40', 'Roma', '08868532', 'AppiAvis@gmail.it', 'www.appiavis.com'), (9, NULL, 'via Scarselli 146', 'Roma', '08862431', 'donazioniviaScarselli@gmail.it', 'www.donareFabene.it'), (10, 'Progetto EMAS', 'via Calvairate 2', 'Roma', '08809963', 'progettoEMAS@gmail.it', 'www.ProgettoEMAS.com');

INSERT INTO `centrosanitario`(`id`, `nome_csn`, `indirizzo`, `citta`, `is_convenzionato`, `id_convenzione`, `portale`, `telefono`, `email`) VALUES (1, 'C&V', 'via Davide 865', 'Roma', false, NULL,  'www.C&Vcentro.it', '06862826', 'C&V.centroSanitario@hotmail.com'), (2, 'Centro Pugliese', 'via Palestro 365', 'Frosinone', true, 328749873,'CentroPugliese@tin.com',  '068609487', 'www.CentroPugliese.it'), (3, 'Fisiosports', 'piazza San Rocco 90', 'Velletri', true, 32749847, 'Fisiosports .it', '08947424', 'Fisiosports@gmail.com'), (4, 'Polispecialistica Rossi', 'via dei Mattarella 644', 'Roma', false, NULL, 'Polispecialistica Rossi .com', '08869731', 'Polispecialistica Rossi@libero.it'), (5, 'Studio Corpore Sano', 'via Lancellotti 34', 'Napoli', false, NULL, 'www.CorporeSanoCentro.com', '08930215', 'CorporeSano@wind.com'), (6, 'Studio Paideia', 'via Cardinal Franchi 21', 'Salerno', true, 3247236, NULL, '08948145', 'StudioPaideia.salerno56@outlook.com'), (7, 'Almamed', 'via Boldi 654', 'Roma', false, NULL, 'www.almamed.it', '088678574', 'AlmaMed@hotmail.com'), (8, 'Poliambulatorio Cecconi', 'via Appia Moderna\r\n157', 'Roma', true, 67463856, 'www.poliCecconi.com', '08867634', 'poliCecconi@gmail.it'), (9, NULL, 'via Bastille 282', 'Roma', true, 7348783, 'www.studioBastille.com', '08862261', 'studioBastille@gmail.it'),  (10, 'Progetto Salute srl', 'via Appia Moderna 4', 'Roma', false, NULL, 'www.ProgettoSalute.com','08865963', 'obbiettivoSalute@gmail.it');

INSERT INTO `centrosportivo` (`id`, `nome_csp`, `indirizzo`, `citta`, `is_agonistico`, `telefono`, `email`, `sitoWeb`) VALUES (1, 'Sport Center', 'via Italo 869', 'Roma', false, '068609452', 'SportCenter@hotmail.com', 'www.SportCenter.it'), (2, 'Sportessere', 'via Carducci 45', 'Frosinone', false, '068609487', 'Sportessere@tin.com', 'www.Sportessere.it'),(3, 'Svago & Forma', 'piazza Repubblica 78', 'Velletri', true, '0894609472', 'Svago.Forma@gmail.com', 'SvagoForma.it'), (4, 'Atletigame', 'via dei Draghi 345', 'Roma', true, '088609463', 'Atletigame@libero.it', 'Atletigame.com'), (5, 'Country\r\nClub', 'via Pozzi 34', 'Napoli', false, '089306789', 'countryclub@wind.com', 'www.CountryClubCentro.com'), (6, 'Salerno Sport', 'via Vietri 56', 'Salerno', true, '089407123', 'centrosportivo.salerno56@outlook.com', NULL), (7, 'Lucen Sport', 'via Conte Max 123', 'Roma', false,  '088678574', 'lucensport@hotmail.com', 'www.lucensport.it'), (8, 'Appia Sport Village', 'via Appia Moderna 777', 'Roma', true, '088645328', 'appiasport.village@gmail.it', 'www.appiasport.com');

INSERT INTO `convenzione_cd` (`idDonazione`, `idSportivo`, `data_inizio_convenzione_d`, `is_annuale`) VALUES (1, 8, '2021-05-12', true), (2, 2, '2020-05-27', false), (2, 3, '2020-11-03', true), (3, 3, '2019-08-05', true), (4, 1, '2020-08-09', true), (5, 5, '2021-03-20', false), (5, 6, '2020-10-28', true), (6, 6, '2021-01-15', true), (7, 1, '2019-12-01', true), (7, 2, '2019-04-30', false), (8, 4, '2019-10-18', true), (9, 8, '2019-11-16', true), (10, 1, '2020-11-26', true);

INSERT INTO `convenzione_cs` (`idSanitario`, `idSportivo`, `data_inizio_convenzione_s`) VALUES (1, 4, '2021-03-18'), (1, 8, '2020-10-30'), (2, 2, '2021-05-07'), (3, 3, '2020-12-01'), (4, 1, '2020-04-23'), (4, 8, '2019-11-03'), (5, 5, '2021-02-05'), (5, 6, '2020-09-21'), (6, 6, '2020-04-22'), (7, 1, '2019-06-03'), (7, 3, '2019-07-29'), (8, 1, '2020-08-24'), (8, 8, '2021-01-10'), (9, 4, '2020-05-08'), (10, 1, '2020-08-16');

