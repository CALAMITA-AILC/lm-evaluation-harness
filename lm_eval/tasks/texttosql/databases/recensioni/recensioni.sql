CREATE DATABASE IF NOT EXISTS recensioni;

USE recensioni;

CREATE TABLE `centrosportivo` (
  `id` int(11) NOT NULL PRIMARY KEY,
  `nome` varchar(30) DEFAULT NULL,
  `indirizzo` varchar(100) NOT NULL,
  `comune` varchar(20) NOT NULL,
  `telefono` varchar(15) NOT NULL,
  `posta_elettronica` varchar(50) NOT NULL,
  `portale_online` varchar(40) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO `centrosportivo` (`id`, `nome`, `indirizzo`, `comune`, `telefono`, `posta_elettronica`, `portale_online`) VALUES (1, 'Sport Center', 'via Italo 869', 'Roma', '068609452', 'SportCenter@hotmail.com', 'www.SportCenter.it'), (2, 'Sportessere', 'via Carducci 45', 'Frosinone', '068609487', 'Sportessere@tin.com', 'www.Sportessere.it'), (3, 'Svago & Forma', 'piazza Repubblica 78', 'Velletri', '0894609472', 'Svago.Forma@gmail.com', 'SvagoForma.it'), (4, 'Atletigame', 'via dei Draghi 345', 'Roma', '088609463', 'Atletigame@libero.it', 'Atletigame.com'), (5, 'Country Club', 'via Pozzi 34', 'Napoli', '089306789', 'countryclub@wind.com', 'www.CountryClubCentro.com'), (6, 'Salerno Sport', 'via Vietri 56', 'Salerno', '089407123', 'centrosportivo.salerno56@outlook.com', NULL), (7, 'Lucen Sport', 'via Conte Max 123', 'Roma', '088678574', 'lucensport@hotmail.com', 'www.lucensport.it'), (8, 'Appia Sport Village', 'via Appia Moderna 777', 'Roma', '088645328', 'appiasport.village@gmail.it', 'www.appiasport.com');

CREATE TABLE `iscritto` (
  `codiceFiscale` varchar(20) NOT NULL PRIMARY KEY,
  `idCentro` int(11) NOT NULL,
  `nome` varchar(20) NOT NULL,
  `cognome` varchar(30) NOT NULL,
  `sesso` enum('maschile','femminile') NOT NULL,
  `dataNascita` date NOT NULL,
  `indirizzo_di_residenza` varchar(100) NOT NULL,
  `citta` varchar(20) NOT NULL,
  `recapito_telefonico` varchar(15) NOT NULL,
  `email` varchar(50) DEFAULT NULL,
  `codiceCertificato` int(11) DEFAULT NULL,
   FOREIGN KEY (`idCentro`) REFERENCES  `centrosportivo`(`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO `iscritto` (`codiceFiscale`, `idCentro`, `nome`, `cognome`, `sesso`, `dataNascita`, `indirizzo_di_residenza`, `citta`, `recapito_telefonico`, `email`, `codiceCertificato`) VALUES ('BRRRCC70C50G482B', 7, 'Rebecca', 'Barry', 'femminile', '1970-03-10', 'via Regina Elisabetta 12', 'Roma', '06867478', 'rebeccabarry@gmail.com', 134524), ('CHRSFO05S52H501J', 1, 'Sofia', 'Chiari', 'femminile', '2005-11-12', 'via Italo 560', 'Roma', '06863521', 'sofiachiari@hotmail.it', 98524), ('CRTLSN73C41A345V', 7, 'Alessandra', 'Cortesi', 'femminile', '1973-03-01', 'via dell’Angelo 45', 'Roma', '06860023', 'alessandracortesi@libero.it', 54324), ('DSNPLA75R01H501K', 8, 'Paolo', 'De Santis', 'maschile', '1975-10-01', 'via Tuscolo 99', 'Roma', '06869526', 'paolodesantis@wind.it', 17784), ('DSTLCU02E49L719G', 3, 'Lucia', 'Destro', 'femminile', '2002-05-09', 'via Argentina 126', 'Velletri', '06849723', 'luciadestro@libero.it', 89982), ('FNTMRC03T24H501H', 4, 'Marco', 'Fanti', 'maschile', '2003-12-24', 'via Lucio Fontana 34', 'Roma', '06869505', 'marcofanti03@gmail.com', 9866), ('FRRMRC99D03D810Y', 2, 'Marco', 'Ferri', 'maschile', '1999-04-03', 'via Veneti 121', 'Frosinone', '06859581', 'marcoferri@gmail.it', 18294), ('PRNGCM00D24D612C', 6, 'Giacomo', 'Parini', 'maschile', '2000-04-24', 'via Brega 93', 'Roma', '06869771', NULL, 1524), ('PSCDGI83P09H703Z', 5, 'Diego', 'Pascoli', 'maschile', '1983-09-09', 'via dei Cardinali 126', 'Caserta', '06959176', 'diegopascoli0909@gmail.com', 1536), ('RMOFNC65M12F839U', 5, 'Francesco', 'Roma', 'maschile', '1965-08-12', 'via De Curtis 222', 'Napoli', '069667213', NULL, 15654), ('RMPTZN94H53D969B', 1, 'Tiziana', 'Rampini', 'maschile', '1979-07-05', 'via del Gladiatore 540', 'Roma', '06869001', 'tizianarampini@hotmail.it', 78124), ('RSSJPL67A30F205H', 1, 'Jean Paul', 'Rossi', 'maschile', '1967-01-30', 'via Armenia 120', 'Roma', '06869521', 'jeanpaulrossi@gmail.com', 44524), ('RSSRRT56C61H501C', 1, 'Roberta', 'Rossi', 'femminile', '1956-03-21', 'via Casali 12', 'Roma', '06869589', 'robertarossi@gmail.it', 1523), ('STNNDR92D58H703R', 6, 'Andrea', 'Stankovic', 'femminile', '1992-04-18', 'via Reni 15', 'Amalfi', '06459832', 'andreastankovic@wind.it', 2524), ('TRPDRA49B19G273G', 4, 'Dario', 'Trapani', 'maschile', '1949-02-19', 'piazza Risorgimento 126', 'Roma', '06865423', 'dariotrapani@gmail.com', 8924), ('VLTGNN93D23A509S', 5, 'Giovanni', 'Veltri', 'maschile', '1993-04-23', 'piazza Caruso 4', 'Napoli', '06969421', 'giovanniveltri@tin.com', 68524), ('VRDMSM79L05F839F', 8, 'Massimo', 'Verdini', 'maschile', '1979-07-05', 'via Appia Vecchia 345', 'Roma', '06869651', 'massimoverdini@outlook.com', 65524);

CREATE TABLE `utente` (
  `id` bigint(20) NOT NULL PRIMARY KEY,
  `codiceFiscale` varchar(20) NOT NULL,
  `idCentro` int(11) NOT NULL,
  `nome_sport` varchar(30) NOT NULL,
  `praticato_a_livello` enum('professionistico','dilettantistico') NOT NULL,
  `sesso` enum('maschile','femminile') NOT NULL,
  `annidiEta` int(10) UNSIGNED NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO `utente` (`id`, `codiceFiscale`, `idCentro`, `nome_sport`, `praticato_a_livello`, `sesso`, `annidiEta`) VALUES (1, 'RSSRRT56C61H501C', 1, 'tennis', 'professionistico', 'femminile', 66), (2, 'RSSRRT56C61H501C', 3, 'nuoto', 'dilettantistico', 'femminile', 66), (3, 'FRRMRC99D03D810Y', 2, 'calcio', 'professionistico', 'maschile', 23), (4, 'FRRMRC99D03D810Y', 2, 'pallacanestro', 'dilettantistico', 'maschile', 23), (5, 'FRRMRC99D03D810Y', 4, 'pallanuoto', 'professionistico', 'maschile', 23), (6, 'DSTLCU02E49L719G', 3, 'calcio', 'professionistico', 'femminile', 20), (7, 'DSNPLA75R01H501K', 8, 'pallavolo', 'professionistico', 'maschile', 47), (8, 'DSNPLA75R01H501K', 7, 'pallanuoto', 'dilettantistico', 'maschile', 47), (9, 'STNNDR92D58H703R', 6, 'tennis', 'professionistico', 'femminile', 30), (10, 'STNNDR92D58H703R', 4, 'nuoto', 'professionistico', 'femminile', 66), (11, 'STNNDR92D58H703R', 2, 'tennis', 'professionistico', 'femminile', 66), (12, 'VLTGNN93D23A509S', 5, 'tennis', 'professionistico', 'maschile', 29), (13, 'VLTGNN93D23A509S', 6, 'pugilato', 'dilettantistico', 'maschile', 29), (14, 'BRRRCC70C50G482B', 7, 'pallavolo', 'dilettantistico', 'femminile', 52), (15, 'CHRSFO05S52H501J', 1, 'pallavolo', 'professionistico', 'femminile', 17), (16, 'CHRSFO05S52H501J', 7, 'pallavolo', 'professionistico', 'femminile', 17), (17, 'CHRSFO05S52H501J', 7, 'pallanuoto', 'professionistico', 'femminile', 17), (18, 'RSSJPL67A30F205H', 1, 'nuoto', 'dilettantistico', 'maschile', 55), (19, 'RSSJPL67A30F205H', 5, 'nuoto', 'dilettantistico', 'maschile', 55), (20, 'VRDMSM79L05F839F', 8, 'pugilato', 'dilettantistico', 'maschile', 43), (21, 'VRDMSM79L05F839F', 6, 'pugilato', 'dilettantistico', 'maschile', 43), (22, 'PSCDGI83P09H703Z', 5, 'pallacanestro', 'dilettantistico', 'maschile', 39), (23, 'RMOFNC65M12F839U', 5, 'pallacanestro', 'dilettantistico', 'maschile', 57), (24, 'RMOFNC65M12F839U', 5, 'pallavolo', 'dilettantistico', 'maschile', 57), (25, 'TRPDRA49B19G273G', 4, 'pugilato', 'dilettantistico', 'maschile', 73), (26, 'CRTLSN73C41A345V', 7, 'pugilato', 'dilettantistico', 'femminile', 49), (27, 'RMPTZN94H53D969B', 1, 'pugilato', 'professionistico', 'femminile', 43), (28, 'FNTMRC03T24H501H', 4, 'pugilato', 'professionistico', 'maschile', 19), (29, 'FNTMRC03T24H501H', 6, 'golf', 'dilettantistico', 'maschile', 19), (30, 'PRNGCM00D24D612C', 6, 'golf', 'professionistico', 'maschile', 22);

CREATE TABLE `recensione` (
  `id` bigint(20) NOT NULL PRIMARY KEY,
  `idUtente` bigint(20) NOT NULL,
  `idCentro` int(11) DEFAULT NULL,
  `data_effettuazione` date NOT NULL,
  `stato_struttura` enum('1','2','3','4','5') NOT NULL,
  `gestione` enum('1','2','3','4','5') NOT NULL,
  `qualifica_staffTecnico` enum('1','2','3','4','5') NOT NULL,
  FOREIGN KEY (`idUtente`) REFERENCES  `utente`(`id`),
  FOREIGN KEY (`idCentro`) REFERENCES  `centrosportivo`(`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO `recensione` (`id`, `idUtente`, `idCentro`, `data_effettuazione`, `stato_struttura`, `gestione`, `qualifica_staffTecnico`) VALUES (1, 1, 1, '2020-05-09', '3', '3', '4'), (2, 2, 3, '2020-01-19', '3', '5', '4'), (3, 4, 2, '2021-03-28', '5', '3', '4'), (4, 9, 6, '2019-10-30', '4', '5', '4'), (5, 16, 7, '2021-04-06', '3', '2', '4'), (6, 28, 6, '2020-11-09', '5', '1', '4'), (7, 30, 6, '2021-04-10', '3', '1', '2'), (8, 22, 5, '2021-01-03', '1', '5', '4'), (9, 1, 1, '2020-12-14', '3', '3', '4'), (10, 5, 4, '2020-07-31', '3', '3', '4'), (11, 1, 1, '2021-09-03', '3', '3', '4'), (12, 18, 1, '2020-06-11', '3', '5', '4'), (13, 14, 7, '2021-05-23', '1', '3', '4'), (14, 27, 1, '2020-08-19', '3', '3', '5');
