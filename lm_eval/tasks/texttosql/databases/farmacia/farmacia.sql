CREATE DATABASE IF NOT EXISTS farmacia;

USE farmacia;


CREATE TABLE `farmacie` (
  `partita_iva` varchar(20) NOT NULL PRIMARY KEY,
  `nome_negozio` varchar(255) NOT NULL,
  `situata_presso` varchar(255) NOT NULL,
  `cap` char(6) NOT NULL,
  `nomeComune` char(25) NOT NULL,
  `inizio_attivita` date NOT NULL,
  `coordinate` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO `farmacie` (`partita_iva`, `nome_negozio`, `situata_presso`, `cap`, `nomeComune`, `inizio_attivita`, `coordinate`) VALUES ('100030816', 'Aleci Vincenzo', 'Via Ettore Scimemi, 6', '00042', 'Roma', '2002-01-07', '37.822417-12.766197'), ('1000320141', 'Farmacia Bacchetta di Bacchetta Elisa e C. s.n.c.', 'Piazza IV novembre,170', '00118', 'Roma', '2019-04-01', '46.136556-9.606605'), ('10005760011', 'CENTRALE SAS DEL DR. ARGANO GUIDO', 'Piazza Leumann,4/A', '00042', 'Roma', '2013-03-01', '45.113867-7.553895'), ('10005770010', 'SACRO CUORE SNC DEI DOTT.RI FERRATO ADOLFO E CRAVINI STEFANIA MARIA', 'Via Nizza, 115', '00044', 'Frascati', '2009-09-01', '45.018047-7.661732'), ('10006140585', 'S. GIUSEPPE DR.SSA ANNA RASULO', 'Via Ciro da Urbino,37', '00042', 'Roma', '2005-01-01', '41.878135-12.540362');

CREATE TABLE `prodotti` (
  `cod_identificativo` varchar(100) CHARACTER SET utf8 NOT NULL PRIMARY KEY,
  `nome_commerciale` varchar(255) NOT NULL,
  `tipologiaProd` varchar(255) NOT NULL,
  `prezzo_eu` decimal(5,2) DEFAULT NULL,
  `casaProduzione` varchar(100) NOT NULL,
  `data_di_rilascio` date NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO `prodotti` (`cod_identificativo`, `nome_commerciale`, `tipologiaProd`, `prezzo_eu`, `casaProduzione`, `data_di_rilascio`) VALUES ('#ART3000', 'LINER IN GEL DI POLIURETANO', 'PROTESI DI ARTO INFERIORE - ALTRE', '4.22', 'OTTO BOCK ITALIA SRL', '2013-04-20'), ('#BLS/BLSL', 'BESTLINER', 'PROTESI DI ARTO INFERIORE - ALTRE', '5.79', 'OTTO BOCK ITALIA SRL', '2008-09-11'), ('#A-6016', 'ADAPTER SLEEVE, MANICOTTO ADATTATORE PER ELETTROCATETERE', 'ELETTROCATETERI PERMANENTI - ACCESSORI', '4.88', 'CARDIAC PACEMAKER INC.A OWNED SUBSIDIARY OF GUIDANT CORP.A OWNED SUBSIDIARY OF BOSTON SCIENTIFIC', '2008-11-28'), ('-AP1700U-', 'FLUIRESPIRA DIDACTYS', 'APPARECCHIATURE PER AEROSOL', '1.06', '3A HEALTH CARE S.R.L.', '2008-09-10'), ('-CRC 51', 'SEDIE DA COMODO FISSE CON/SENZA WC', 'SEDIE PER WC E DOCCIA', '6.51', 'PIAI ORTOTECH SRL', '2008-12-10');

CREATE TABLE `vendite` (
  `id` smallint(5) UNSIGNED NOT NULL PRIMARY KEY,
  `data_fatturazione` date NOT NULL,
  `acquirente` char(16) NOT NULL,
  `negozio` varchar(20) NOT NULL,
  `prodotto` varchar(100) NOT NULL,
  `quantitaVenduta` tinyint(4) NOT NULL,
  `totale` decimal(5,2) NOT NULL,
  FOREIGN KEY (`prodotto`) REFERENCES `prodotti`(`cod_identificativo`),
  FOREIGN KEY (`negozio`) REFERENCES `farmacie`(`partita_iva`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO `vendite` (`id`, `data_fatturazione`, `acquirente`, `negozio`, `prodotto`, `quantitaVenduta`, `totale`) VALUES (1, '2020-03-01', 'FLZPLA13S49B962J', '100030816', '#ART3000', 2, '8.00'), (2, '2020-03-02', 'NDDTTR52E10C794N', '100030816', '-CRC 51', 9, '58.59'), (3, '2020-03-03', 'NSPRMT27T05L228P', '1000320141', '-AP1700U-', 8, '8.00'), (4, '2020-03-04', 'CLVVRC20A04A655B', '10005760011', '#A-6016', 1, '4.88'), (5, '2020-03-05', 'GBBLNZ90L71C313R', '10005770010', '-AP1700U-', 3, '0.00'), (6, '2020-03-11', 'GBBLNZ90L71C313R', '10005770010', '#A-6016', 4, '18.00'), (7, '2020-04-20', 'GBBLNZ90L71C313R', '10006140585', '-CRC 51', 1, '6.51'), (8, '2020-04-11', 'GBBLNZ90L71C313R', '10006140585', '#BLS/BLSL', 2, '11.58');