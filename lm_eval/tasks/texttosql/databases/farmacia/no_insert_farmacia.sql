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

CREATE TABLE `prodotti` (
  `cod_identificativo` varchar(100) CHARACTER SET utf8 NOT NULL PRIMARY KEY,
  `nome_commerciale` varchar(255) NOT NULL,
  `tipologiaProd` varchar(255) NOT NULL,
  `prezzo_eu` decimal(5,2) DEFAULT NULL,
  `casaProduzione` varchar(100) NOT NULL,
  `data_di_rilascio` date NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

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
