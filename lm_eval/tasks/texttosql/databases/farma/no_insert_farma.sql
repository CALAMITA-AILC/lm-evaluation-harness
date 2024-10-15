CREATE DATABASE IF NOT EXISTS farma;

USE farma;

CREATE TABLE `farmaco` (
  `Codice` int(11) NOT NULL PRIMARY KEY,
  `Azienda` varchar(50) NOT NULL,
  `NomeCommerciale` varchar(80) NOT NULL,
  `Prezzo_EUR` float NOT NULL,
  `Posologia_ml` varchar(120) DEFAULT NULL,
  `IndicazioneTerapeutica` varchar(120) DEFAULT NULL,
  `PrincipioAttivo` varchar(50) NOT NULL,
  `regimeFornitura` varchar(5) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `medico` (
  `Matricola` varchar(13) NOT NULL PRIMARY KEY,
  `Nome` varchar(32) DEFAULT NULL,
  `Cognome` varchar(40) DEFAULT NULL,
  `recapito_tel` varchar(12) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `paziente` (
  `codice_fiscale` varchar(32) NOT NULL PRIMARY KEY,
  `nome_paz` varchar(32) DEFAULT NULL,
  `cognome_paz` varchar(38) DEFAULT NULL,
  `curante` varchar(13) DEFAULT NULL,
  FOREIGN KEY (`curante`) REFERENCES  `medico`(`Matricola`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `ricetta` (
  `Identificativo` varchar(32) NOT NULL PRIMARY KEY,
  `Data_Prescrizione` date DEFAULT NULL,
  `CodiceFiscale` varchar(32) NOT NULL,
  `Matricola` varchar(13) DEFAULT NULL,
  `Cod_d` int(11) NOT NULL,
  FOREIGN KEY (`CodiceFiscale`) REFERENCES  `paziente`(`codice_fiscale`),
  FOREIGN KEY (`Matricola`) REFERENCES  `medico`(`Matricola`),
  FOREIGN KEY (`Cod_d`) REFERENCES  `farmaco`(`Codice`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;