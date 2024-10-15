CREATE DATABASE IF NOT EXISTS pratica;

USE pratica;

CREATE TABLE `amministrativo` (
  `id_amministrativi` int NOT NULL PRIMARY KEY,
  `nome` varchar(24) NOT NULL,
  `cognome` varchar(24) NOT NULL,
  `nascita` date NOT NULL,
  `assunzione` date NOT NULL,
  `succursale` varchar(64) NOT NULL,
  `manager` int NOT NULL
) ENGINE=INNODB;

CREATE TABLE `assistente` (
  `id_assistente` int NOT NULL PRIMARY KEY,
  `nome` varchar(24) NOT NULL,
  `cognome` varchar(24) NOT NULL,
  `nascita` date NOT NULL,
  `assunzione` date NOT NULL,
  `inizio_assistenza` date NOT NULL
) ENGINE=INNODB;

CREATE TABLE `pratica` (
  `codice_pratica` int NOT NULL PRIMARY KEY,
  `data_consegna` date NOT NULL,
  `data_approvazione` date NOT NULL,
  `amministrativo` int NOT NULL,
  `assistente` int NOT NULL,
  FOREIGN KEY (`assistente`) REFERENCES  `assistente`(`id_assistente`),
  FOREIGN KEY (`amministrativo`) REFERENCES  `amministrativo`(`id_amministrativi`)
) ENGINE=INNODB;
