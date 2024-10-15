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

CREATE TABLE `utente` (
  `id` bigint(20) NOT NULL PRIMARY KEY,
  `codiceFiscale` varchar(20) NOT NULL,
  `idCentro` int(11) NOT NULL,
  `nome_sport` varchar(30) NOT NULL,
  `praticato_a_livello` enum('professionistico','dilettantistico') NOT NULL,
  `sesso` enum('maschile','femminile') NOT NULL,
  `annidiEta` int(10) UNSIGNED NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

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
