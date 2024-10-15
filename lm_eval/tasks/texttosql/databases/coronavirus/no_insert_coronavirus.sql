CREATE DATABASE IF NOT EXISTS coronavirus;

USE coronavirus;

CREATE TABLE `persone` (
  `cognome` varchar(30) NOT NULL,
  `nome` varchar(30) NOT NULL,
  `sesso` char(1) NOT NULL,
  `nas_comune` char(4) NOT NULL,
  `data_nascita` date NOT NULL,
  `cod_fiscale` char(16) NOT NULL PRIMARY KEY,
  `res_comune` char(4) DEFAULT NULL,
  `indirizzo` varchar(70) NOT NULL,
  `telefono` varchar(12) NOT NULL,
  `email` varchar(50) NOT NULL,
  `professione` varchar(40) NOT NULL,
  `stato_civile` varchar(10) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `infetti` (
  `id_infezione` int(11) NOT NULL PRIMARY KEY,
  `persona` char(16) NOT NULL,
  `data_infezione` date NOT NULL,
  `stato` set('asintomatico','lieve','grave','critico','guarito') NOT NULL,
  FOREIGN KEY (`persona`) REFERENCES  `persone`(`cod_fiscale`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `deceduti` (
  `id_infezione` int(11) NOT NULL,
  `data_decesso` date NOT NULL,
  PRIMARY KEY(`id_infezione`,`data_decesso`),
  FOREIGN KEY (`id_infezione`) REFERENCES  `infetti`(`id_infezione`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `guariti` (
  `id_infezione` int(11) NOT NULL,
  `data_guarigione` date NOT NULL,
   PRIMARY KEY(`id_infezione`,`data_guarigione`),
   FOREIGN KEY (`id_infezione`) REFERENCES  `infetti`(`id_infezione`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
