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

INSERT INTO `persone` (`cognome`, `nome`, `sesso`, `nas_comune`, `data_nascita`, `cod_fiscale`, `res_comune`, `indirizzo`, `telefono`, `email`, `professione`, `stato_civile`) VALUES ('Bai', 'Benito', 'M', 'L752', '1989-03-22', 'BAIBNT89C22L752H', 'D006', 'Strada Consortile Chiaramonti, 293', '0832/845520', 'benito.bai@gmail.com', 'Anestesista', 'Separato'), ('Bai', 'Gino', 'M', 'H629', '1942-11-07', 'BAIGNI42S07H629I', 'I534', 'Via L.Beltrami, 108/a', '098/1071752', 'gino.bai@lycos.it', 'Angiologo', 'Coniugato'), ('Bau', 'Valeria', 'F', 'B748', '1935-05-19', 'BAUVLR07E59B748M', 'E548', 'Via L.Eredia, 1', '06/558795', 'vale.bau@gmail.it', 'Anestesista', 'Coniugata'), ('Abbà', 'Vladimiro', 'M', 'B465', '1934-05-23', 'BBAVDM10E23B465L', 'H294', 'Via V.Brocchi, 117', '0541/274523', 'vladimiro.abb@gmail.it', 'Infermiere Caposala', 'Separato'), ('Bebbo', 'Concetta', 'F', 'B396', '1956-03-06', 'BBBCCT03P55B396G', 'H327', 'Piazzale Broni, 21', '049/168785', 'concetta.bebbo@aruba.it', 'Anestesista', 'Nubile'), ('Bobbiese', 'Ruperto', 'M', 'E349', '1921-08-09', 'BBBRRT21M09E349W', 'H059', 'Via Bolama, 202', '0173/784639', 'rupe.bobb@hotmail.com', 'Microfonista', 'Coniugato'), ('Tornidi', 'Mario', 'M', 'E349', '1956-04-09', 'MNTRND75R18L813R', 'H059', 'Via Cordi, 4', '0173/784671', 'tornidi.mario@hotmail.com', 'Anestesista', 'Coniugato'), ('Di Bivio', 'Pino', 'M', 'B465', '1971-05-19', 'PNDBVN82M50B592Q', 'E548', 'Via L.Beltrami , 16b', '091/754272', 'pinodibiv22@gmail.com', 'Infermiere Caposala', 'Separato'), ('Melese', 'Marta', 'F', 'B465', '1990-11-19', 'MRTMLE29T71B649P', 'E548', 'Via Parisi , 213', '091/345621', 'marta.malese00@libero.it', 'Anestesista', 'Nubile'), ('Tegidi', 'Letizia', 'F', 'B465', '1945-11-19', 'LTTGDI09B45A429N', 'I534', 'Piazza Zarri , 7', '098/588634', 'letytegy45@outlook.com', 'Pensionata', 'Coniugata'), ('Nili', 'Filippo Carlo', 'M', 'H629', '1995-11-19', 'FLCLNI19R52E160C', 'H294', 'Via Parenzio 89', '0541/58004', 'filcarlonili27@gmail.com', 'Angiologo', 'Coniugato');

CREATE TABLE `infetti` (
  `id_infezione` int(11) NOT NULL PRIMARY KEY,
  `persona` char(16) NOT NULL,
  `data_infezione` date NOT NULL,
  `stato` set('asintomatico','lieve','grave','critico','guarito') NOT NULL,
  FOREIGN KEY (`persona`) REFERENCES  `persone`(`cod_fiscale`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO `infetti` (`id_infezione`, `persona`, `data_infezione`, `stato`) VALUES (1, 'MNTRND75R18L813R', '2020-03-01', 'lieve'), (2, 'PNDBVN82M50B592Q', '2020-03-02', 'grave'), (3, 'MRTMLE29T71B649P', '2020-03-02', 'asintomatico'), (4, 'LTTGDI09B45A429N', '2020-03-02', 'lieve'), (5, 'FLCLNI19R52E160C', '2020-03-02', 'grave'), (6, 'BAIBNT89C22L752H', '2020-03-09', 'grave'), (7, 'BAIGNI42S07H629I', '2020-03-30', 'lieve'), (8, 'BAUVLR07E59B748M', '2020-03-30', 'grave'), (9, 'BBAVDM10E23B465L', '2020-04-26', 'grave'), (10, 'LTTGDI09B45A429N', '2020-10-02', 'grave'), (11, 'BAIBNT89C22L752H', '2020-10-01', 'lieve');

CREATE TABLE `deceduti` (
  `id_infezione` int(11) NOT NULL,
  `data_decesso` date NOT NULL,
  PRIMARY KEY(`id_infezione`,`data_decesso`),
  FOREIGN KEY (`id_infezione`) REFERENCES  `infetti`(`id_infezione`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO `deceduti` (`id_infezione`, `data_decesso`) VALUES (3, '2020-04-28'), (8, '2020-04-18'), (6, '2020-03-19');

CREATE TABLE `guariti` (
  `id_infezione` int(11) NOT NULL,
  `data_guarigione` date NOT NULL,
   PRIMARY KEY(`id_infezione`,`data_guarigione`),
   FOREIGN KEY (`id_infezione`) REFERENCES  `infetti`(`id_infezione`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO `guariti` (`id_infezione`, `data_guarigione`) VALUES (1, '2020-03-18'), (2, '2020-04-19'), (4, '2020-04-23'), (5, '2020-04-28');
