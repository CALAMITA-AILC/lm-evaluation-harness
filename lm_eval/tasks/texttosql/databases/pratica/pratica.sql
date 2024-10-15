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

INSERT INTO `amministrativo` VALUES (0,'Abel','Spencer','1985-02-02','2018-01-14','New Mostropolis',11);
INSERT INTO `amministrativo` VALUES (1,'Erick','Leonard','1976-12-30','2019-12-26','Albinoleffe',4);
INSERT INTO `amministrativo` VALUES (2,'Janice','Montes','1953-05-29','2020-02-09','Albinoleffe',4);
INSERT INTO `amministrativo` VALUES (3,'Gretchen','Proctor','1988-08-26','2018-12-02','New Mostropolis',2);
INSERT INTO `amministrativo` VALUES (4,'Lawanda','Velazquez','1953-03-22','2020-02-08','Garoilles',11);
INSERT INTO `amministrativo` VALUES (5,'Robbie','Wilkins','1977-12-06','2012-05-25','Nocera Inferiore',4);
INSERT INTO `amministrativo` VALUES (6,'Carla','Randall','1986-11-15','2017-06-23','Garoilles',1);
INSERT INTO `amministrativo` VALUES (7,'Heath','Dickson','1977-08-10','2013-02-16','Nocera Inferiore',11);
INSERT INTO `amministrativo` VALUES (8,'Kendra','Rodgers','1960-11-29','2010-11-11','Garoilles',7);
INSERT INTO `amministrativo` VALUES (9,'Brandie','Finley','1990-11-30','2015-06-26','Garoilles',0);
INSERT INTO `amministrativo` VALUES (10,'Rose','Escobar','1981-09-11','2010-04-10','New Mostropolis',7);

INSERT INTO `assistente` VALUES (0,'Latisha','Mathews','1963-06-23','2005-05-22','2005-05-22');
INSERT INTO `assistente` VALUES (1,'Diana','Ortiz','1991-04-08','2016-07-04','2016-07-04');
INSERT INTO `assistente` VALUES (2,'Benny','Rojas','1999-10-01','2019-11-25','2019-11-25');
INSERT INTO `assistente` VALUES (3,'Evan','Daniels','1987-01-27','2014-10-30','2014-10-30');
INSERT INTO `assistente` VALUES (4,'Eva','Bullock','1962-01-06','2004-10-21','2004-10-21');
INSERT INTO `assistente` VALUES (5,'Dale','Erickson','1961-11-19','2004-10-02','2004-10-02');
INSERT INTO `assistente` VALUES (6,'Greg','Houston','1986-02-19','2014-06-15','2014-06-15');
INSERT INTO `assistente` VALUES (7,'Alicia','Phillips','1957-05-29','2002-12-18','2002-12-18');
INSERT INTO `assistente` VALUES (8,'Arnold','Connell','1980-03-30','2012-02-05','2012-02-05');
INSERT INTO `assistente` VALUES (11,'Norman','Marquez','1995-07-02','2018-03-14','2018-03-14');
INSERT INTO `assistente` VALUES (12,'Charlie','Brennan','1979-04-14','2011-09-18','2011-09-18');

INSERT INTO `pratica` VALUES (22,'2020-02-17','2020-09-10',5,1);
INSERT INTO `pratica` VALUES (23,'2020-05-14','2020-10-14',3,2);
INSERT INTO `pratica` VALUES (24,'2020-02-23','2020-11-07',7,3);
INSERT INTO `pratica` VALUES (26,'2020-01-09','2020-01-21',4,5);
INSERT INTO `pratica` VALUES (27,'2020-01-01','2020-04-10',3,2);
INSERT INTO `pratica` VALUES (29,'2020-01-04','2020-12-13',7,7);
INSERT INTO `pratica` VALUES (32,'2020-02-08','2020-12-29',2,11);
INSERT INTO `pratica` VALUES (33,'2020-04-02','2020-09-03',8,12);
INSERT INTO `pratica` VALUES (35,'2020-04-14','2020-05-02',9,1);
INSERT INTO `pratica` VALUES (36,'2020-05-30','2020-06-23',6,5);
