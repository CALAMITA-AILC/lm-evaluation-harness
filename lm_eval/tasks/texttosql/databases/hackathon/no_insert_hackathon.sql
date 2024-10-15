CREATE DATABASE IF NOT EXISTS hackathon;

USE hackathon;

CREATE TABLE Societa(
    contattoTelefonico varchar(20),
    sede_legale varchar(20) not NULL,
    nomeSocieta varchar(20) not NULL unique,
    PRIMARY KEY(nomeSocieta)
) ENGINE=INNODB;

CREATE TABLE Hackathon(
    ID int unsigned not NULL PRIMARY KEY,
    nome_evento varchar(20) not NULL,
    starting_date DATE not NULL,
    ending_date DATE not NULL,
    organizzatore varchar(20) not NULL,
    FOREIGN KEY(organizzatore) REFERENCES Societa(nomeSocieta)
) ENGINE=INNODB;

CREATE TABLE Sede(
    nome varchar(20) not NULL,
    nazione varchar(20) not NULL,
    comune varchar(20) not NULL,
    PRIMARY KEY(nome)
) ENGINE=INNODB;

CREATE TABLE Avere_luogo(
    nome_sede varchar(20) not NULL,
    IDevento int unsigned not NULL,
    PRIMARY KEY(nome_sede, IDevento),
    FOREIGN KEY(nome_sede) REFERENCES Sede(nome),
    FOREIGN KEY(IDevento) REFERENCES Hackathon(ID)
) ENGINE=INNODB;