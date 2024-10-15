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

insert into Sede (nome, nazione, comune) values ( 'Universita Sapienza', 'Italia', 'Roma');
insert into Sede (nome, nazione, comune) values ( 'Fiera', 'Italia', 'Roma');
insert into Sede (nome, nazione, comune) values ( 'OpenHouse', 'Germany', 'Aub');
insert into Sede (nome, nazione, comune) values ( 'Boston place', 'USA', 'Boston');
insert into Sede (nome, nazione, comune) values ( 'NASAHQ', 'USA', 'Washington');
insert into Sede (nome, nazione, comune) values ( 'ESAHQ', 'France', 'Paris');
insert into Sede (nome, nazione, comune) values ( 'TIMHQ', 'Italia', 'Milano');
insert into Sede (nome, nazione, comune) values ( 'ComuneZagarolo', 'Italia', 'Zagarolo');
insert into Sede (nome, nazione, comune) values ( 'CloudHQ', 'USA', 'NewYork');

insert into Societa(nomeSocieta, sede_legale, contattoTelefonico) values ('NASA'      , 'NASAHQ', 444);
insert into Societa(nomeSocieta, sede_legale, contattoTelefonico) values ('ESA'       , 'ESAHQ', 445  );
insert into Societa(nomeSocieta, sede_legale) values ('TIM'       , 'TIMHQ'  );
insert into Societa(nomeSocieta, sede_legale) values ('ZAGAROLO'  , 'ComuneZagarolo'  );
insert into Societa(nomeSocieta, sede_legale) values ('CloudSigma', 'CloudHQ'  );
insert into Societa(nomeSocieta, sede_legale) values ('Sapienza'  ,  'Universita Sapienza'  );
insert into Societa(nomeSocieta, sede_legale) values ('Sponsorizzo'  ,  'OpenHouse'  );

insert into Hackathon(ID, nome_evento, starting_date, ending_date, organizzatore) values (1, 'SpaceApps', '2016-10-19', '2016-10-20','NASA'        );
insert into Hackathon(ID, nome_evento, starting_date, ending_date, organizzatore) values (2, 'HACKBOX'  , '2019-01-01', '2019-01-05','ZAGAROLO'    );
insert into Hackathon(ID, nome_evento, starting_date, ending_date, organizzatore) values (3, 'TEST'     , '2012-06-13', '2012-06-14','TIM'         );
insert into Hackathon(ID, nome_evento, starting_date, ending_date, organizzatore) values (4, 'North'    , '1999-10-12', '1999-10-13','CloudSigma'  );
insert into Hackathon(ID, nome_evento, starting_date, ending_date, organizzatore) values (5, 'SpaceApps', '2013-10-16', '2013-10-20','NASA'        );
insert into Hackathon(ID, nome_evento, starting_date, ending_date, organizzatore) values (6, 'NorthZ'    , '2017-10-12', '2017-10-14','CloudSigma'  );

insert into Avere_luogo (nome_sede, IDevento) values ('Boston place' , 1);
insert into Avere_luogo (nome_sede, IDevento) values ('Universita Sapienza' , 1);
insert into Avere_luogo (nome_sede, IDevento) values ('OpenHouse', 2);
insert into Avere_luogo (nome_sede, IDevento) values ('Fiera', 3);
insert into Avere_luogo (nome_sede, IDevento) values ('OpenHouse', 4);
insert into Avere_luogo (nome_sede, IDevento) values ('Universita Sapienza', 4);
insert into Avere_luogo (nome_sede, IDevento) values ('Boston place', 5);