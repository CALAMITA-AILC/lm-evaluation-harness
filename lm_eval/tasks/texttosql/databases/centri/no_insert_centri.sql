CREATE DATABASE IF NOT EXISTS centri;

USE centri;

CREATE TABLE IF NOT EXISTS centrodonazione(
id  bigint,
nome_cd  varchar(30),
indirizzo  varchar(100),
citta  varchar(20),
telefono  varchar(15),
email  varchar(50),
sitoWeb  varchar(40),
PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS centrosanitario(
id  bigint,
nome_csn  varchar(30),
indirizzo  varchar(100),
citta  varchar(20),
is_convenzionato boolean,
id_convenzione bigint,
portale  varchar(40),
telefono  varchar(15),
email  varchar(50),

PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS centrosportivo(
id  int,
nome_csp  varchar(30),
indirizzo  varchar(100),
citta  varchar(20),
is_agonistico boolean,
telefono  varchar(15),
email  varchar(50),
sitoWeb  varchar(40),
PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS convenzione_cd(
idDonazione  bigint,
idSportivo  int,
data_inizio_convenzione_d  date,
is_annuale boolean,
PRIMARY KEY (idDonazione,idSportivo),
FOREIGN KEY ( idDonazione)   REFERENCES centrodonazione(id),
FOREIGN KEY ( idSportivo)   REFERENCES centrosportivo(id)
);

CREATE TABLE IF NOT EXISTS convenzione_cs(
idSanitario  bigint,
idSportivo  int,
data_inizio_convenzione_s  date,
PRIMARY KEY (idSanitario,idSportivo),
FOREIGN KEY ( idSanitario)   REFERENCES centrosanitario(id),
FOREIGN KEY ( idSportivo)   REFERENCES centrosportivo(id)
);
