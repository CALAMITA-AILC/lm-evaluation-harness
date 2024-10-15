CREATE DATABASE IF NOT EXISTS galleria;

USE galleria;

CREATE TABLE IF NOT EXISTS Artista(
ID  int(11),
movimento  varchar(100),
nome  varchar(100),
cognome  varchar(100),
nazione  varchar(50),
nome_darte varchar(50),
PRIMARY KEY (ID)
);

CREATE TABLE IF NOT EXISTS Opera(
ID  int(11),
nome  varchar(255),
anno_pubblicazione  int(11),
dimensioni varchar(20),
tecnica varchar(50),
id_museo varchar(255),
PRIMARY KEY (ID)
);

CREATE TABLE IF NOT EXISTS Crea(
artista  int(11),
opera  int(11),
committente varchar(100),
PRIMARY KEY (artista,opera),
FOREIGN KEY (artista)   REFERENCES Artista(ID),
FOREIGN KEY (opera)   REFERENCES Opera(ID)
);