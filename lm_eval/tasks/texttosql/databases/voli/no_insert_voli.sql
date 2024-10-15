CREATE DATABASE IF NOT EXISTS voli;

USE voli;

create table cliente(
passaporto varchar(8) primary key,
nome varchar(20) not null,
cognome varchar(20) not null,
sesso enum('F','M') not null,
naz varchar(2) not null,
data date not null
);

create table aereo(
cod_aereo varchar(6) primary key,
modello varchar(20) not null,
num_posti int unsigned not null,
comp varchar(2) not null
);

create table volo(
num_volo varchar(6) primary key,
data datetime not null,
p_a enum('part','arr') not null,
comp varchar(2) not null,
aereo varchar(6) not null,
aeroporto varchar(3) not null
);

create table biglietto(
num_b int unsigned primary key auto_increment,
volo varchar(6) not null,
passaporto varchar(8) not null,
costo int unsigned not null,
posto varchar(3) not null,
foreign key (volo) references volo (num_volo) on delete cascade,
foreign key (passaporto) references cliente (passaporto) on delete cascade
);
