CREATE DATABASE IF NOT EXISTS voli;

USE voli;

create table cliente(
num_passaporto varchar(8) primary key,
nome varchar(20) not null,
cognome varchar(25) not null,
gender enum('F','M') not null,
naz varchar(2) not null,
data date not null
);

create table aereo(
cod_aereo varchar(6) primary key,
modello_di_aereo varchar(20) not null,
num_posti int unsigned not null,
comp varchar(2) not null
);

create table volo(
numero_volo varchar(8) primary key,
data datetime not null,
p_a enum('part','arr') not null,
comp varchar(2) not null,
aereo varchar(6) not null,
aeroporto varchar(3) not null
);

create table biglietto(
num_b int unsigned primary key auto_increment,
volo varchar(8) not null,
num_passaporto varchar(8) not null,
costo_euro int unsigned not null,
posto varchar(3) not null,
foreign key (volo) references volo (numero_volo) on delete cascade,
foreign key (num_passaporto) references cliente (num_passaporto) on delete cascade
);

insert into cliente values ('SD3245GH','EMILIA','SOLOMON','F','PG','1978-8-18');
insert into cliente values ('UD0254LC','FRANCA','DOGLIO','F','MM','1942-3-11');
insert into cliente values ('VZ6434NL','ROSINA','GRISHAJ','F','KP','2004-11-13');
insert into cliente values ('XT7754MX','ROSETTA','LI CAUSI','F','UG','1973-8-22');
insert into cliente values ('NK6051PK','ADA','SIMONI','F','VU','1939-6-6');
insert into cliente values ('UT0717CL','SEFORA','SONCIN','F','BJ','1921-8-25');
insert into cliente values ('AJ0662XX','DANA','LURDHI','F','GA','1949-6-1');
insert into cliente values ('RY3443UR','TRISHA ','SOLITO','F','UA','1995-5-5');
insert into cliente values ('ZV0935QU','ZARA','AMBROGIO','F','EC','2020-6-21');
insert into cliente values ('BT8596BV','CANDIDA','LERA','F','GY','1956-6-13');
insert into cliente values ('OG0371CM','MARGARITA','MANNINI','F','CA','1984-6-15');
insert into cliente values ('KR5485TA','MADDALENA','ELEZI','F','VU','1945-5-10');
insert into cliente values ('SJ3363DQ','STACY','SAIA','F','ET','1995-8-16');
insert into cliente values ('FO0443DB','VERA','MARANZANA','F','ZR','1948-12-4');
insert into cliente values ('JV5711BN','JOSEPHINE','PACIFICO','F','BT','1927-3-22');
insert into cliente values ('YU0369JL','ROSARIA','SLAMA','F','HU','2018-2-7');
insert into cliente values ('GM0744DQ','DORA','VENDRAME','F','MQ','1997-10-8');
insert into cliente values ('OO5986KV','FEMKE','BELLESINI','F','ID','2011-9-25');
insert into cliente values ('IH2497SV','CARMELA','MANFRIN','F','SL','1984-7-14');
insert into cliente values ('YD3036HH','MILVIA','FORNACA','F','ZW','1979-6-1');
insert into cliente values ('ZX5622RH','FILIPPO ','HALILI','M','TZ','1986-12-1');
insert into cliente values ('PT0110LU','BEDA ','HANXHARI','M','ER','1954-7-23');
insert into cliente values ('IB3661TM','PIERGIORGIO ','CAMPAGNA','M','IR','1981-11-4');
insert into cliente values ('FU5238DP','NIKITA ','MARCHESE','M','AS','2008-1-12');
insert into cliente values ('TR7347KA','SWAN ','SARACCO','M','GP','1978-7-15');
insert into cliente values ('NM0476EA','GIONATAN ','SALAH','M','PL','1915-10-10');
insert into cliente values ('JS3785DQ','SVEVO ','SANTANGELO','M','BA','2012-11-25');
insert into cliente values ('YQ4170SR','GUERRINO ','BIGLIANI','M','BA','1956-11-10');
insert into cliente values ('KO3297HC','FELICE ','CESTARI','M','TM','1928-4-24');
insert into cliente values ('JS0710TT','ROSSANO','NATTA','M','KH','1950-10-14');
insert into cliente values ('ZZ3130KS','LIBORIO ','CUMINO','M','PH','1953-5-27');
insert into cliente values ('FU4080ZG','AMBROGIO ','FERRONATO','M','BR','1983-12-22');
insert into cliente values ('RA3665UZ','PRIMO ','GNAN','M','MR','1927-9-17');
insert into cliente values ('YR3623QP','RAMON ','LUKA','M','GI','1989-1-4');
insert into cliente values ('GP8404DB','ALBERICO ','PIO','M','BG','1994-6-14');
insert into cliente values ('UC8755GH','TERENZIO ','PORZIO','M','CF','2020-6-14');
insert into cliente values ('LU7754NI','ZACCARIA','IPPOLITO','M','HU','1997-5-17');
insert into cliente values ('YU5946CO','LUKE ','PIOTTI','M','UG','1995-3-18');
insert into cliente values ('DF3266KZ','ARCADIO','ZERBINATI','M','MC','1938-12-3');
insert into cliente values ('XN6979MR','AUDENICO','ILENGO','M','SI','1966-9-16');

insert into aereo values ('VF5766','Embraer 175',88,'ZY');
insert into aereo values ('XO1615','Airbus A330-200',251,'AC');
insert into aereo values ('SX3059','Boeing 777-300ER',408,'JP');
insert into aereo values ('DL7819','Airbus A380-800',484,'BP');
insert into aereo values ('HX8038','Boeing 737-700',132,'5D');
insert into aereo values ('QY4009','Airbus A330-300',292,'UU');
insert into aereo values ('GU6936','Boeing 777-300ER',408,'UU');
insert into aereo values ('EL7368','Airbus A380-800',484,'SM');
insert into aereo values ('KE6277','Airbus A330-200',268,'KC');
insert into aereo values ('WU2191','Boeing 777-200ER',316,'G9');
insert into aereo values ('RQ0069','Airbus A330-300',297,'ZY');
insert into aereo values ('KO6718','Boeing 737-900',184,'AW');
insert into aereo values ('FG6467','Boeing 777-300ER',408,'AH');
insert into aereo values ('LU6266','Airbus A330-200',251,'VW');
insert into aereo values ('KL1654','Boeing 737-700',132,'ZI');

insert into volo values ('IH0576','2020-6-13 1:5:00','arr','UU','EL7368','POM');
insert into volo values ('YT7879','2019-8-20 0:35:00','arr','AC','RQ0069','YBG');
insert into volo values ('PD9154','2020-5-28 16:50:00','arr','5D','WU2191','POM');
insert into volo values ('DM5487','2020-9-21 21:45:00','part','A3','KE6277','EBB');
insert into volo values ('IY1585','2020-8-28 11:35:00','arr','A3','KE6277','EBB');
insert into volo values ('FK4521','2020-8-14 20:5:00','arr','A3','QY4009','REK');
insert into volo values ('WO9622','2019-11-10 5:5:00','part','G9','LU6266','YQB');
insert into volo values ('DE3097','2020-9-1 15:0:00','arr','G9','KL1654','YOW');
insert into volo values ('CN2817','2019-4-13 20:0:00','arr','2J','XO1615','REK');
insert into volo values ('BH9973','2019-4-21 0:50:00','part','KC','HX8038','REK');
insert into volo values ('YC7037','2020-9-17 19:50:00','arr','SM','FG6467','EBB');
insert into volo values ('FG8970','2019-8-12 0:15:00','arr','JP','VF5766','POM');
insert into volo values ('LG5669','2020-8-10 20:40:00','part','JP','XO1615','YOW');
insert into volo values ('PF0601','2019-8-23 12:5:00','arr','2J','VF5766','UAK');
insert into volo values ('FG2724','2019-7-27 4:25:00','part','BP','KL1654','YHZ');
insert into volo values ('EW4308','2020-6-10 15:5:00','arr','ZY','SX3059','UAK');
insert into volo values ('AZ1507','2019-12-25 2:45:00','part','ZY','QY4009','POM');
insert into volo values ('BS4001','2019-12-5 3:25:00','part','KC','WU2191','REK');
insert into volo values ('BU1634','2020-2-11 23:25:00','part','TY','EL7368','EBB');
insert into volo values ('MH2747','2019-7-11 21:30:00','arr','AR','RQ0069','YOW');

insert into biglietto(volo,num_passaporto,costo_euro,posto) values ('YC7037','KR5485TA','475','A29');
insert into biglietto(volo,num_passaporto,costo_euro,posto) values ('CN2817','UC8755GH','1343','A07');
insert into biglietto(volo,num_passaporto,costo_euro,posto) values ('BU1634','UD0254LC','387','C26');
insert into biglietto(volo,num_passaporto,costo_euro,posto) values ('BH9973','UT0717CL','328','D05');
insert into biglietto(volo,num_passaporto,costo_euro,posto) values ('DE3097','OO5986KV','193','F09');
insert into biglietto(volo,num_passaporto,costo_euro,posto) values ('IH0576','AJ0662XX','1459','E09');
insert into biglietto(volo,num_passaporto,costo_euro,posto) values ('BH9973','BT8596BV','1948','E08');
insert into biglietto(volo,num_passaporto,costo_euro,posto) values ('CN2817','GM0744DQ','629','A10');
insert into biglietto(volo,num_passaporto,costo_euro,posto) values ('AZ1507','DF3266KZ','1815','A26');
insert into biglietto(volo,num_passaporto,costo_euro,posto) values ('CN2817','FO0443DB','1446','C13');
insert into biglietto(volo,num_passaporto,costo_euro,posto) values ('FG2724','FU4080ZG','1036','A08');
insert into biglietto(volo,num_passaporto,costo_euro,posto) values ('IY1585','GP8404DB','930','A13');
insert into biglietto(volo,num_passaporto,costo_euro,posto) values ('BU1634','NK6051PK','878','C06');
insert into biglietto(volo,num_passaporto,costo_euro,posto) values ('AZ1507','NK6051PK','1808','A01');
insert into biglietto(volo,num_passaporto,costo_euro,posto) values ('FK4521','NK6051PK','812','A16');
