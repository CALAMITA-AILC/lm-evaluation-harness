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

insert into Artista (ID,movimento,nome,cognome,nazione, nome_darte) values (1,'Impressionismo','Claude','Monet','Francia',NULL);
insert into Artista (ID,movimento,nome,cognome,nazione, nome_darte) values (2,'Impressionismo','Paul','Cezanne','Francia',NULL);
insert into Artista (ID,movimento,nome,cognome,nazione, nome_darte) values (3,'Impressionismo','Edgar','Degas','Francia',NULL);
insert into Artista (ID,movimento,nome,cognome,nazione, nome_darte) values (4,'Impressionismo','Max','Liebermann','Germania',NULL);
insert into Artista (ID,movimento,nome,cognome,nazione, nome_darte) values (5,'Espressionismo','Ernst Ludwig','Kirchner','Germania',NULL);
insert into Artista (ID,movimento,nome,cognome,nazione, nome_darte) values (6,'Espressionismo','Henri','Matisse','Francia',NULL);
insert into Artista (ID,movimento,nome,cognome,nazione, nome_darte) values (7,'Espressionismo','Edvard','Munch','Danimarca',NULL);
insert into Artista (ID,movimento,nome,cognome,nazione, nome_darte) values (8,'Espressionismo','Ludwig','Meidner','Francia',NULL);
insert into Artista (ID,movimento,nome,cognome,nazione, nome_darte) values (9,'Realismo','Giuseppe','Pelizza Da Volpedo','Italia',NULL);
insert into Artista (ID,movimento,nome,cognome,nazione, nome_darte) values (10,'Realismo','Gustave','Courbet','Francia',NULL);

insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (1,'Impression, soleil levant',1872,"48x63","olio su tela", "Musee Marmottan Monet");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (2,'Le Bateau-atelier',1876,"72x60","olio su tela", "Barnes Foundation");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (3,'La rue de l épicerie à Rouen',1892,"72x60","olio su tela", "Barnes Foundation");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (4,'Jardin à Sainte-Adresse',1867,"72x60","olio su tela", "Barnes Foundation");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (5,'Montagne Sainte-Victoire',1906,"72x60","olio su tela", "Barnes Foundation");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (6,'Madame Cézanne',1894,"72x60","olio su tela", "Barnes Foundation");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (7,'La famille Bellelli',1867,"72x60","olio su tela", "Barnes Foundation");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (8,'L orchestre',1868,"72x60","olio su tela", "Barnes Foundation");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (9,'Berliner Straßenszene',1913,"72x60","olio su tela", "Barnes Foundation");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (10,'Apokalyptische Landschaft',1912,"72x60","olio su tela", "Barnes Foundation");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (11,'Zwei Reiter am Strand',1901,NULL,"olio su tela", "Barnes Foundation");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (12,'Women Plucking Geese',1872,"72x60","olio su tela", "Barnes Foundation");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (13,'Le bonheur de vivre',1906,"72x60","olio su tela", "Barnes Foundation");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (14,'Les Poissons Rouges',1912,"72x60","olio su tela", "Barnes Foundation");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (15,'Nu bleu',1907,"72x60","olio su tela", "Barnes Foundation");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (16,'Death in the Sickroom',1893,"72x60","olio su tela", "Barnes Foundation");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (17,'Girls on the pier',1902,"72x60",NULL, "Barnes Foundation");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (19,'Det syke barn',1886,"72x60","olio su tela", "Barnes Foundation");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (20,'Il quarto stato',1901,"72x60","olio su tela", "Barnes Foundation");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (21,'Membra Stanche',1907,"72x60","olio su tela", "Barnes Foundation");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (22,'Prato Fiorito',1903,"72x60","olio su tela", "Barnes Foundation");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (23,'Il Sole',1904,"72x60","olio su tela", "Barnes Foundation");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (24,'Un enterrement à Ornans',1850,"72x60","olio su tela", "Barnes Foundation");
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (25,'Le Désespéré',1845,"72x60","olio su tela", NULL);
insert into Opera (ID,nome,anno_pubblicazione,dimensioni,tecnica,id_museo) values (26,'les casseurs de pierre',1849,"72x60","olio su tela", "Barnes Foundation");

insert into Crea (opera, artista,committente) values (1, 1, NULL);
insert into Crea (opera, artista,committente) values (2, 1, NULL);
insert into Crea (opera, artista,committente) values (3, 1, NULL);
insert into Crea (opera, artista,committente) values (4, 1, NULL);
insert into Crea (opera, artista,committente) values (5, 2, NULL);
insert into Crea (opera, artista,committente) values (6, 2, NULL);
insert into Crea (opera, artista,committente) values (7, 3, NULL);
insert into Crea (opera, artista,committente) values (8, 3, NULL);
insert into Crea (opera, artista,committente) values (9, 5, NULL);
insert into Crea (opera, artista,committente) values (10, 8, NULL);
insert into Crea (opera, artista,committente) values (11, 4, NULL);
insert into Crea (opera, artista,committente) values (12, 4, NULL);
insert into Crea (opera, artista,committente) values (13, 6, 'Ščukin');
insert into Crea (opera, artista,committente) values (14, 6, NULL);
insert into Crea (opera, artista,committente) values (15, 6, NULL);
insert into Crea (opera, artista,committente) values (16, 7, NULL);
insert into Crea (opera, artista,committente) values (17, 7, NULL);
insert into Crea (opera, artista,committente) values (19, 7, NULL);
insert into Crea (opera, artista,committente) values (20, 9, NULL);
insert into Crea (opera, artista,committente) values (21, 9, NULL);
insert into Crea (opera, artista,committente) values (22, 9, NULL);
insert into Crea (opera, artista,committente) values (23, 9, NULL);
insert into Crea (opera, artista,committente) values (24, 10, NULL);
insert into Crea (opera, artista,committente) values (25, 10, NULL);
insert into Crea (opera, artista,committente) values (26, 10, NULL);






