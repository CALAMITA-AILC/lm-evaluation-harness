CREATE DATABASE IF NOT EXISTS farma;

USE farma;

CREATE TABLE `farmaco` (
  `Codice` int(11) NOT NULL PRIMARY KEY,
  `Azienda` varchar(50) NOT NULL,
  `NomeCommerciale` varchar(80) NOT NULL,
  `Prezzo_EUR` float NOT NULL,
  `Posologia_ml` varchar(120) DEFAULT NULL,
  `IndicazioneTerapeutica` varchar(120) DEFAULT NULL,
  `PrincipioAttivo` varchar(50) NOT NULL,
  `regimeFornitura` varchar(5) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO `farmaco` (`Codice`, `Azienda`, `NomeCommerciale`, `Prezzo_EUR`, `Posologia_ml`, `IndicazioneTerapeutica`, `PrincipioAttivo`, `regimeFornitura`) VALUES (1085709, 'Bocca-Gioberti', 'TAREG', 25.78, 'Molestias suscipit repellat dignissimos blanditiis.', 'Quaerat exercitationem impedit expedita quia.', 'Valsartan', 'OSP'), (1892910, 'Monte-Contarini e figli', 'Stendra', 5.78, 'Ad dicta voluptas eveniet ab tempore.', 'Est nobis vitae quisquam quisquam nobis est.', 'Avanafil', 'RRL'), (2884363, 'Monte-Contarini e figli', 'VERMOX', 39.86, 'Eligendi placeat deleniti libero cupiditate laudantium.', 'Neque iste eligendi eos aspernatur qui cum.', 'Mebendazolo', 'RNR'), (3499406, 'Calbo-Mantegna Group', 'FLUXUM', 22.45, 'Velit magnam excepturi repellendus.', 'Corrupti earum doloribus cumque culpa dignissimos ab.', 'Parnaparina', 'OSP'), (3807950, 'Argurio e figli', 'capecitabina', 50.32, 'Recusandae labore tempora.', 'Animi ipsa commodi laudantium repellat praesentium.', 'Capecitabina', 'USPL'), (4227450, 'Bocca-Gioberti', 'ASACOL', 3.85, 'Modi reprehenderit nam aspernatur magni.', 'Cupiditate nobis illo modi repellendus ipsa doloremque.', 'Mesalazina', 'RRL'), (4367441, 'Procacci e figli', 'Seleparina', 51.09, 'Molestias molestiae nisi aut atque.', 'Aliquam quibusdam iste tempore.', 'Nadroparina', 'RNRL'), (4367442, 'Procacci e figli', 'tachipirina', 51.09, 'Molestias molestiae nisi aut atque.', 'Aliquam quibusdam iste tempore.', 'Paracetamolo', 'RNRL'), (6174201, 'Russo-Pizzo Group', 'mesalazina', 16.31, 'Ratione adipisci explicabo atque.', 'Magnam nulla omnis natus.', 'Mesalazina', 'RNRL'), (6217421, 'Procacci e figli', 'accusamusnon', 52.3, 'Illum eius numquam labore.', 'Alias dolore eveniet quas consequuntur.', 'Lidocaina', 'RNRL'), (6515781, 'Filippini, Casale e Prada Group.', 'Xadago', 48.05, 'Aspernatur iure itaque.', 'Esse vel beatae beatae est consequatur error eveniet.', 'Safinamide', 'RRL');

CREATE TABLE `medico` (
  `Matricola` varchar(13) NOT NULL PRIMARY KEY,
  `Nome` varchar(32) DEFAULT NULL,
  `Cognome` varchar(40) DEFAULT NULL,
  `recapito_tel` varchar(12) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO `medico` (`Matricola`, `Nome`, `Cognome`, `recapito_tel`) VALUES ('0010045998832', 'Romeo', 'Brunelleschi', '377403650423'), ('0022042359216', 'Fiamma', 'Monduzzi', '042104610994'), ('0032127900662', 'Tiziano', 'Bova', '031807613392'), ('0032529554825', 'Antonia', 'Zamorani', '322834893512'), ('0055861951113', 'Raffaele', 'Guariento', '076532263357');

CREATE TABLE `paziente` (
  `codice_fiscale` varchar(32) NOT NULL PRIMARY KEY,
  `nome_paz` varchar(32) DEFAULT NULL,
  `cognome_paz` varchar(38) DEFAULT NULL,
  `curante` varchar(13) DEFAULT NULL,
  FOREIGN KEY (`curante`) REFERENCES  `medico`(`Matricola`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO `paziente` (`codice_fiscale`, `nome_paz`, `cognome_paz`, `curante`) VALUES ('BCAQZN56P54C296S', 'Graziella', 'Trebbi', '0010045998832'), ('FBTLKJ79G08P636C', 'Gian', 'Botticelli', '0022042359216'), ('FWFODT03X92O941O', 'Marco', 'Maderna', '0010045998832'), ('IQJCCB09X08X155D', 'Loredana', 'Segni', '0055861951113'), ('ITGNFQ61W55D765I', 'Gioachino', 'Monteverdi', '0055861951113'), ('KWGVJZ03Z13F625U', 'Fernanda', 'Guidotti', '0022042359216'), ('NUHYDA15U72F533T', 'Gaspare', 'Bergoglio', '0032529554825'), ('PCOEGR29Y24L406G', 'Angelina', 'Casarin', '0010045998832'), ('QPQGMJ25D65S531A', 'Alfio', 'Tarantini', '0032127900662'), ('QTBZZP71X29S676B', 'Fausto', 'Tarchetti', '0032127900662'), ('TICIFP99Z06S200Z', 'Lidia', 'Gatto', '0055861951113'), ('VKTNOC70U62F178F', 'Nina', 'Bruno', '0032529554825'), ('XSUNVV38F35W974Y', 'Michele', 'Sgalambro', '0032529554825'), ('YAOFXT33P41F105P', 'Giulietta', 'Bixio', '0010045998832'), ('YJZHSE20R29A326J', 'Liana', 'Paruta', '0032127900662'), ('YQYQPQ12K74V211Q', 'Adriano', 'Tamburini', '0055861951113');

CREATE TABLE `ricetta` (
  `Identificativo` varchar(32) NOT NULL PRIMARY KEY,
  `Data_Prescrizione` date DEFAULT NULL,
  `CodiceFiscale` varchar(32) NOT NULL,
  `Matricola` varchar(13) DEFAULT NULL,
  `Cod_d` int(11) NOT NULL,
  FOREIGN KEY (`CodiceFiscale`) REFERENCES  `paziente`(`codice_fiscale`),
  FOREIGN KEY (`Matricola`) REFERENCES  `medico`(`Matricola`),
  FOREIGN KEY (`Cod_d`) REFERENCES  `farmaco`(`Codice`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO `ricetta` (`Identificativo`, `Data_Prescrizione`, `CodiceFiscale`, `Matricola`, `Cod_d`) VALUES ('14374', '2012-08-18', 'TICIFP99Z06S200Z', '0032127900662', 3499406), ('1649988', '1989-04-22', 'ITGNFQ61W55D765I', '0032127900662', 4227450), ('2097761', '2020-02-12', 'QTBZZP71X29S676B', '0055861951113', 4227450), ('3319407', '1999-01-15', 'QPQGMJ25D65S531A', '0010045998832', 6217421), ('536097', '1976-05-15', 'VKTNOC70U62F178F', '0055861951113', 4367441), ('6879423', '2018-01-27', 'QTBZZP71X29S676B', '0032529554825', 3807950), ('7553271', '2020-12-14', 'YQYQPQ12K74V211Q', '0032127900662', 1085709), ('8734728', '1991-02-01', 'PCOEGR29Y24L406G', '0032529554825', 3499406), ('8880192', '1993-03-11', 'YQYQPQ12K74V211Q', '0055861951113', 6515781), ('9530747', '1979-04-19', 'FBTLKJ79G08P636C', '0010045998832', 6174201);