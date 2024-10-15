import pandas as pd
import sqlalchemy
import numpy as np
from sqlalchemy import create_engine, MetaData
from sqlalchemy import text
from sqlalchemy import exc

def read_queries(dataframe):    #takes empty dataframe as input

    database_list=["voli","coronavirus","recensioni","hackathon","bowling","galleria","pratica","farma","farmacia","centri"]

    data = []

    for db in database_list:

        # Read the gold-query files content
        
        file_gold = "db_dir/{}/gold.txt".format(db) #containes gold query for each lines
        with open(file_gold, 'r') as file:
            lines_gold = file.readlines()

        # Read the pred-query files content
        
        file_generate = "db_dir/{}/predict.txt".format(db) #containes predicted query for each lines
        with open(file_generate, 'r') as file:
            lines_generate = file.readlines()

        # Lists for memorize gold queries and predicetd queries
        queries_gold = []
        queries_generate = []

        # Parsing gold-queries file
        for line in lines_gold:
            if '\t' in line:  # Verifica se c'è un tab tra query e tabella
                query, _ = line.split('\t')
                queries_gold.append(query.strip())

        # Parsing pred-queries file
        for line in lines_generate:
            if '\t' in line:  # Verifica se c'è un tab tra query e tabella
                query, _ = line.split('\t')
                queries_generate.append(query.strip())

        for i in range(len(queries_gold)):
            data.append({
                'db': db,   #db_name or db_id
                'query_index': i,
                'query_gold': queries_gold[i],
                'query_generate': queries_generate[i],
                'result': np.nan
            })

        # Create new df
        df = pd.DataFrame(data)

    return df

df = pd.DataFrame({
    'db': pd.Series(dtype='str'),               # Col for db_name/db_id
    'query_index': pd.Series(dtype='int'),      # Col for query index (indexer for a single db)
    'query_gold': pd.Series(dtype='str'),       # Col for gold queries
    'query_generate': pd.Series(dtype='str'),   # Col for pred queries
    'result': pd.Series(dtype='int')            # Col for result (before evaluation it will contain NaN)
})

df = read_queries(df)

def connect_to_db(db_name):

        #url="mysql+mysqlconnector://user:Password@127.0.0.1:3306/name_of_db"
        url="mysql+mysqlconnector://john:password@127.0.0.1:3306/{}".format(db_name)

        # Connect to b
        engine = create_engine(url)

        connection = engine.connect()

        metadata = MetaData()

        metadata.reflect(bind=engine)

        return engine


##############################################################

def compare_results(cursor_result_1,cursor_result_2):   #compare gold query and generated query records (EXECUTION ACCURACY)
        
        #print(cursor_result_1.all() == cursor_result_2.all())

        if( cursor_result_1.all() == cursor_result_2.all() ):   #CursorResult.all() returns all result's rows in a list

            return 1

        else:

            return 0

def execute_queries():

    #MODIFY IF YOU GENERATED QUERIES JUST FOR SOME DATABASES
    database_list = ["voli","coronavirus","recensioni","hackathon","bowling","galleria","pratica","farma","farmacia","centri"]

    for db in database_list:

        #url="mysql+mysqlconnector://user:Password@127.0.0.1:3306/name_of_db"
        url="mysql+mysqlconnector://john:password@127.0.0.1:3306/{}".format(db)

        # Connect to db
        engine = create_engine(url)

        connection = engine.connect()

        metadata = MetaData()

        metadata.reflect(bind=engine)

        for i in range(len(df)):

            if(df.loc[i,"db"] == db):        

                #print(df.loc[i,"db"])

                query_gold = df["query_gold"].loc[i]

                query_gen = df["query_generate"].loc[i]

                #print(query_gold,query_gen)

                escaped_sql_gold = sqlalchemy.text(query_gold)

                escaped_sql_gen = sqlalchemy.text(query_gen)

                with engine.connect() as conn:

                    result_gold = conn.execute(escaped_sql_gold)

                    #gestire query sbagliate sintatticamente
                    try:
                        
                        result_gen = conn.execute(escaped_sql_gen)

                        df.loc[i,"result"] = compare_results(result_gold,result_gen)

                    except :

                        result_gen = []

                        print("-----ERROR-----")
                        print("error on query:",query_gen)
                        print("db:",db)
                        print("---------------")

                        df.loc[i,"result"] = 0


print(df)

execute_queries()


print("-"*120)
print(df)

#REPORT OVERALL ACCURACY

print("TOTAL ACCURACY:",df["result"].mean())


#REPORT ACCURACY BY DB

accuracy_by_db = df.groupby('db')['result'].mean().reset_index()

accuracy_by_db.columns = ['Database', 'Accuracy']

print("ACCURACY BY DB:")
print(accuracy_by_db.to_string(index=False))

