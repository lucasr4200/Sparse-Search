from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import psycopg2
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
import HitRateQueriesAndGroundTruth

model_id = 'naver/splade-cocondenser-ensembledistil'
 
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)

def sparseVectorSearch(cursor, search_query, desired_result):
    success = 0
    tokens = tokenizer(
            search_query, return_tensors='pt',
            padding=True, truncation=True
            )
    output = model(**tokens)

    vecs = torch.max(
            torch.log(1 + torch.relu(output.logits)) * tokens.attention_mask.unsqueeze(-1), dim=1
        )[0].squeeze()
        

    # extract non-zero positions
    cols = vecs.nonzero().squeeze().cpu().tolist()
    print(len(cols))

    # extract the non-zero values
    weights = vecs[cols].cpu().tolist()
    # use to create a dictionary of token ID to weight
    sparse_dict = dict(zip(cols, weights))
    
    sparse_vector_string = f"'{sparse_dict}/30522'"

    query = f"SELECT column, 1 - (column <=> {sparse_vector_string}) AS cosine_similarity FROM table ORDER BY cosine_similarity DESC LIMIT 2;"
    cursor.execute(query)

    results = cursor.fetchall()

    for row in results:
        print("str(row): ", str(row))
        if str(row) == desired_result:
            print("Success for sparse! Desired title was found")
            success += 1
        

    print("success: ", success)
    return success


def hitRate10Sparse(cursor):

    total_successes = 0
    #use sparse vector search first
    loop_number = 1
    for test in range(len(HitRateQueriesAndGroundTruth.QueriesAndTruths)):
        success = sparseVectorSearch(cursor, HitRateQueriesAndGroundTruth.QueriesAndTruths[test][0], 
                           HitRateQueriesAndGroundTruth.QueriesAndTruths[test][1])
        
        total_successes = total_successes + success
        print(total_successes, " / ", loop_number)
        loop_number += 1
    print("final sparse hitrate: ", total_successes, " / ", loop_number)

def KeyWordSearch(cursor, search_query, desired_result):

    success = 0

    # search_query = search_query.replace(" ", " & ")

    query = f"SELECT column FROM table where column @@ phraseto_tsquery('english', '{search_query}') ORDER BY id LIMIT 10;"
    cursor.execute(query)
    results = cursor.fetchall()

    print("search_query: ", search_query)
    for row in results:
        print(str(row))
        if str(row) == desired_result:
            print("----------")
            print(str(row))
            print("Success for KeyWord! Desired title was found")
            success += 1

    return success

def hitRate10KeyWord(cursor):

    total_successes = 0
    #use keyword search next
    loop_number = 0
    loop_number = 0
    for test in range(len(HitRateQueriesAndGroundTruth.QueriesAndTruths)):
        success = KeyWordSearch(cursor, HitRateQueriesAndGroundTruth.QueriesAndTruths[test][0], 
                           HitRateQueriesAndGroundTruth.QueriesAndTruths[test][1])
        
        loop_number += 1
        total_successes = total_successes + success
        print(total_successes, " / ", loop_number)
    print("final keyword hitrate: ", total_successes, " / ", loop_number)

try:
    connection = psycopg2.connect(
        host="ip", #put ip here
        database="database",
        user="database",
        password="password", #password
        port="port"
    )

    cursor = connection.cursor()

    hitRate10Sparse(cursor)

    #hitRate10KeyWord(cursor)

except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL", error)
 
finally:
    # Close the cursor and connection
    if connection:
        cursor.close()
        connection.close()