import itertools
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import psycopg2
    
#load model
model = AutoModelForMaskedLM.from_pretrained("opensearch-project/opensearch-neural-sparse-encoding-v1")
tokenizer = AutoTokenizer.from_pretrained("opensearch-project/opensearch-neural-sparse-encoding-v1")


'''
This file is for testing a new model other than splade
'''

def text_to_sparse_vector(text, total_length):

        #Tokenize the text into words or characters (change this if needed)
        tokens = text.split()
        
        #Count occurrences of each token
        token_counts = defaultdict(int)
        for token in tokens:
            token_counts[token] += 1
        
        #Create the sparse vector representation
        sparse_vector = {i + 1: count for i, (token, count) in enumerate(token_counts.items())}
        
        #Format sparse vector as string
        sparse_vector_str = "{" + ",".join(f"{k}:{v}" for k, v in sparse_vector.items()) + "}"
        
        return f"{sparse_vector_str}/{total_length}"

try:
    connection = psycopg2.connect(
        host="ip", #put ip here
        database="database",
        user="database",
        password="password", #password
        port="port"
    )

    cursor = connection.cursor()

    # search_query = input("Search: ")

    from collections import defaultdict

    # Example usage
    text = "this is a query"
    vector_length = 30522# Using the number of tokens as the total length
    sparse_vector_string = f"'{text_to_sparse_vector(text, vector_length)}'"
    #print(sparse_vector_string)

    query = f"SELECT reference, title, description, id, last_modified, 1 - (column <=> {sparse_vector_string}) AS cosine_similarity FROM table ORDER BY cosine_similarity DESC LIMIT 2;"

    cursor.execute(query)

    results = cursor.fetchall()

    for row in results:
         print(row)

except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL", error)
 
finally:
    # Close the cursor and connection
    if connection:
        cursor.close()
        connection.close()