from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import psycopg2
import numpy as np
import json

model_id = 'naver/splade-cocondenser-ensembledistil'
 
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)

try:
    connection = psycopg2.connect(
        host="ip", #put ip here
        database="database",
        user="database",
        password="password", #password
        port="port"
    )
    cursor = connection.cursor()

    search_query = input("Search: ")

    tokens = tokenizer(
            search_query, return_tensors='pt',
            padding=True, truncation=True
            )
    output = model(**tokens)

    vecs = torch.max(
            torch.log(1 + torch.relu(output.logits)) * tokens.attention_mask.unsqueeze(-1), dim=1
        )[0].squeeze()
        

    #extract non-zero positions
    cols = vecs.nonzero().squeeze().cpu().tolist()
    print(len(cols))

    #extract the non-zero values
    weights = vecs[cols].cpu().tolist()
    #use to create a dictionary of token ID to weight
    sparse_dict = dict(zip(cols, weights))
    
    sparse_vector_string = f"'{sparse_dict}/30522'"
    
    query = f"SELECT columns, 1 - (column <=> {sparse_vector_string}) AS cosine_similarity FROM table ORDER BY cosine_similarity DESC LIMIT 2;"

    cursor.execute(query)

    results = cursor.fetchall()
    
    response_items = []
    for result in results:
        title, description, similarity = result

        response_item = {
            "title": title,
            "description": description if description else None,
            "cosineSimilarity": similarity,
        }
        
        response_items.append(response_item)

    response = {
        "isSuccess": True,
        "response": response_items,
        "errors": None
    }

    json_response = json.dumps(response, indent=2)
    print(json_response)


except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL", error)
 
finally:
    # Close the cursor and connection
    if connection:
        cursor.close()
        connection.close()