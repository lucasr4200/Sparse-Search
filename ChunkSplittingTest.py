from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import psycopg2


model_id = 'naver/splade-cocondenser-ensembledistil'
 
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)
def content_to_chunks_using_tokenizer(content_text):
   splitter = SentenceTransformersTokenTextSplitter(model_name="naver/splade-cocondenser-ensembledistil",tokens_per_chunk=500,chunk_overlap=5)
   text_chunks =splitter.split_text(text=content_text)
   return text_chunks

try:
    connection = psycopg2.connect(
        host="ip", #put ip here
        database="database",
        user="database",
        password="password", #password
        port="port"
    )

    cursor = connection.cursor()


    for i in range(2):
    #get items from table here
        print("loop number: ", i)
        
        cursor.execute(f"SELECT column, id FROM table ORDER BY id LIMIT 1 OFFSET {i}")
        text_rows = cursor.fetchall()

        tuple_text_rows = (f"{text_rows[0][0]}")
        
        content_chunks = content_to_chunks_using_tokenizer(tuple_text_rows)

        for chunk in content_chunks:
            
            tokens = tokenizer(
            text_rows[0][0], return_tensors='pt',
            padding=True, truncation=True
            )
            
            output = model(**tokens)
            

            #aggregate the token-level vecs and transform to sparse
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

            cursor.execute("INSERT INTO table (chunk, column, foreignkey) VALUES (%s, %s, %s);", (str(chunk), f"{sparse_dict}/30522", str(text_rows[0][1])))
            
            connection.commit()

except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL", error)
 
finally:
    #Close the cursor and connection
    if connection:
        cursor.close()
        connection.close()