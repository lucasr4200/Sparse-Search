from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import psycopg2


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

    for i in range(2):
    #get items from table here
        print("loop number: ", i)
        all_texts = []
        cursor.execute(f"SELECT column FROM table ORDER BY id LIMIT 1 OFFSET {i}")
        text_rows = cursor.fetchall()
        for texts in text_rows:
            all_texts.append(texts[0])

        tokens = tokenizer(
        all_texts, return_tensors='pt',
        padding=True, truncation=True
        )
        print("we make it before output")
        output = model(**tokens)
        print("we make it after output")

        # aggregate the token-level vecs and transform to sparse
        vecs = torch.max(
            torch.log(1 + torch.relu(output.logits)) * tokens.attention_mask.unsqueeze(-1), dim=1
        )[0].squeeze()


        #extract non-zero positions
        cols = vecs.nonzero().squeeze().cpu().tolist()
        print(len(cols))

        #extract non-zero values
        weights = vecs[cols].cpu().tolist()
        #use to create a dictionary of token ID to weight
        sparse_dict = dict(zip(cols, weights))

        cursor.execute("INSERT INTO items (text, embedding) VALUES (%s, %s);", (str(all_texts[0]), f"{sparse_dict}/30522"))

        #commented out so as not to make a new table every time
        #cursor.execute("CREATE TABLE items (id bigserial PRIMARY KEY, column1, column2 sparsevec(30522));")

        connection.commit()
 
except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL", error)
 
finally:
    # Close the cursor and connection
    if connection:
        cursor.close()
        connection.close()